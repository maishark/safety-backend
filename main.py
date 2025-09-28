# main.py
import os, time, math, gzip, pickle, requests, joblib, numpy as np, networkx as nx
from io import BytesIO
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ------------------------------------------
# Config (env)
# ------------------------------------------
GRAPH_URL  = os.getenv("GRAPH_URL")    # e.g. https://.../graph_min.gpickle.gz
LGBM_URL   = os.getenv("LGBM_URL")     # e.g. https://.../best_model_lgbm.pkl
SCALER_URL = os.getenv("SCALER_URL")   # e.g. https://.../scaler.pkl
KMEANS_URL = os.getenv("KMEANS_URL")   # e.g. https://.../kmeans.pkl

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service key preferred (server only)

CRIME_RADIUS_METERS = float(os.getenv("CRIME_RADIUS_METERS", "200"))  # influence radius
CRIME_WINDOW_DAYS   = int(os.getenv("CRIME_WINDOW_DAYS", "5"))        # lookback window

assert GRAPH_URL and LGBM_URL and SCALER_URL and KMEANS_URL, "Missing model/graph URLs"
assert SUPABASE_URL and SUPABASE_KEY, "Missing Supabase URL/KEY"

# ------------------------------------------
# Helpers
# ------------------------------------------
def dl(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ------------------------------------------
# Load graph (gpickle.gz expected)
# ------------------------------------------
g_bytes = dl(GRAPH_URL)
with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
    G = pickle.load(gz)
del g_bytes

# Normalize node/edge attributes, store base_length once
for n, data in list(G.nodes(data=True)):
    x = float(data.get("x", data.get("lon", 0.0)))
    y = float(data.get("y", data.get("lat", 0.0)))
    G.nodes[n].clear()
    G.nodes[n]["x"] = x  # lon
    G.nodes[n]["y"] = y  # lat

for u, v, k, data in list(G.edges(keys=True, data=True)):
    base_len = float(data.get("length", 1.0))
    data.clear()
    data["length"] = base_len
    data["base_length"] = base_len  # immutable baseline for every request

NODE_IDS: List[str] = list(G.nodes())
LATS = np.array([G.nodes[n]["y"] for n in NODE_IDS], dtype=np.float32)  # lat
LONS = np.array([G.nodes[n]["x"] for n in NODE_IDS], dtype=np.float32)  # lon

# ------------------------------------------
# Load ML pipeline
# ------------------------------------------
model  = joblib.load(BytesIO(dl(LGBM_URL)))     # sklearn-compatible LightGBM
scaler = joblib.load(BytesIO(dl(SCALER_URL)))   # e.g., StandardScaler
kmeans = joblib.load(BytesIO(dl(KMEANS_URL)))   # sklearn KMeans

# ------------------------------------------
# Supabase (server-side)
# ------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------
# Routing utilities
# ------------------------------------------
def nearest_node(lat: float, lon: float) -> str:
    d2 = (LATS - lat)**2 + (LONS - lon)**2  # OK within city scale
    return NODE_IDS[int(np.argmin(d2))]

_cached_hour: Optional[int] = None

def set_weights_for_hour(hour: int, force: bool = False) -> None:
    """
    Rebuild edge weights from immutable base_length + model 'predicted_safety'.
    Cached by hour, unless force=True.
    """
    global _cached_hour
    if _cached_hour == hour and not force:
        return

    # 1) Cluster zones for features
    km_dtype = getattr(kmeans, "cluster_centers_", np.array([[0.0, 0.0]])).dtype
    coords = np.stack([LATS, LONS], axis=1)
    coords = np.ascontiguousarray(coords, dtype=km_dtype)
    zones = kmeans.predict(coords)

    # 2) Build features with scaler dtype
    sc_dtype = getattr(scaler, "mean_", np.array([0.0], dtype=np.float64)).dtype
    ang = 2.0 * math.pi * (hour % 24) / 24.0
    hour_sin = np.full(LATS.shape, math.sin(ang), dtype=sc_dtype)
    hour_cos = np.full(LATS.shape, math.cos(ang), dtype=sc_dtype)
    X = np.column_stack([
        LATS.astype(sc_dtype, copy=False),
        LONS.astype(sc_dtype, copy=False),
        hour_sin,
        hour_cos,
        zones.astype(sc_dtype, copy=False),
    ])
    X = np.ascontiguousarray(X, dtype=sc_dtype)

    # 3) Predict node safety
    Xs = scaler.transform(X)
    preds = model.predict(Xs).astype(np.float32, copy=False)
    for nid, s in zip(NODE_IDS, preds):
        G.nodes[nid]["predicted_safety"] = float(s)

    # 4) Rebuild edge weights fresh from base
    alpha = 0.05  # influence of model safety
    for u, v, k, data in G.edges(keys=True, data=True):
        base = data.get("base_length", data.get("length", 1.0))
        su = G.nodes[u].get("predicted_safety", 0.0)
        sv = G.nodes[v].get("predicted_safety", 0.0)
        data["weight"] = float(base) * (1.0 + alpha * (su + sv) * 0.5)

    _cached_hour = hour

def adjust_weights_for_crimes():
    """
    Penalize edges near recent crime reports exactly once per request.
    No compounding across requests (we always rebuild weights first).
    """
    # pull recent reports (UTC ISO)
    since_time = time.time() - CRIME_WINDOW_DAYS * 24 * 3600
    iso_since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since_time))

    res = supabase.table("reports") \
        .select("lat,lon,severity,created_at") \
        .gte("created_at", iso_since) \
        .execute()

    rows = res.data or []
    if not rows:
        return

    RADIUS = CRIME_RADIUS_METERS
    SEV_SCALE = 0.25   # severity increases penalty contribution
    MAX_FACTOR = 3.0   # cap per-edge penalty

    for u, v, k, edata in G.edges(keys=True, data=True):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]   # lon, lat
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]

        penalty = 1.0
        for rpt in rows:
            rlat = float(rpt.get("lat", 0.0))
            rlon = float(rpt.get("lon", 0.0))
            sev  = float(rpt.get("severity") or 0.0)

            du = haversine(uy, ux, rlat, rlon)
            dv = haversine(vy, vx, rlat, rlon)

            if du < RADIUS or dv < RADIUS:
                penalty = min(MAX_FACTOR, penalty + (1.0 + SEV_SCALE * sev) - 1.0)

        edata["weight"] = edata["weight"] * penalty  # assign once

# ------------------------------------------
# FastAPI app
# ------------------------------------------
class PointIn(BaseModel):
    lat: float
    lon: float

class RouteReq(BaseModel):
    source: PointIn
    dest: PointIn
    local_time: Optional[str] = None  # "HH:MM" from client (optional)

app = FastAPI(title="Safer Routing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "nodes": len(NODE_IDS),
        "edges": G.number_of_edges(),
        "has_model": model is not None,
        "has_scaler": scaler is not None,
        "has_kmeans": kmeans is not None,
        "cached_hour": _cached_hour,
        "crime_window_days": CRIME_WINDOW_DAYS,
        "crime_radius_m": CRIME_RADIUS_METERS,
    }

@app.get("/debug/dtypes")
def dtypes() -> Dict[str, str]:
    return {
        "LATS": str(LATS.dtype),
        "LONS": str(LONS.dtype),
        "kmeans_centers": str(getattr(kmeans, "cluster_centers_", np.array([[0.0]])).dtype),
        "scaler_dtype": str(getattr(scaler, "mean_", np.array([0.0], dtype=np.float64)).dtype),
    }

@app.post("/route")
def route(req: RouteReq):
    # 1) Determine hour (client-provided or server local)
    hour = time.localtime().tm_hour
    if req.local_time:
        try:
            hour = int(req.local_time.split(":")[0]) % 24
        except Exception:
            pass

    # 2) Rebuild baseline weights fresh for this hour
    set_weights_for_hour(hour, force=True)

    # 3) Apply live crime penalty once
    adjust_weights_for_crimes()

    # 4) Snap to nearest nodes & route
    orig = nearest_node(req.source.lat, req.source.lon)
    dest = nearest_node(req.dest.lat, req.dest.lon)

    try:
        path = nx.shortest_path(G, orig, dest, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(status_code=400, detail="No path found between these points.")

    # 5) Build response
    coords = [{"lat": float(G.nodes[n]["y"]), "lon": float(G.nodes[n]["x"])} for n in path]

    dist_m = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edata_all = G.get_edge_data(u, v)
        if edata_all:
            # pick first multi-edge; or choose min length if desired
            any_edge = next(iter(edata_all.values()))
            dist_m += float(any_edge.get("length", 0.0))

    scores = [float(G.nodes[n].get("predicted_safety", 0.0)) for n in path]
    safety = float(np.mean(scores)) if scores else 0.0

    return {
        "polyline": coords,           # [{lat, lon}, ...]
        "distance_m": dist_m,         # meters
        "duration_min": dist_m / 70.0,  # ~walking pace 4.2 km/h
        "safety_score": safety,       # mean predicted safety along path
        "hour": hour,
        "crime_reports_considered": True,
    }
