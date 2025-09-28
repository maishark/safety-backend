# main.py
import os, time, math, gzip, pickle, requests, joblib, numpy as np, networkx as nx
from io import BytesIO
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ------------------------------------------
# ✅ Config: Environment Variables
# ------------------------------------------
GRAPH_URL  = os.getenv("GRAPH_URL")    # e.g. https://.../graph_min.gpickle.gz
LGBM_URL   = os.getenv("LGBM_URL")     # e.g. https://.../best_model_lgbm.pkl
SCALER_URL = os.getenv("SCALER_URL")   # e.g. https://.../scaler.pkl
KMEANS_URL = os.getenv("KMEANS_URL")   # e.g. https://.../kmeans.pkl

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

CRIME_RADIUS_METERS = 200.0  # 🔥 crime influence radius

# ------------------------------------------
# ✅ Download helper
# ------------------------------------------
def dl(url: str) -> bytes:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content

# ------------------------------------------
# ✅ Load Graph
# ------------------------------------------
g_bytes = dl(GRAPH_URL)
with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
    G = pickle.load(gz)
del g_bytes

# Simplify nodes
for n, data in list(G.nodes(data=True)):
    x = float(data.get("x", data.get("lon", 0.0)))
    y = float(data.get("y", data.get("lat", 0.0)))
    G.nodes[n].clear()
    G.nodes[n]["x"] = x
    G.nodes[n]["y"] = y

for u, v, k, data in list(G.edges(keys=True, data=True)):
    length = float(data.get("length", 1.0))
    G.edges[u, v, k].clear()
    G.edges[u, v, k]["length"] = length

NODE_IDS: List[str] = list(G.nodes())
LATS = np.array([G.nodes[n]["y"] for n in NODE_IDS], dtype=np.float32)
LONS = np.array([G.nodes[n]["x"] for n in NODE_IDS], dtype=np.float32)

# ------------------------------------------
# ✅ Load ML Pipeline
# ------------------------------------------
model  = joblib.load(BytesIO(dl(LGBM_URL)))
scaler = joblib.load(BytesIO(dl(SCALER_URL)))
kmeans = joblib.load(BytesIO(dl(KMEANS_URL)))

# ------------------------------------------
# ✅ Connect Supabase
# ------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------
# ✅ Utilities
# ------------------------------------------
def nearest_node(lat: float, lon: float) -> str:
    d2 = (LATS - lat)**2 + (LONS - lon)**2
    return NODE_IDS[int(np.argmin(d2))]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

_cached_hour: Optional[int] = None

def set_weights_for_hour(hour: int) -> None:
    global _cached_hour
    if _cached_hour == hour:
        return

    # KMeans clustering
    km_dtype = getattr(kmeans, "cluster_centers_", np.array([[0.0, 0.0]])).dtype
    coords = np.stack([LATS, LONS], axis=1)
    coords = np.ascontiguousarray(coords, dtype=km_dtype)
    zones = kmeans.predict(coords)

    # Features
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

    # Predict safety
    Xs = scaler.transform(X)
    preds = model.predict(Xs).astype(np.float32, copy=False)
    for nid, s in zip(NODE_IDS, preds):
        G.nodes[nid]["predicted_safety"] = float(s)

    # Reset edge weights
    alpha = 0.05
    for u, v, k, data in G.edges(keys=True, data=True):
        base = data.get("length", 1.0)
        su = G.nodes[u].get("predicted_safety", 0.0)
        sv = G.nodes[v].get("predicted_safety", 0.0)
        data["weight"] = float(base) * (1.0 + alpha * (su + sv) / 2.0)

    _cached_hour = hour

def adjust_weights_for_crimes():
    # 🔥 Get recent reports (last 5 days)
    since_time = time.time() - 5 * 24 * 3600
    iso_since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since_time))

    data = supabase.table("reports").select("lat,lon,created_at").gte("created_at", iso_since).execute().data
    if not data:
        return

    for u, v, k, edata in G.edges(keys=True, data=True):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]

        for report in data:
            rlat, rlon = float(report["lat"]), float(report["lon"])
            du = haversine(uy, ux, rlat, rlon)
            dv = haversine(vy, vx, rlat, rlon)

            if du < CRIME_RADIUS_METERS or dv < CRIME_RADIUS_METERS:
                edata["weight"] *= 3.0  # ⚠️ Penalize risky edges

# ------------------------------------------
# ✅ FastAPI App
# ------------------------------------------
class PointIn(BaseModel):
    lat: float
    lon: float

class RouteReq(BaseModel):
    source: PointIn
    dest: PointIn
    local_time: Optional[str] = None

app = FastAPI()
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
        "has_model": model is not None,
        "has_scaler": scaler is not None,
        "has_kmeans": kmeans is not None,
        "cached_hour": _cached_hour,
    }

@app.post("/route")
def route(req: RouteReq):
    # 1️⃣ Compute safety for current hour
    hour = time.localtime().tm_hour
    if req.local_time:
        try:
            hour = int(req.local_time.split(":")[0]) % 24
        except:
            pass
    set_weights_for_hour(hour)

    # 2️⃣ Adjust weights with live crime reports
    adjust_weights_for_crimes()

    # 3️⃣ Compute shortest path
    orig = nearest_node(req.source.lat, req.source.lon)
    dest = nearest_node(req.dest.lat, req.dest.lon)
    try:
        path = nx.shortest_path(G, orig, dest, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(status_code=400, detail="No path found between these points.")

    coords = [{"lat": float(G.nodes[n]["y"]), "lon": float(G.nodes[n]["x"])} for n in path]

    dist_m = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edata_all = G.get_edge_data(u, v)
        if edata_all:
            any_edge = next(iter(edata_all.values()))
            dist_m += float(any_edge.get("length", 0.0))

    scores = [float(G.nodes[n].get("predicted_safety", 0.0)) for n in path]
    safety = float(np.mean(scores)) if scores else 0.0

    return {
        "polyline": coords,
        "distance_m": dist_m,
        "duration_min": dist_m / 70.0,
        "safety_score": safety,
        "hour": hour,
        "crime_reports_considered": True,
    }
