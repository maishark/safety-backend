# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, time, math, gzip, pickle, requests, joblib, numpy as np, networkx as nx
from io import BytesIO
from typing import Optional, List, Dict, Any

# ----------------------------
# Config via environment
# ----------------------------
GRAPH_URL  = os.getenv("GRAPH_URL")   # gpickle.gz or graphml.gz URL
LGBM_URL   = os.getenv("LGBM_URL")    # joblib dump URL
SCALER_URL = os.getenv("SCALER_URL")  # joblib dump URL
KMEANS_URL = os.getenv("KMEANS_URL")  # joblib dump URL

def dl(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

# ----------------------------
# Load graph (gpickle.gz or graphml.gz)
# ----------------------------
g_bytes = dl(GRAPH_URL)
ext = GRAPH_URL.lower()
if ext.endswith(".gpickle.gz") or ext.endswith(".pickle.gz"):
    with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
        G = pickle.load(gz)  # expects networkx graph
else:
    with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
        G = nx.read_graphml(gz)
del g_bytes

# Keep only what we need (x=lon, y=lat; edges keep length)
for n, data in list(G.nodes(data=True)):
    x = float(data.get("x", data.get("lon", 0.0)))
    y = float(data.get("y", data.get("lat", 0.0)))
    G.nodes[n].clear()
    G.nodes[n]["x"] = x
    G.nodes[n]["y"] = y

for u, v, k, data in list(G.edges(keys=True, data=True)):
    length = float(data.get("length", 1.0))  # meters
    G.edges[u, v, k].clear()
    G.edges[u, v, k]["length"] = length

NODE_IDS: List[str] = list(G.nodes())
LATS = np.array([G.nodes[n]["y"] for n in NODE_IDS], dtype=np.float32)  # lat
LONS = np.array([G.nodes[n]["x"] for n in NODE_IDS], dtype=np.float32)  # lon

# ----------------------------
# Load models
# ----------------------------
model  = joblib.load(BytesIO(dl(LGBM_URL)))     # LightGBM regressor (sklearn wrapper)
scaler = joblib.load(BytesIO(dl(SCALER_URL)))   # e.g., StandardScaler
kmeans = joblib.load(BytesIO(dl(KMEANS_URL)))   # sklearn KMeans

_cached_hour: Optional[int] = None

def nearest_node(lat: float, lon: float) -> str:
    """Snap to nearest graph node by squared euclidean in (lat,lon)."""
    d2 = (LATS - lat)**2 + (LONS - lon)**2
    return NODE_IDS[int(np.argmin(d2))]

def set_weights_for_hour(hour: int) -> None:
    """
    Compute per-node 'predicted_safety' via ML pipeline and
    set edge 'weight' accordingly. Cached by hour.
    """
    global _cached_hour
    if _cached_hour == hour:
        return

    # 1) KMeans zones
    km_dtype = getattr(kmeans, "cluster_centers_", np.array([[0.0, 0.0]], dtype=np.float64)).dtype
    coords = np.stack([LATS, LONS], axis=1)
    coords = np.ascontiguousarray(coords, dtype=km_dtype)
    zones = kmeans.predict(coords)

    # 2) Build features with scaler dtype
    sc_proto = getattr(scaler, "mean_", None)
    sc_dtype = sc_proto.dtype if sc_proto is not None else np.float64

    ang = 2.0 * math.pi * (hour % 24) / 24.0
    hour_sin = np.full(LATS.shape, math.sin(ang), dtype=sc_dtype)
    hour_cos = np.full(LATS.shape, math.cos(ang), dtype=sc_dtype)

    # Feature order must match training: [lat, lon, hour_sin, hour_cos, zone]
    X = np.column_stack([
        LATS.astype(sc_dtype, copy=False),
        LONS.astype(sc_dtype, copy=False),
        hour_sin,
        hour_cos,
        zones.astype(sc_dtype, copy=False),
    ])
    X = np.ascontiguousarray(X, dtype=sc_dtype)

    # 3) Transform + predict
    Xs = scaler.transform(X)
    preds = model.predict(Xs).astype(np.float32, copy=False)

    # 4) Write per-node safety
    for nid, s in zip(NODE_IDS, preds):
        G.nodes[nid]["predicted_safety"] = float(s)

    # 5) Edge weight = length * (1 + alpha * avg(node safety))
    alpha = 0.05
    for u, v, k, data in G.edges(keys=True, data=True):
        base_len = data.get("length", 1.0)
        su = G.nodes[u].get("predicted_safety", 0.0)
        sv = G.nodes[v].get("predicted_safety", 0.0)
        data["weight"] = float(base_len) * (1.0 + alpha * (su + sv) * 0.5)

    _cached_hour = hour

# ----------------------------
# Dhaka traffic base speeds
# ----------------------------
def dhaka_base_speed(mode: str, hour: int, dow: int) -> float:
    """
    Return base speed in meters/minute for the given mode & time.
    dow: 0=Mon .. 6=Sun (Bangladesh weekend Fri=4, Sat=5 if Monday=0)
    """
    mode = (mode or "walk").lower()

    if mode == "walk":
        return 80.0  # ~4.8 km/h

    if mode == "bike":
        base = 220.0  # ~13.2 km/h
        # mild peak slowdowns
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            base *= 0.9
        # weekend boost
        if dow in (4, 5):  # Fri, Sat
            base *= 1.1
        return base

    # car
    # rough city profile (tune!)
    if 7 <= hour <= 10 or 17 <= hour <= 21:      # strong peaks
        base = 360.0   # ~21.6 km/h
    elif 11 <= hour <= 16:                        # mid-day moderate
        base = 520.0   # ~31.2 km/h
    else:                                          # late-night/early-morning freer
        base = 800.0   # ~48 km/h

    if dow in (4, 5):  # Fri, Sat: often lighter
        base *= 1.15

    return max(250.0, min(base, 1000.0))  # clamp

# ----------------------------
# API
# ----------------------------
class PointIn(BaseModel):
    lat: float
    lon: float

class RouteReq(BaseModel):
    source: PointIn
    dest: PointIn
    local_time: Optional[str] = None  # "HH:MM" (client local)
    mode: Optional[str] = "walk"      # "walk" | "bike" | "car"

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
    # 1) Use client-provided hour if given ("HH:MM"), else server local hour
    hour = time.localtime().tm_hour
    if req.local_time:
        try:
            hour = int(req.local_time.split(":")[0]) % 24
        except Exception:
            pass
    set_weights_for_hour(hour)

    # weekday for traffic profile
    now_t = time.localtime()
    dow = now_t.tm_wday  # 0=Mon .. 6=Sun

    # 2) Snap to nearest nodes
    orig = nearest_node(req.source.lat, req.source.lon)
    dest = nearest_node(req.dest.lat, req.dest.lon)

    # 3) Safety-weighted shortest path
    try:
        path = nx.shortest_path(G, orig, dest, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(status_code=400, detail="No path found between these points.")

    # 4) Polyline + distance + per-edge duration with Dhaka speeds
    coords = [{"lat": float(G.nodes[n]["y"]), "lon": float(G.nodes[n]["x"])} for n in path]

    mode = (req.mode or "walk").lower()
    BASE_SPEED = dhaka_base_speed(mode, hour, dow)  # m/min
    MIN_SPEED  = max(0.5 * BASE_SPEED, 40.0)        # clamp

    # Risk slowdown: if your model's higher value = *riskier*,
    # divide speed; if higher = *safer*, flip sign to multiply.
    GAMMA = 0.05  # 5% slow per unit avg "risk"

    dist_m = 0.0
    dur_min = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edata_all = G.get_edge_data(u, v)
        if not edata_all:
            continue
        e = next(iter(edata_all.values()))
        elen = float(e.get("length", 0.0))  # meters
        dist_m += elen

        su = float(G.nodes[u].get("predicted_safety", 0.0))
        sv = float(G.nodes[v].get("predicted_safety", 0.0))
        s_avg = 0.5 * (su + sv)

        # Slow slightly on "riskier" segments
        speed = BASE_SPEED / (1.0 + GAMMA * max(0.0, s_avg))
        speed = max(MIN_SPEED, speed)

        dur_min += elen / speed

    return {
        "polyline": coords,
        "distance_m": dist_m,
        "duration_min": dur_min,
        "mode": mode,
        "hour": hour,
        "note": "ETA uses Dhaka time-of-day base speeds + small slowdown on riskier segments.",
    }
