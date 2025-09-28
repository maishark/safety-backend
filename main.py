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
GRAPH_URL  = os.getenv("GRAPH_URL") 
LGBM_URL   = os.getenv("LGBM_URL")   
SCALER_URL = os.getenv("SCALER_URL")  
KMEANS_URL = os.getenv("KMEANS_URL")  
def dl(url: str) -> bytes:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content

# ----------------------------
# Load graph (gpickle.gz or graphml.gz)
# ----------------------------
g_bytes = dl(GRAPH_URL)
ext = GRAPH_URL.lower()
if ext.endswith(".gpickle.gz") or ext.endswith(".pickle.gz"):
    with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
        G = pickle.load(gz)
else:
    with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
        G = nx.read_graphml(gz)
del g_bytes

# Keep only what we need to minimize memory
for n, data in list(G.nodes(data=True)):
    x = float(data.get("x", data.get("lon", 0.0)))
    y = float(data.get("y", data.get("lat", 0.0)))
    G.nodes[n].clear()
    G.nodes[n]["x"] = x
    G.nodes[n]["y"] = y

for u, v, k, data in list(G.edges(keys=True, data=True)):
    length = float(data.get("length", 1.0))
    # preserve only length; weights get computed per hour
    G.edges[u, v, k].clear()
    G.edges[u, v, k]["length"] = length

NODE_IDS: List[str] = list(G.nodes())
# store as float32; we'll cast to match model/scaler as needed
LATS = np.array([G.nodes[n]["y"] for n in NODE_IDS], dtype=np.float32)  # lat ~ y
LONS = np.array([G.nodes[n]["x"] for n in NODE_IDS], dtype=np.float32)  # lon ~ x

# ----------------------------
# Load models
# ----------------------------
model  = joblib.load(BytesIO(dl(LGBM_URL)))     # LightGBM regressor (sklearn wrapper)
scaler = joblib.load(BytesIO(dl(SCALER_URL)))   # e.g., StandardScaler
kmeans = joblib.load(BytesIO(dl(KMEANS_URL)))   # sklearn KMeans

_cached_hour: Optional[int] = None

def nearest_node(lat: float, lon: float) -> str:
    # squared euclidean on lat/lon (OK within city scale)
    d2 = (LATS - lat)**2 + (LONS - lon)**2
    return NODE_IDS[int(np.argmin(d2))]

def set_weights_for_hour(hour: int) -> None:
    """
    Compute per-node 'predicted_safety' from the ML pipeline and
    set edge 'weight' accordingly. Caches by hour to avoid recompute.
    Handles dtype/contiguity strictly to satisfy scikit-learn cython code.
    """
    global _cached_hour
    if _cached_hour == hour:
        return

    # ---- 1) KMeans expects the dtype it was trained with
    km_dtype = getattr(kmeans, "cluster_centers_", np.array([[0.0, 0.0]], dtype=np.float64)).dtype
    coords = np.stack([LATS, LONS], axis=1)  # shape (N, 2)
    coords = np.ascontiguousarray(coords, dtype=km_dtype)
    zones = kmeans.predict(coords)  # np.ndarray (dtype from sklearn, usually int32/int64)

    # ---- 2) Build features for scaler/model using scaler's dtype if available
    sc_prototype = getattr(scaler, "mean_", None)
    if sc_prototype is None:
        sc_dtype = np.float64
    else:
        sc_dtype = sc_prototype.dtype

    ang = 2.0 * math.pi * (hour % 24) / 24.0
    hour_sin = np.full(LATS.shape, math.sin(ang), dtype=sc_dtype)
    hour_cos = np.full(LATS.shape, math.cos(ang), dtype=sc_dtype)

    # Feature order must match what the scaler/model expect during training:
    # [lat, lon, hour_sin, hour_cos, zone]
    X = np.column_stack([
        LATS.astype(sc_dtype, copy=False),
        LONS.astype(sc_dtype, copy=False),
        hour_sin,
        hour_cos,
        zones.astype(sc_dtype, copy=False),
    ])
    X = np.ascontiguousarray(X, dtype=sc_dtype)

    # ---- 3) Transform + predict
    Xs = scaler.transform(X)
    preds = model.predict(Xs).astype(np.float32, copy=False)

    # ---- 4) Write results back to graph
    for nid, s in zip(NODE_IDS, preds):
        G.nodes[nid]["predicted_safety"] = float(s)

    # Edge weight = base_length * (1 + alpha * avg(node safety))
    # tweak alpha as you like; this keeps weight proportional to distance with safety inflation
    alpha = 0.05
    for u, v, k, data in G.edges(keys=True, data=True):
        base = data.get("length", 1.0)
        su = G.nodes[u].get("predicted_safety", 0.0)
        sv = G.nodes[v].get("predicted_safety", 0.0)
        data["weight"] = float(base) * (1.0 + alpha * (su + sv) / 2.0)

    _cached_hour = hour

# ----------------------------
# API
# ----------------------------
class PointIn(BaseModel):
    lat: float
    lon: float

class RouteReq(BaseModel):
    source: PointIn
    dest: PointIn
    local_time: Optional[str] = None  # "HH:MM" from client, optional

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
    # Parse hour from client if provided ("HH:MM"), else server local hour
    hour = time.localtime().tm_hour
    if req.local_time:
        try:
            hour = int(req.local_time.split(":")[0]) % 24
        except Exception:
            pass

    # Compute weights for this hour (cached)
    set_weights_for_hour(hour)

    # Snap to nearest nodes
    orig = nearest_node(req.source.lat, req.source.lon)
    dest = nearest_node(req.dest.lat, req.dest.lon)

    # Shortest path by our dynamic 'weight'
    try:
        path = nx.shortest_path(G, orig, dest, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(status_code=400, detail="No path found between these points.")

    # Polyline in (lat, lon)
    coords = [{"lat": float(G.nodes[n]["y"]), "lon": float(G.nodes[n]["x"])} for n in path]

    # Sum raw lengths in meters
    dist_m = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edata_all = G.get_edge_data(u, v)
        if edata_all:
            # If multi-edges exist, take the first (or the one with min length if you prefer)
            any_edge = next(iter(edata_all.values()))
            dist_m += float(any_edge.get("length", 0.0))

    # Average node safety along the path
    scores = [float(G.nodes[n].get("predicted_safety", 0.0)) for n in path]
    safety = float(np.mean(scores)) if scores else 0.0

    # Very rough duration: 70 m/min (~4.2 km/h) like walking; adjust to your use case
    return {
        "polyline": coords,         # list of {lat, lon}
        "distance_m": dist_m,       # meters
        "duration_min": dist_m / 70.0,
        "safety_score": safety,     # mean predicted safety along route
        "hour": hour,
    }
