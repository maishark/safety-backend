# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, time, math, gzip, pickle, requests, joblib, numpy as np, networkx as nx
from io import BytesIO

GRAPH_URL  = os.getenv("GRAPH_URL")   # e.g. .../models/graph_min.gpickle.gz or graph.graphml.gz
LGBM_URL   = os.getenv("LGBM_URL")
SCALER_URL = os.getenv("SCALER_URL")
KMEANS_URL = os.getenv("KMEANS_URL")

def dl(url: str) -> bytes:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content

# ---- Load compact graph (supports .gpickle.gz or .graphml.gz) ----
g_bytes = dl(GRAPH_URL)
ext = GRAPH_URL.lower()
if ext.endswith(".gpickle.gz") or ext.endswith(".pickle.gz"):
    with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
        G = pickle.load(gz)
else:  # assume graphml.gz
    with gzip.GzipFile(fileobj=BytesIO(g_bytes)) as gz:
        G = nx.read_graphml(gz)
del g_bytes

# Defensive: keep only minimal attrs
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

NODE_IDS = list(G.nodes())
LATS = np.array([G.nodes[n]["y"] for n in NODE_IDS], dtype=np.float32)
LONS = np.array([G.nodes[n]["x"] for n in NODE_IDS], dtype=np.float32)

# ---- Load models ----
model  = joblib.load(BytesIO(dl(LGBM_URL)))
scaler = joblib.load(BytesIO(dl(SCALER_URL)))
kmeans = joblib.load(BytesIO(dl(KMEANS_URL)))

_cached_hour = None

def nearest_node(lat: float, lon: float) -> str:
    d2 = (LATS - lat)**2 + (LONS - lon)**2
    return NODE_IDS[int(np.argmin(d2))]

def set_weights_for_hour(hour: int):
    global _cached_hour
    if _cached_hour == hour:
        return
    zones = kmeans.predict(np.c_[LATS, LONS])
    ang = 2 * math.pi * (hour % 24) / 24
    X = np.c_[
        LATS,
        LONS,
        np.full_like(LATS, math.sin(ang), dtype=np.float32),
        np.full_like(LATS, math.cos(ang), dtype=np.float32),
        zones.astype(np.float32),
    ]
    Xs = scaler.transform(X)
    preds = model.predict(Xs).astype(np.float32)

    for nid, s in zip(NODE_IDS, preds):
        G.nodes[nid]["predicted_safety"] = float(s)

    for u, v, k, data in G.edges(keys=True, data=True):
        base = data.get("length", 1.0)
        su = G.nodes[u].get("predicted_safety", 0.0)
        sv = G.nodes[v].get("predicted_safety", 0.0)
        data["weight"] = float(base) * (1.0 + 0.05 * (su + sv) / 2.0)

    _cached_hour = hour

# -------- API --------
class PointIn(BaseModel):
    lat: float
    lon: float

class RouteReq(BaseModel):
    source: PointIn
    dest: PointIn
    local_time: str | None = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "nodes": len(NODE_IDS)}

@app.post("/route")
def route(req: RouteReq):
    hour = time.localtime().tm_hour
    if req.local_time:
        try:
            hour = int(req.local_time.split(":")[0]) % 24
        except Exception:
            pass

    set_weights_for_hour(hour)

    orig = nearest_node(req.source.lat, req.source.lon)
    dest = nearest_node(req.dest.lat, req.dest.lon)

    try:
        path = nx.shortest_path(G, orig, dest, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(status_code=400, detail="No path found between these points.")

    coords = [{"lat": float(G.nodes[n]["y"]), "lon": float(G.nodes[n]["x"])} for n in path]

    dist = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edata_all = G.get_edge_data(u, v)
        if edata_all:
            first_edge = next(iter(edata_all.values()))
            dist += float(first_edge.get("length", 0.0))

    scores = [float(G.nodes[n].get("predicted_safety", 0.0)) for n in path]
    safety = float(np.mean(scores)) if scores else 0.0

    return {
        "polyline": coords,
        "distance_m": dist,
        "duration_min": dist / 70.0,
        "safety_score": safety,
    }
