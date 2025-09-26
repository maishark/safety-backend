# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, io, time, math, requests, joblib, numpy as np
import networkx as nx

# ---- Env vars (you will set these on your host) ----
GRAPH_URL  = os.getenv("GRAPH_URL")
LGBM_URL   = os.getenv("LGBM_URL")
SCALER_URL = os.getenv("SCALER_URL")
KMEANS_URL = os.getenv("KMEANS_URL")

def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

# ---- Load assets once at startup ----
graphml_bytes = download_bytes(GRAPH_URL)
G = nx.read_graphml(io.BytesIO(graphml_bytes))

NODE_IDS = list(G.nodes())
LATS = np.array([float(G.nodes[n].get("y", 0.0)) for n in NODE_IDS])
LONS = np.array([float(G.nodes[n].get("x", 0.0)) for n in NODE_IDS])

model  = joblib.load(io.BytesIO(download_bytes(LGBM_URL)))
scaler = joblib.load(io.BytesIO(download_bytes(SCALER_URL)))
kmeans = joblib.load(io.BytesIO(download_bytes(KMEANS_URL)))

_cached_hour = None

def nearest_node(lat: float, lon: float) -> str:
    d2 = (LATS - lat)**2 + (LONS - lon)**2
    return NODE_IDS[int(np.argmin(d2))]

def set_weights_for_hour(hour: int):
    global _cached_hour
    if _cached_hour == hour:
        return
    zones = kmeans.predict(np.c_[LATS, LONS])
    angle = 2 * math.pi * (hour % 24) / 24
    X = np.c_[LATS, LONS,
              np.full_like(LATS, math.sin(angle)),
              np.full_like(LATS, math.cos(angle)),
              zones]
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    for nid, s in zip(NODE_IDS, preds):
        G.nodes[nid]["predicted_safety"] = float(s)
    for u, v, k, data in G.edges(keys=True, data=True):
        base = float(data.get("length", 1.0))
        su = float(G.nodes[u].get("predicted_safety", 0.0))
        sv = float(G.nodes[v].get("predicted_safety", 0.0))
        data["weight"] = base * (1.0 + 0.05 * (su + sv) / 2.0)
    _cached_hour = hour

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
        try: hour = int(req.local_time.split(":")[0]) % 24
        except: pass
    set_weights_for_hour(hour)
    orig = nearest_node(req.source.lat, req.source.lon)
    dest = nearest_node(req.dest.lat, req.dest.lon)
    path = nx.shortest_path(G, orig, dest, weight="weight")
    coords = [{"lat": float(G.nodes[n]["y"]), "lon": float(G.nodes[n]["x"])} for n in path]
    dist = sum(float(list(G.get_edge_data(path[i], path[i+1]).values())[0].get("length", 0.0))
               for i in range(len(path)-1))
    scores = [float(G.nodes[n].get("predicted_safety", 0.0)) for n in path]
    safety = float(np.mean(scores)) if scores else 0.0
    return {
        "polyline": coords,
        "distance_m": dist,
        "duration_min": dist/70.0,
        "safety_score": safety
    }
