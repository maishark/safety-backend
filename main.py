# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import time
import math
import gzip
import requests
import joblib
import numpy as np
import networkx as nx
from io import BytesIO

# ---------- Environment variables (set these on your host) ----------
GRAPH_URL  = os.getenv("GRAPH_URL")   # e.g. https://.../models/graph.graphml.gz
LGBM_URL   = os.getenv("LGBM_URL")    # e.g. https://.../models/best_model_lgbm.pkl
SCALER_URL = os.getenv("SCALER_URL")  # e.g. https://.../models/scaler.pkl
KMEANS_URL = os.getenv("KMEANS_URL")  # e.g. https://.../models/kmeans.pkl

def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content

# ---------- Load assets once at startup ----------
# Graph: gzipped GraphML from Supabase Storage
graph_bytes = download_bytes(GRAPH_URL)
with gzip.GzipFile(fileobj=BytesIO(graph_bytes)) as gz:
    G = nx.read_graphml(gz)

# Ensure numeric lat/lon arrays + stable node list
NODE_IDS = list(G.nodes())
LATS = np.array([float(G.nodes[n].get("y", G.nodes[n].get("lat", 0.0))) for n in NODE_IDS], dtype=float)
LONS = np.array([float(G.nodes[n].get("x", G.nodes[n].get("lon", 0.0))) for n in NODE_IDS], dtype=float)

# Models: LightGBM + preprocessing
model  = joblib.load(io.BytesIO(download_bytes(LGBM_URL)))
scaler = joblib.load(io.BytesIO(download_bytes(SCALER_URL)))
kmeans = joblib.load(io.BytesIO(download_bytes(KMEANS_URL)))

_cached_hour = None

def nearest_node(lat: float, lon: float) -> str:
    # Fast vectorized nearest by squared Euclidean distance in degrees (OK for city scale)
    d2 = (LATS - lat) ** 2 + (LONS - lon) ** 2
    return NODE_IDS[int(np.argmin(d2))]

def set_weights_for_hour(hour: int):
    """
    Recompute node safety and edge weights when the hour changes.
    Uses your features: [Latitude, Longitude, hour_sin, hour_cos, zone]
    """
    global _cached_hour
    if _cached_hour == hour:
        return

    zones = kmeans.predict(np.c_[LATS, LONS])
    angle = 2 * math.pi * (hour % 24) / 24
    hour_sin, hour_cos = math.sin(angle), math.cos(angle)

    X = np.c_[
        LATS,
        LONS,
        np.full_like(LATS, hour_sin, dtype=float),
        np.full_like(LATS, hour_cos, dtype=float),
        zones,
    ]

    Xs = scaler.transform(X)
    preds = model.predict(Xs).astype(float)

    # Attach node scores
    for nid, s in zip(NODE_IDS, preds):
        G.nodes[nid]["predicted_safety"] = float(s)

    # Recompute edge weights once per hour
    for u, v, k, data in G.edges(keys=True, data=True):
        base = float(data.get("length", 1.0))
        su = float(G.nodes[u].get("predicted_safety", 0.0))
        sv = float(G.nodes[v].get("predicted_safety", 0.0))
        data["weight"] = base * (1.0 + 0.05 * (su + sv) / 2.0)

    _cached_hour = hour

# ---------- API ----------
class PointIn(BaseModel):
    lat: float
    lon: float

class RouteReq(BaseModel):
    source: PointIn
    dest: PointIn
    local_time: str | None = None  # optional "HH:MM"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "nodes": len(NODE_IDS)}

@app.post("/route")
def route(req: RouteReq):
    # Pick hour: provided or server local
    hour = time.localtime().tm_hour
    if req.local_time:
        try:
            hour = int(req.local_time.split(":")[0]) % 24
        except Exception:
            pass

    # Prepare graph weights for this hour
    set_weights_for_hour(hour)

    # Snap endpoints to nearest nodes
    orig = nearest_node(req.source.lat, req.source.lon)
    dest = nearest_node(req.dest.lat, req.dest.lon)

    # Safest (weighted-shortest) path
    path = nx.shortest_path(G, orig, dest, weight="weight")

    # Build response
    coords = [{"lat": float(G.nodes[n].get("y", 0.0)),
               "lon": float(G.nodes[n].get("x", 0.0))} for n in path]

    # Sum edge lengths (handles multi-edges)
    dist = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edata_all = G.get_edge_data(u, v)
        if edata_all:
            # take the first edge's length if multiples
            first_edge = next(iter(edata_all.values()))
            dist += float(first_edge.get("length", 0.0))

    scores = [float(G.nodes[n].get("predicted_safety", 0.0)) for n in path]
    safety = float(np.mean(scores)) if scores else 0.0

    return {
        "polyline": coords,
        "distance_m": dist,
        "duration_min": dist / 70.0,  # ~walking pace; adjust if needed
        "safety_score": safety,
    }
