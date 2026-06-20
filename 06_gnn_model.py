#!/usr/bin/env python3
"""FlowCast AI — Graph Attention Network for congestion propagation (Module C)."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATConv

from flowcast_config import (
    FEATURE_MATRIX_PATH,
    GNN_MODEL_PATH,
    GRAPH_GPKG,
    GRAPH_PT,
    OUTPUTS_EDA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    SEGMENT_ID_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DEVICE = torch.device("cpu")


class CongestionGAT(nn.Module):
    """Three-layer GAT predicting CRS per graph node."""

    def __init__(self, node_features: int = 7) -> None:
        super().__init__()
        self.gat1 = GATConv(node_features, 64, heads=4, dropout=0.2)
        self.bn1 = BatchNorm(64 * 4)
        self.gat2 = GATConv(64 * 4, 32, heads=4, dropout=0.2)
        self.bn2 = BatchNorm(32 * 4)
        self.gat3 = GATConv(32 * 4, 1, heads=1, concat=False, dropout=0.2)
        self.bn3 = BatchNorm(1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returning CRS predictions per node."""
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.sigmoid(self.bn3(self.gat3(x, edge_index))) * 10.0
        return x.squeeze(-1)


def build_or_load_graph() -> tuple[nx.Graph, dict[str, int]]:
    """Load OSM graph if cached; otherwise build synthetic Bengaluru-like graph."""
    GRAPH_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if GRAPH_PT.exists():
        with open(GRAPH_PT, "rb") as f:
            payload = pickle.load(f)
        return payload["graph"], payload["node_map"]

    G: nx.Graph | None = None
    if GRAPH_GPKG.exists():
        try:
            import osmnx as ox
            G = ox.load_graph_graphml(str(GRAPH_GPKG.with_suffix(".graphml")))
        except Exception:
            G = None

    if G is None:
        try:
            import osmnx as ox
            logger.info("Downloading Bengaluru road graph via OSMnx (may take minutes)...")
            G = ox.graph_from_place("Bengaluru, Karnataka, India", network_type="drive")
            ox.save_graphml(G, str(GRAPH_GPKG.with_suffix(".graphml")))
        except Exception as exc:
            logger.warning("OSMnx unavailable (%s) — using synthetic graph.", exc)
            G = nx.grid_2d_graph(7, 8)
            G = nx.convert_node_labels_to_integers(G)

    # Map demo segment IDs to graph nodes
    nodes = list(G.nodes())
    from generate_demo_data import SEGMENTS
    node_map = {seg[0]: nodes[i % len(nodes)] for i, seg in enumerate(SEGMENTS)}
    with open(GRAPH_PT, "wb") as f:
        pickle.dump({"graph": G, "node_map": node_map}, f)
    return G, node_map


def build_node_features(df: pd.DataFrame, node_map: dict[str, int]) -> np.ndarray:
    """Build 7-dim node feature matrix aligned to segment order."""
    latest = df.sort_values("timestamp").groupby(SEGMENT_ID_COL).tail(1).set_index(SEGMENT_ID_COL)
    feats = []
    for seg_id in node_map:
        row = latest.loc[seg_id] if seg_id in latest.index else None
        crs = float(row["congestion_risk_score"]) if row is not None else 5.0
        hist = float(df[df[SEGMENT_ID_COL] == seg_id]["congestion_risk_score"].mean())
        lat = float(row["latitude"]) if row is not None else 12.97
        lon = float(row["longitude"]) if row is not None else 77.59
        near_stadium = 1.0 if abs(lat - 12.9788) < 0.02 else 0.0
        near_hospital = 1.0 if abs(lat - 12.9716) < 0.015 else 0.0
        on_ring = 1.0 if "ORR" in seg_id else 0.0
        degree = 0.5
        betweenness = 0.4
        feats.append([crs, hist, degree, betweenness, near_stadium, near_hospital, on_ring])
    return np.array(feats, dtype=np.float32)


def graph_to_pyg(G: nx.Graph, x: np.ndarray, node_map: dict[str, int]) -> Data:
    """Convert NetworkX graph and features to PyG Data."""
    seg_ids = list(node_map.keys())
    id_to_idx = {seg: i for i, seg in enumerate(seg_ids)}
    edges = []
    for u, v in G.edges():
        su = seg_ids[u % len(seg_ids)] if isinstance(u, int) else str(u)
        sv = seg_ids[v % len(seg_ids)] if isinstance(v, int) else str(v)
        if su in id_to_idx and sv in id_to_idx:
            edges.append([id_to_idx[su], id_to_idx[sv]])
            edges.append([id_to_idx[sv], id_to_idx[su]])
    if not edges:
        for i in range(len(seg_ids) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index)


def train_gnn(df: pd.DataFrame, epochs: int = 50) -> CongestionGAT:
    """Train GAT with MSE + spatial smoothness regularizer."""
    torch.manual_seed(RANDOM_SEED)
    G, node_map = build_or_load_graph()
    x = build_node_features(df, node_map)
    data = graph_to_pyg(G, x, node_map)

    # Labels: future mean CRS per segment (proxy for cascade)
    labels = []
    for seg in node_map:
        seg_df = df[df[SEGMENT_ID_COL] == seg]["congestion_risk_score"]
        labels.append(float(seg_df.shift(-4).fillna(seg_df.mean()).iloc[-1]))
    y = torch.tensor(labels, dtype=torch.float32)

    model = CongestionGAT(node_features=x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = data.to(DEVICE)
    y = y.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        pred = model(data.x, data.edge_index)
        mse = F.mse_loss(pred, y)
        # Spatial smoothness: adjacent nodes should have similar predictions
        src, dst = data.edge_index
        smooth = torch.mean((pred[src] - pred[dst]) ** 2)
        loss = mse + 0.1 * smooth
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d loss=%.4f", epoch + 1, loss.item())

    GNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "node_map": node_map}, GNN_MODEL_PATH)
    return model


def simulate_propagation(incident_node_id: str, severity: float, time_steps: int = 8) -> dict[str, list[float]]:
    """Simulate congestion spread from incident source over 15-min steps."""
    G, node_map = build_or_load_graph()
    if not GNN_MODEL_PATH.exists():
        # Demo wave propagation
        result = {seg: [min(10.0, severity * max(0, 1 - i * 0.08)) for i in range(time_steps)] for seg in node_map}
        if incident_node_id in node_map:
            for i in range(time_steps):
                result[incident_node_id][i] = min(10.0, severity + i * 0.2)
        return result

    ckpt = torch.load(GNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CongestionGAT()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    x = build_node_features(df, node_map)
    x[list(node_map.keys()).index(incident_node_id)][0] = severity if incident_node_id in node_map else severity
    data = graph_to_pyg(G, x, node_map)

    with torch.no_grad():
        base = model(data.x, data.edge_index).numpy()

    result: dict[str, list[float]] = {}
    seg_ids = list(node_map.keys())
    for idx, seg in enumerate(seg_ids):
        series = []
        val = base[idx]
        for t in range(time_steps):
            val = min(10.0, val * (1.02 if seg == incident_node_id else 0.95) + 0.1)
            series.append(float(val))
        result[seg] = series
    return result


def generate_propagation_map(propagation_dict: dict[str, list[float]], timestamp: str, step: int = 0) -> str:
    """Return Folium HTML with colour-coded segments for one time step."""
    import folium

    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    latest = df.groupby(SEGMENT_ID_COL).agg({"latitude": "first", "longitude": "first"}).reset_index()
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="CartoDB dark_matter")

    def crs_color(crs: float) -> str:
        if crs < 4:
            return "#2ecc71"
        if crs < 7:
            return "#f39c12"
        return "#e74c3c"

    for seg, series in propagation_dict.items():
        if step >= len(series):
            continue
        row = latest[latest[SEGMENT_ID_COL] == seg]
        if row.empty:
            continue
        lat, lon = float(row.iloc[0]["latitude"]), float(row.iloc[0]["longitude"])
        crs = series[step]
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=crs_color(crs),
            fill=True,
            fill_opacity=0.8,
            popup=f"{seg}: CRS={crs:.1f}",
        ).add_to(m)

    out = PROJECT_ROOT / "outputs" / "eda" / f"propagation_{step}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return str(out)


def update_graph_features(segment_id: str, new_crs: float) -> dict[str, float]:
    """Integration hook: update node CRS and return downstream predictions."""
    G, node_map = build_or_load_graph()
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    sub = df[df[SEGMENT_ID_COL] == segment_id]
    if not sub.empty:
        df.loc[sub.index[-1], "congestion_risk_score"] = new_crs

    if not GNN_MODEL_PATH.exists():
        return {seg: new_crs * 0.85 for seg in node_map if seg != segment_id}

    ckpt = torch.load(GNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CongestionGAT()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x = build_node_features(df, node_map)
    if segment_id in node_map:
        idx = list(node_map.keys()).index(segment_id)
        x[idx, 0] = new_crs
    data = graph_to_pyg(G, x, node_map)
    with torch.no_grad():
        pred = model(data.x, data.edge_index).numpy()
    return {seg: float(pred[i]) for i, seg in enumerate(node_map.keys())}


def main() -> None:
    """Train GNN and save propagation demo map."""
    if not FEATURE_MATRIX_PATH.exists():
        from generate_demo_data import main as gen
        gen()
        import importlib
        importlib.import_module("02_features").main()
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    train_gnn(df, epochs=50)
    prop = simulate_propagation("MG001", severity=8.5, time_steps=8)
    generate_propagation_map(prop, timestamp="2025-03-12T19:00:00", step=4)
    logger.info("GNN saved to %s", GNN_MODEL_PATH)


if __name__ == "__main__":
    main()
