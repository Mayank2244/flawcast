"""Graph Neural Network-inspired congestion propagation model."""
import math
from typing import Optional

import networkx as nx
import numpy as np

from app.ml.features import haversine_km, CORRIDOR_IMPACT

# Bengaluru key corridors as graph nodes (simplified OSMnx topology)
BENGALURU_GRAPH_NODES = {
    "MG Road": (12.9716, 77.5946),
    "Brigade Road": (12.9719, 77.6070),
    "Residency Road": (12.9680, 77.6010),
    "Kasturba Road": (12.9790, 77.5920),
    "Queens Road": (12.9856, 77.5977),
    "Cubbon Road": (12.9788, 77.5995),
    "ORR Marathahalli": (12.9591, 77.6974),
    "ORR Silk Board": (12.9175, 77.6226),
    "ORR Hebbal": (13.0354, 77.5970),
    "Hosur Road": (12.9169, 77.6100),
    "Mysore Road": (12.9446, 77.5274),
    "Tumkur Road": (13.0374, 77.5181),
    "Bellary Road": (13.0634, 77.5933),
    "Old Madras Road": (12.9753, 77.6257),
    "Bannerghatta Road": (12.9077, 77.6006),
    "Whitefield Road": (12.9698, 77.7500),
    "Electronic City": (12.8456, 77.6603),
    "Jayanagar": (12.9274, 77.5807),
    "Indiranagar": (12.9784, 77.6408),
    "Koramangala": (12.9352, 77.6245),
    "Hebbal Flyover": (13.0354, 77.5947),
    "Silk Board Junction": (12.9175, 77.6226),
    "Chinnaswamy Stadium": (12.9788, 77.5995),
    "Peenya": (13.0374, 77.5181),
    "Yeshwanthpur": (13.0289, 77.5442),
}

CORRIDOR_TO_NODES = {
    "ORR East 1": ["ORR Marathahalli", "Whitefield Road", "Indiranagar"],
    "ORR East 2": ["ORR Marathahalli", "Whitefield Road"],
    "ORR West 1": ["Mysore Road", "Peenya"],
    "ORR North 1": ["ORR Hebbal", "Hebbal Flyover", "Bellary Road"],
    "ORR North 2": ["ORR Hebbal", "Yeshwanthpur"],
    "Mysore Road": ["Mysore Road", "Peenya"],
    "Bellary Road 1": ["Bellary Road", "Hebbal Flyover", "Queens Road"],
    "Bellary Road 2": ["Bellary Road", "ORR Hebbal"],
    "Hosur Road": ["Hosur Road", "Silk Board Junction", "Electronic City"],
    "Bannerghata Road": ["Bannerghatta Road", "Jayanagar", "Silk Board Junction"],
    "Tumkur Road": ["Tumkur Road", "Peenya", "Yeshwanthpur"],
    "Old Madras Road": ["Old Madras Road", "Indiranagar", "MG Road"],
    "Magadi Road": ["Peenya", "Mysore Road"],
    "CBD 1": ["MG Road", "Brigade Road", "Cubbon Road"],
    "CBD 2": ["Chinnaswamy Stadium", "Queens Road", "Kasturba Road"],
    "Non-corridor": [],
}


class CongestionGraphEngine:
    """GAT-inspired graph propagation for spatial congestion spread."""

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for name, (lat, lng) in BENGALURU_GRAPH_NODES.items():
            G.add_node(name, lat=lat, lng=lng)

        nodes = list(BENGALURU_GRAPH_NODES.keys())
        for i, n1 in enumerate(nodes):
            lat1, lng1 = BENGALURU_GRAPH_NODES[n1]
            for n2 in nodes[i + 1:]:
                lat2, lng2 = BENGALURU_GRAPH_NODES[n2]
                dist = haversine_km(lat1, lng1, lat2, lng2)
                if dist < 6.0:
                    weight = 1.0 / (dist + 0.1)
                    G.add_edge(n1, n2, weight=weight, distance_km=dist)
        return G

    def find_nearest_node(self, lat: float, lng: float) -> tuple[str, float]:
        best, best_dist = "MG Road", 999.0
        for name, (nlat, nlng) in BENGALURU_GRAPH_NODES.items():
            d = haversine_km(lat, lng, nlat, nlng)
            if d < best_dist:
                best, best_dist = name, d
        return best, best_dist

    def propagate(
        self,
        source_lat: float,
        source_lng: float,
        crs_score: float,
        corridor: str = "Non-corridor",
        horizon_minutes: int = 120,
    ) -> list[dict]:
        source_node, _ = self.find_nearest_node(source_lat, source_lng)
        impact_radius = CORRIDOR_IMPACT.get(corridor, 2.0)

        propagation = []
        visited = {source_node: 0.0}

        for node in nx.single_source_dijkstra_path_length(self.graph, source_node, weight="distance_km"):
            if node == source_node:
                dist = 0.0
            else:
                try:
                    path = nx.shortest_path(self.graph, source_node, node, weight="distance_km")
                    dist = sum(
                        self.graph[path[i]][path[i + 1]]["distance_km"]
                        for i in range(len(path) - 1)
                    )
                except nx.NetworkXNoPath:
                    continue

            if dist > impact_radius * 2:
                continue

            decay = max(0.05, 1 - (dist / (impact_radius * 2)))
            attention = self._gat_attention(source_node, node, dist)
            propagated_crs = crs_score * decay * attention

            if propagated_crs < 5:
                continue

            nlat, nlng = BENGALURU_GRAPH_NODES[node]
            for t in [30, 60, 90, 120]:
                if t <= horizon_minutes:
                    time_decay = 1 + (t / 120) * 0.3
                    propagation.append({
                        "node": node,
                        "latitude": nlat,
                        "longitude": nlng,
                        "distance_km": round(dist, 2),
                        "crs_score": round(min(100, propagated_crs * time_decay), 1),
                        "time_lag_min": t,
                        "attention_weight": round(attention, 3),
                    })

        corridor_nodes = CORRIDOR_TO_NODES.get(corridor, [])
        for cn in corridor_nodes:
            if cn in BENGALURU_GRAPH_NODES and cn not in visited:
                nlat, nlng = BENGALURU_GRAPH_NODES[cn]
                propagation.append({
                    "node": cn, "latitude": nlat, "longitude": nlng,
                    "distance_km": round(haversine_km(source_lat, source_lng, nlat, nlng), 2),
                    "crs_score": round(crs_score * 0.7, 1),
                    "time_lag_min": 45, "attention_weight": 0.8,
                })

        seen = set()
        unique = []
        for p in sorted(propagation, key=lambda x: -x["crs_score"]):
            key = (p["node"], p["time_lag_min"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[:25]

    def _gat_attention(self, source: str, target: str, distance: float) -> float:
        if source == target:
            return 1.0
        try:
            edge_weight = self.graph[source][target]["weight"]
        except KeyError:
            edge_weight = 1.0 / (distance + 1)
        degree_src = self.graph.degree(source) or 1
        degree_tgt = self.graph.degree(target) or 1
        return min(1.0, edge_weight * math.sqrt(degree_src * degree_tgt) / 10)

    def get_graph_stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4),
            "avg_degree": round(sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()), 2),
        }
