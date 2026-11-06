"""Task D: threshold sensitivity."""
import numpy as np
import pandas as pd
import networkx as nx
from .network_builder import build_graph, giant
from . import config as C


def sweep(corr, sector):
    rows = []
    for r in C.R_SWEEP:
        G = build_graph(corr, r, sector)
        gc = giant(G)
        rows.append({
            "r": r, "edges": G.number_of_edges(), "density": nx.density(G),
            "components": nx.number_connected_components(G),
            "giant_frac": gc.number_of_nodes() / G.number_of_nodes(),
            "avg_path_giant": nx.average_shortest_path_length(gc) if gc.number_of_nodes() > 1 else np.nan,
            "avg_clustering": nx.average_clustering(G),
        })
    return pd.DataFrame(rows)
