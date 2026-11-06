"""Extras: market mode, robustness, temporal stability, MST."""
import random
import numpy as np
import pandas as pd
import networkx as nx
from .network_builder import build_graph
from . import config as C


def market_mode(corr):
    w = np.linalg.eigvalsh(corr.values)
    return {"top_eigenvalue": w[-1], "variance_share": w[-1] / w.sum(), "second_share": w[-2] / w.sum()}


def attack_curve(G, targeted, seed=C.SEED):
    rng = random.Random(seed)
    nodes = [n for n, _ in sorted(G.degree, key=lambda x: -x[1])] if targeted else rng.sample(list(G), len(G))
    H, N, sizes = G.copy(), G.number_of_nodes(), []
    for nd in nodes:
        H.remove_node(nd)
        sizes.append(len(max(nx.connected_components(H), key=len)) / N if H.number_of_nodes() else 0)
    return np.arange(1, N + 1) / N, np.array(sizes)


def temporal_stability(ret, sector, r=C.R_MAIN, k=10):
    half = len(ret) // 2
    tops = []
    for part in (ret.iloc[:half], ret.iloc[half:]):
        g = build_graph(part.corr(), r, sector)
        tops.append(set(pd.Series(dict(g.degree())).nlargest(k).index))
    return {"jaccard_top10": len(tops[0] & tops[1]) / len(tops[0] | tops[1]),
            "first_half": sorted(tops[0]), "second_half": sorted(tops[1])}


def mst(corr):
    D = np.sqrt(2 * (1 - corr))
    return nx.minimum_spanning_tree(nx.from_pandas_adjacency(D))
