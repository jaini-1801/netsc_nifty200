"""Correlation matrix -> thresholded graph."""
import numpy as np
import networkx as nx


def build_graph(corr, r, sector=None, positive_only=True):
    """Edge (i,j) iff rho_ij > r (or |rho_ij| > r).
    Stores 'weight' = rho and 'dist' = sqrt(2(1-rho)); use 'dist' for path-based metrics."""
    tick = list(corr.columns)
    A = corr.values.copy()
    np.fill_diagonal(A, 0)
    mask = (A > r) if positive_only else (np.abs(A) > r)
    G = nx.Graph()
    G.add_nodes_from(tick)
    for i, j in zip(*np.where(np.triu(mask, 1))):
        rho = A[i, j]
        G.add_edge(tick[i], tick[j], weight=rho, dist=float(np.sqrt(2 * (1 - rho))))
    if sector:
        nx.set_node_attributes(G, {t: sector.get(t, "NA") for t in G}, "sector")
    return G


def giant(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()
