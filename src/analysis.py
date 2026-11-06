"""Tasks A, B, C: degree distribution, centralities, clustering, homophily, communities."""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi
from .network_builder import giant
from . import config as C


def basic_stats(G):
    gc = giant(G)
    deg = [d for _, d in G.degree()]
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "density": nx.density(G), "avg_degree": float(np.mean(deg)), "max_degree": max(deg),
        "components": nx.number_connected_components(G), "giant_size": gc.number_of_nodes(),
        "diameter_giant": nx.diameter(gc), "avg_path_giant": nx.average_shortest_path_length(gc),
        "avg_clustering": nx.average_clustering(G), "transitivity": nx.transitivity(G),
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
    }


def degree_fit(G):
    """Power-law vs exponential vs log-normal. R>0 favours the power law."""
    import powerlaw
    deg = np.array([d for _, d in G.degree() if d > 0])
    fit = powerlaw.Fit(deg, discrete=True, verbose=False)
    out = {"alpha": fit.power_law.alpha, "kmin": fit.xmin}
    for alt in ["exponential", "lognormal"]:
        R, p = fit.distribution_compare("power_law", alt)
        out[f"R_vs_{alt}"], out[f"p_vs_{alt}"] = R, p
    return out


def centralities(G, sector):
    df = pd.DataFrame({
        "degree": nx.degree_centrality(G),
        "betweenness_hop": nx.betweenness_centrality(G),
        "betweenness_dist": nx.betweenness_centrality(G, weight="dist"),
        "closeness": nx.closeness_centrality(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=2000),
        "core_number": pd.Series(nx.core_number(G)),
        "clustering": pd.Series(nx.clustering(G)),
    })
    df["sector"] = df.index.map(sector)
    return df


def heavyweight_ranks(cent):
    names = {t.split(".")[0]: t for t in cent.index}
    rows = [names[h] for h in C.HEAVYWEIGHTS if h in names]
    out = cent.loc[rows, ["degree", "betweenness_hop", "eigenvector"]].copy()
    for c in ["degree", "betweenness_hop", "eigenvector"]:
        out[c + "_rank"] = cent[c].rank(ascending=False)[rows]
    return out


def clustering_vs_null(G, n_er=20, n_rew=10):
    n, m = G.number_of_nodes(), G.number_of_edges()
    er = [nx.average_clustering(nx.gnm_random_graph(n, m, seed=s)) for s in range(n_er)]
    rew = []
    for s in range(n_rew):                       # degree-preserving randomisation
        h = G.copy()
        nx.double_edge_swap(h, nswap=5 * m, max_tries=50 * m, seed=s)
        rew.append(nx.average_clustering(h))
    return {"real": nx.average_clustering(G), "ER_mean": np.mean(er), "ER_std": np.std(er),
            "rewired_mean": np.mean(rew), "rewired_std": np.std(rew)}


def sector_homophily(G):
    secs = nx.get_node_attributes(G, "sector")
    frac = pd.Series(secs).value_counts(normalize=True)
    return {
        "sector_assortativity": nx.attribute_assortativity_coefficient(G, "sector"),
        "frac_edges_same_sector": float(np.mean([secs[u] == secs[v] for u, v in G.edges()])),
        "expected_random": float((frac ** 2).sum()),
    }


def communities(G, sector):
    comms = nx.community.louvain_communities(G, seed=C.SEED)
    lab = {v: i for i, cs in enumerate(comms) for v in cs}
    nodes = list(G.nodes())
    summary = {"n_communities": len(comms), "modularity": nx.community.modularity(G, comms),
               "NMI_vs_sector": nmi([sector[v] for v in nodes], [lab[v] for v in nodes])}
    table = pd.crosstab(pd.Series(lab, name="community"),
                        pd.Series({v: sector[v] for v in nodes}, name="sector"))
    return summary, table, lab
