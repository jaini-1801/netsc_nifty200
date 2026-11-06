"""All figures. Each function saves a PNG into out/figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from . import config as C


def _save(name):
    plt.tight_layout(); plt.savefig(f"{C.FIG_DIR}/{name}.png", dpi=200, bbox_inches="tight"); plt.close()


def plot_sweep(sweep):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, c, t in zip(ax, ["edges", "avg_path_giant", "giant_frac"],
                       ["Number of edges", "Avg path length (giant comp.)", "Giant component fraction"]):
        a.plot(sweep["r"], sweep[c], "o-"); a.set_xlabel("threshold r"); a.set_title(t); a.grid(alpha=.3)
    _save("D_threshold_sensitivity")


def plot_degree(G):
    deg = np.array([d for _, d in G.degree() if d > 0])
    v, c = np.unique(deg, return_counts=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(v, c); ax[0].set_title("Degree distribution (linear)"); ax[0].set_ylabel("count")
    ax[1].loglog(v, c / c.sum(), "o"); ax[1].set_title("Degree distribution (log-log)"); ax[1].set_ylabel("P(k)")
    for a in ax: a.set_xlabel("degree k")
    _save("A_degree_distribution")


def plot_network_by_sector(G, sector, top_labels):
    nodes = list(G.nodes())
    sectors = sorted(set(sector[v] for v in nodes)); cmap = plt.get_cmap("tab20")
    pos = nx.spring_layout(G, k=0.25, seed=C.SEED, iterations=200)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=.05, width=.4)
    nx.draw_networkx_nodes(G, pos, node_size=[20 + 6 * G.degree(v) for v in nodes],
                           node_color=[cmap(sectors.index(sector[v]) % 20) for v in nodes])
    nx.draw_networkx_labels(G, pos, {v: v.split(".")[0] for v in top_labels}, font_size=8)
    plt.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=cmap(i % 20), label=s)
                        for i, s in enumerate(sectors)], fontsize=7, loc="lower left")
    plt.axis("off"); _save("C_network_by_sector")


def plot_heatmap(df, name, title, annot=False):
    plt.figure(figsize=(max(6, .45 * df.shape[1] + 3), max(4, .35 * df.shape[0] + 2)))
    sns.heatmap(df, annot=annot, fmt=".2f" if annot and df.dtypes.iloc[0] == float else "d" if annot else "",
                cmap="viridis")
    plt.title(title); _save(name)


def plot_robustness(curves):
    plt.figure(figsize=(6, 4))
    for label, (x, y) in curves.items():
        plt.plot(x, y, label=label)
    plt.xlabel("fraction of nodes removed"); plt.ylabel("giant component fraction"); plt.legend(); plt.grid(alpha=.3)
    _save("E_robustness")
