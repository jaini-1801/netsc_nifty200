"""Run the full pipeline:  python main.py"""
import os, random, json
import numpy as np, pandas as pd, networkx as nx
from src import config as C
from src.data_loader import get_data
from src.network_builder import build_graph
from src import analysis as A, sensitivity as S, extras as X, plotting as P

for d in (C.DATA_DIR, C.FIG_DIR, C.TAB_DIR):
    os.makedirs(d, exist_ok=True)
random.seed(C.SEED); np.random.seed(C.SEED)
T = lambda name: f"{C.TAB_DIR}/{name}.csv"

# 1. data -> returns -> correlation
ret, sector, dropped = get_data()
print(f"Returns: {ret.shape[0]} days x {ret.shape[1]} stocks | dropped {len(dropped)}: {dropped}")
corr = ret.corr(method="pearson")

# 2. threshold sweep (Task D) -- also justifies R_MAIN
sweep = S.sweep(corr, sector); sweep.to_csv(T("D_sweep"), index=False); P.plot_sweep(sweep)
print("\n", sweep.round(3).to_string(index=False))

# 3. main graph + stats
G = build_graph(corr, C.R_MAIN, sector)
stats = A.basic_stats(G); pd.Series(stats).to_csv(T("network_stats"))
print("\nMain network:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}, indent=1))

# 4. Task A: degree distribution
P.plot_degree(G)
try:
    fit = A.degree_fit(G); pd.Series(fit).to_csv(T("A_powerlaw_fit")); print("\nPower-law fit:", fit)
except ImportError:
    print("pip install powerlaw")

# 5. Task B: centralities
cent = A.centralities(G, sector); cent.to_csv(T("centralities_all"))
for c in ["degree", "betweenness_hop", "betweenness_dist", "eigenvector"]:
    cent.nlargest(10, c)[[c, "sector"]].to_csv(T(f"B_top10_{c}"))
    print(f"\nTop 10 by {c}:\n", cent.nlargest(10, c)[[c, "sector"]])
hw = A.heavyweight_ranks(cent); hw.to_csv(T("B_heavyweights")); print("\nHeavyweights:\n", hw.round(3))
P.plot_heatmap(cent.drop(columns="sector").corr(method="spearman"), "B_centrality_corr",
               "Spearman correlation between centralities", annot=True)

# 6. Task C: clustering, homophily, communities
cl = A.clustering_vs_null(G); hom = A.sector_homophily(G)
csum, ctab, lab = A.communities(G, sector)
pd.Series({**cl, **hom, **csum}).to_csv(T("C_summary")); ctab.to_csv(T("C_community_vs_sector"))
print("\nClustering vs null:", cl, "\nHomophily:", hom, "\nCommunities:", csum)
P.plot_heatmap(ctab, "C_community_sector_heatmap", "Community x sector (stock counts)", annot=True)
P.plot_network_by_sector(G, sector, cent.nlargest(10, "degree").index)

# 7. extras
print("\nMarket mode:", X.market_mode(corr))
P.plot_robustness({"random": X.attack_curve(G, False), "targeted (degree)": X.attack_curve(G, True)})
ts = X.temporal_stability(ret, sector); print("Temporal stability:", ts)
M = X.mst(corr); print("MST highest-degree nodes:", sorted(M.degree, key=lambda x: -x[1])[:5])
print("\nDone. Tables in out/tables, figures in out/figures.")
