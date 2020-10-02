#%% [markdown]
## gzipping a connectome
# > A (perhaps silly) experiment running gzip on some connectome edgelists
#
# - toc: false
# - badges: false
# - categories: [pedigo, graspologic]
# - hide: false
# - search_exclude: false
#%%
import gzip
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from neuropull import load_witvilet_2020
from src.io import savefig
from src.visualization import set_theme

FNAME = os.path.basename(__file__)[:-3]

set_theme()


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def relabel_nodes(g, method="degree"):
    nodelist = np.array(list(sorted(g.nodes())))
    if method == "degree":
        degree_map = dict(g.degree(nodelist))
        degrees = list(map(degree_map.get, nodelist))
        sort_inds = np.argsort(degrees)[::-1]
    elif method == "random":
        sort_inds = np.random.permutation(len(g))
    node_name_remap = dict(zip(nodelist[sort_inds], range(len(g))))
    renamed_g = nx.relabel_nodes(g, node_name_remap, copy=True)
    return renamed_g


def get_bytes_edgelist(g, weight=False):
    if weight:
        data = ["weight"]
    else:
        data = False
    edgelist_lines = ""
    for line in nx.generate_edgelist(g, delimiter=" ", data=data):
        edgelist_lines += line + "\n"
    return bytes(edgelist_lines, encoding="utf8")


def calc_edgelist_n_bytes(g, weight=False):
    edgelist_bytes = get_bytes_edgelist(g, weight=weight)
    n_bytes = len(edgelist_bytes)
    edgelist_bytes_compressed = gzip.compress(edgelist_bytes)
    n_bytes_compressed = len(edgelist_bytes_compressed)
    return n_bytes, n_bytes_compressed


# Maggot brain connectome dataset (data not yet public)
edgelist_loc = "maggot_models/data/processed/2020-09-23/G.edgelist"

maggot_g = nx.read_weighted_edgelist(
    edgelist_loc, delimiter=" ", nodetype=int, create_using=nx.DiGraph
)

# C. elegans data from Witvilet et al. 2020
elegans_graphs = load_witvilet_2020()

adult_celegans_g1 = elegans_graphs[-2]
adult_celegans_g2 = elegans_graphs[-1]

g = adult_celegans_g1.copy()
chemical_g = nx.DiGraph(
    ((u, v, e) for u, v, e in g.edges(data=True) if e["type"] == "chemical")
)

graphs = [adult_celegans_g1, adult_celegans_g2, maggot_g]
graph_names = ["C. elegans adult 1", "C. elegans adult 2", "Maggot brain"]

rows = []
for graph, graph_name in zip(graphs, graph_names):
    for weight in [True, False]:
        degree_relabeled_graph = relabel_nodes(graph, method="degree")
        n_bytes, n_bytes_compressed = calc_edgelist_n_bytes(
            degree_relabeled_graph, weight=weight
        )
        rows.append(
            {
                "n_bytes": n_bytes,
                "n_bytes_compressed": n_bytes_compressed,
                "method": "Degree",
                "weight": weight,
                "graph": graph_name,
            }
        )
        for i in range(10):
            random_relabeled_graph = relabel_nodes(graph, method="random")
            n_bytes, n_bytes_compressed = calc_edgelist_n_bytes(
                random_relabeled_graph, weight=weight
            )
            rows.append(
                {
                    "n_bytes": n_bytes,
                    "n_bytes_compressed": n_bytes_compressed,
                    "method": "Random",
                    "weight": weight,
                    "graph": graph_name,
                }
            )

n_bytes_df = pd.DataFrame(rows)
n_bytes_df["compression_ratio"] = (
    n_bytes_df["n_bytes_compressed"] / n_bytes_df["n_bytes"]
)
n_bytes_df
#%%
fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

for i, weight in enumerate([True, False]):
    for j, graph_name in enumerate(graph_names):
        ax = axs[i, j]
        plot_df = n_bytes_df[n_bytes_df["graph"] == graph_name]
        plot_df = plot_df[plot_df["weight"] == weight]
        sns.stripplot(x="method", y="compression_ratio", data=plot_df, ax=ax)
        ax.set_ylabel("")
        ax.set(ylabel="", xlabel="")
        if i == 0:
            ax.set_title(graph_name, fontweight="bold", fontsize="large")
        if j == 0:
            if weight:
                ylabel = "Weighted"
            else:
                ylabel = "Unweighted"
            ax.set_ylabel(ylabel)

fig.text(
    0.07,
    0.5,
    "Compression ratio",
    rotation=90,
    fontsize="large",
    fontweight="bold",
    ha="center",
    va="center",
)
fig.text(
    0.52,
    0.05,
    "Node labeling method",
    rotation=0,
    fontsize="large",
    fontweight="bold",
    ha="center",
    va="center",
)
stashfig("edgelist-compression")