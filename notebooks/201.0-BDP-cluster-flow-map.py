# %% [markdown]
# ##
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.embed import OmnibusEmbed, selectSVD
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    to_laplace,
)
from scipy.stats import poisson
from topologic.io import tensor_projection_writer

from src.cluster import BinaryCluster
from src.data import load_metagraph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.utils import get_paired_inds
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
    set_theme,
)

# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/evaluate_clustering/")

CLASS_KEY = "merge_class"
CLASS_ORDER = "median_node_visits"


def stashfig(name, fmt="pdf", **kws):
    savefig(name, pathname=save_path / "figs", fmt=fmt, save_on=True, **kws)


# %% [markdown]
# ##
omni_method = "color_iso"
d = 8
bic_ratio = 0.95
min_split = 32

basename = f"-method={omni_method}-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"
meta = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/meta{basename}.csv",
    index_col=0,
)
meta["lvl0_labels"] = meta["lvl0_labels"].astype(str)
adj_df = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/adj{basename}.csv",
    index_col=0,
)
adj = adj_df.values

name_map = {
    "Sens": "Sensory",
    "LN": "Local",
    "PN": "Projection",
    "KC": "Kenyon cell",
    "LHN": "Lateral horn",
    "MBIN": "MBIN",
    "Sens2o": "2nd order sensory",
    "unk": "Unknown",
    "MBON": "MBON",
    "FBN": "MB feedback",
    "CN": "Convergence",
    "PreO": "Pre-output",
    "Outs": "Output",
    "Motr": "Motor",
}
meta["simple_class"] = meta["simple_class"].map(name_map)
print(meta["simple_class"].unique())
meta["merge_class"] = meta["simple_class"]  # HACK


graph_type = "Gad"
n_init = 256
max_hops = 16
allow_loops = False
include_reverse = False
walk_spec = f"gt={graph_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}"
walk_meta = pd.read_csv(
    f"maggot_models/experiments/walk_sort/outs/meta_w_order-{walk_spec}-include_reverse={include_reverse}.csv",
    index_col=0,
)
meta["median_node_visits"] = walk_meta["median_node_visits"]

walk_path = "maggot_models/experiments/walk_sort/outs/walks-"
forward_walk_path = walk_path + walk_spec + "-reverse=False" + ".txt"

#%%


def process_paths(walk_path):
    with open(walk_path, "r") as f:
        paths = f.read().splitlines()
    print(f"# of paths: {len(paths)}")

    paths = list(set(paths))
    paths.remove("")  # TODO maybe not?
    print(f"# of paths after removing duplicates: {len(paths)}")

    paths = [path.split(" ") for path in paths]
    paths = [[int(node) for node in path] for path in paths]
    return paths


forward_paths = process_paths(forward_walk_path)

#%%
all_edges = set()
edge_visits = {}
for path in forward_paths:
    for i, edge in enumerate(nx.utils.pairwise(path)):
        if edge not in edge_visits:
            edge_visits[edge] = []
        edge_visits[edge].append(i)
# [[all_nodes.add(node) for node in path] for path in forward_paths]
all_edges = list(set(edge_visits.keys()))

#%%

# median_node_visits = {}
# for edge in all_edges:
#     median_node_visits[edge] = np.median(edge_visits[edge])

#%%
mg = load_metagraph("G")
edgelist = nx.to_pandas_edgelist(mg.g)
#%%
edgelist = edgelist.set_index(["source", "target"])
#%%
for i in range(max_hops):
    edgelist[f"p_visit_hop_{i}"] = 0

for edge in all_edges:
    visits = edge_visits[edge]
    visit_orders, counts = np.unique(visits, return_counts=True)
    # probs = counts / counts.sum()
    for visit_order, count in zip(visit_orders, counts):
        edgelist.loc[edge, f"p_visit_hop_{visit_order}"] = count

#%%
level = 7
edgelist = edgelist.reset_index()
#%%
edgelist["source_cluster"] = edgelist["source"].map(meta[f"lvl{level}_labels"])
edgelist["target_cluster"] = edgelist["target"].map(meta[f"lvl{level}_labels"])
#%%
cluster_edge_counts = (
    edgelist.groupby(["source_cluster", "target_cluster"])
    .sum()
    .drop(["source", "target"], axis=1)
)

#%%
level_key = f"lvl{level}_labels"
color_key = "merge_class"

cluster_visit_order = meta.groupby(level_key)["median_node_visits"].mean()
meta["cluster_visit_order"] = meta[level_key].map(cluster_visit_order)

class_visit_order = meta.groupby(color_key)["median_node_visits"].mean()
meta["class_visit_order"] = meta[color_key].map(class_visit_order)

meta = meta.sort_values(
    ["cluster_visit_order", level_key, "class_visit_order", "median_node_visits"]
)

sizes = meta.groupby([level_key, color_key], sort=False).size()

clusters = sizes.index.unique(level=0)

gap = 40
n_nodes = sizes.sum()

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
max_hops = 10
y_max = n_nodes + gap * len(clusters)
ax.set_xlim((-1, 2 * max_hops))
ax.set_ylim((y_max, 0))
width = 0.5


def calc_bar_params(sizes, label, top, palette=None):
    if palette is None:
        palette = CLASS_COLOR_DICT
    heights = sizes.loc[label]
    offset = top  # - n_in_bar / 2
    starts = heights.cumsum() - heights + offset
    colors = np.vectorize(palette.get)(heights.index)
    return heights, starts, colors


# draw cluster bars
mids = []
mid_map = {}
for x in np.arange(2 * max_hops, step=2):
    top = 0
    for i, cluster in enumerate(clusters):
        cluster_sizes = sizes.loc[cluster]
        heights, starts, colors = calc_bar_params(
            sizes, cluster, top, palette=CLASS_COLOR_DICT
        )
        if x == 0:
            minimum = starts[0]
            maximum = starts[-1] + heights[-1]
            mid = (minimum + maximum) / 2
            mids.append(mid)
            mid_map[cluster] = mid
        for i in range(len(heights)):
            ax.bar(
                x=x,
                height=heights[i],
                width=width,
                bottom=starts[i],
                color=colors[i],
            )
        top += heights.sum() + gap


# cluster_edge_counts = cluster_edge_counts.reset_index()
cluster_edges = cluster_edge_counts.reset_index().melt(
    id_vars=["source_cluster", "target_cluster"],
    value_vars=[f"p_visit_hop_{i}" for i in range(max_hops)],
    value_name="visit_weight",
    var_name="hop",
)
cluster_edges["hop"] = cluster_edges["hop"].apply(
    lambda x: int(x.replace("p_visit_hop_", ""))
)
# cluster_edges


edge_info = []
for i, row in cluster_edges.iterrows():
    ax.plot(
        [2 * row["hop"] + width / 2, 2 * row["hop"] + 2 - width / 2],
        [mid_map[row["source_cluster"]], mid_map[row["target_cluster"]]],
        color="black",
        alpha=0.01 * row["visit_weight"],
        zorder=-1,
        linewidth=0.1,
    )
ax.axis("off")
stashfig("line-barplot")