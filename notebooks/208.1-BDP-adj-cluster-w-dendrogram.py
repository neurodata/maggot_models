# %% [markdown]
# ##
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from anytree import Node, NodeMixin, Walker
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


CLASS_KEY = "simple_class"  # "merge_class"
group_order = "median_node_visits"
FORMAT = "pdf"


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


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
# meta["merge_class"] = meta["simple_class"]  # HACK


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
meta["median_node_visits"] = walk_meta["median_node_visits"]  # make the sorting right


lowest_level = 7
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]

#%%


def sort_meta(meta, group, group_order=group_order, item_order=[], ascending=True):
    sort_class = group
    group_order = [group_order]
    total_sort_by = []
    for sc in sort_class:
        for co in group_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    meta = meta.sort_values(total_sort_by, ascending=ascending)
    return meta


sorted_meta = meta.copy()
sorted_meta["sort_inds"] = np.arange(len(sorted_meta))
group = level_names + ["merge_class"]
sorted_meta = sort_meta(
    sorted_meta,
    group,
    group_order=group_order,
    item_order=["merge_class", "median_node_visits"],
)
sorted_meta["new_inds"] = np.arange(len(sorted_meta))
sorted_meta[["merge_class", "lvl7_labels", "median_node_visits"]]

sort_inds = sorted_meta["sort_inds"]
sorted_adj = adj[np.ix_(sort_inds, sort_inds)]


class MetaNode(NodeMixin):
    def __init__(self, name, parent=None, children=None, meta=None):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children
        self.meta = meta

    def hierarchical_mean(self, key):
        if self.is_leaf:
            meta = self.meta
            var = meta[key]
            return np.mean(var)
        else:
            children = self.children
            child_vars = [child.hierarchical_mean(key) for child in children]
            return np.mean(child_vars)


def get_parent_label(label):
    if len(label) <= 1:
        return None
    elif label[-1] == "-":
        return label[:-1]
    else:  # then ends in a -number
        return label[:-2]


def make_node(label, node_map):
    if label not in node_map:
        node = MetaNode(label)
        node_map[label] = node
    else:
        node = node_map[label]
    return node


meta = sorted_meta
node_map = {}
for i in range(lowest_level, -1, -1):
    level_labels = meta[f"lvl{i}_labels"].unique()
    for label in level_labels:
        node = make_node(label, node_map)
        node.meta = meta[meta[f"lvl{i}_labels"] == label]
        parent_label = get_parent_label(label)
        if parent_label is not None:
            parent = make_node(parent_label, node_map)
            node.parent = parent

root = node

#%%


def get_x_y(xs, ys, orientation):
    if orientation == "h":
        return xs, ys
    elif orientation == "v":
        return (ys, xs)


def plot_dendrogram(
    ax, root, index_key="_new_idx", orientation="h", linewidth=0.7, cut=None
):
    for node in (root.descendants) + (root,):
        y = node.hierarchical_mean(index_key)
        x = node.depth
        node.y = y
        node.x = x

    walker = Walker()
    walked = []

    for node in root.leaves:
        upwards, common, downwards = walker.walk(node, root)
        curr_node = node
        for up_node in (upwards) + (root,):
            edge = (curr_node, up_node)
            if edge not in walked:
                xs = [curr_node.x, up_node.x]
                ys = [curr_node.y, up_node.y]
                xs, ys = get_x_y(xs, ys, orientation)
                ax.plot(
                    xs,
                    ys,
                    linewidth=linewidth,
                    color="black",
                    alpha=1,
                )
                walked.append(edge)
            curr_node = up_node
        y_max = node.meta[index_key].max()
        y_min = node.meta[index_key].min()
        xs = [node.x, node.x, node.x + 1, node.x + 1]
        ys = [node.y - 3, node.y + 3, y_max, y_min]
        xs, ys = get_x_y(xs, ys, orientation)
        ax.fill(xs, ys, facecolor="black")

    if orientation == "h":
        ax.set(xlim=(-1, lowest_level + 1))
        if cut is not None:
            ax.axvline(cut - 1, linewidth=1, color="grey", linestyle=":")
    elif orientation == "v":
        ax.set(ylim=(lowest_level + 1, -1))
        if cut is not None:
            ax.axhline(cut - 1, linewidth=1, color="grey", linestyle=":")

    ax.axis("off")


#%%
cut = None  # where to draw a dashed line on the dendrogram
line_level = 6
level = lowest_level

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

ax = axs[0]
ax, divider, top, _ = adjplot(
    sorted_adj,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    sort_class=group[:line_level],
    item_order="new_inds",
    class_order=group_order,
    meta=meta,
    palette=CLASS_COLOR_DICT,
    colors=CLASS_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),  # 0.2
)

left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, root, orientation="h", cut=cut)

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, root, orientation="v", cut=cut)


model = DCSBMEstimator
adj = sorted_adj.copy()
adj = binarize(adj)
meta = sorted_meta
labels = meta[f"lvl{level}_labels_side"].values
estimator = model(directed=True, loops=True)
uni_labels, inv = np.unique(labels, return_inverse=True)
estimator.fit(adj, inv)
sample_adj = np.squeeze(estimator.sample())

ax = axs[1]
ax, divider, top, _ = adjplot(
    sample_adj,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    sort_class=group[:line_level],
    item_order="new_inds",
    class_order=group_order,
    meta=meta,
    palette=CLASS_COLOR_DICT,
    colors=CLASS_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),
)

left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, root, orientation="h", cut=cut)

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, root, orientation="v", cut=cut)

stashfig(
    f"both-heatmap-w-dendrogram-level={level}-line_level={line_level}-cut={cut}-classes={CLASS_KEY}",
    dpi=300,
)


#%% new
# load nblast scores/similarities
from src.nblast import preprocess_nblast

data_dir = Path("maggot_models/experiments/nblast/outs")

symmetrize_mode = "geom"
transform = "ptr"
nblast_type = "scores"

side = "left"
nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-{nblast_type}.csv", index_col=0)
nblast_sim.columns = nblast_sim.columns.values.astype(int)
print(f"{len(nblast_sim)} neurons in NBLAST data on {side}")
# get neurons that are in both
left_intersect_index = np.intersect1d(meta.index, nblast_sim.index)
print(f"{len(left_intersect_index)} neurons in intersection on {side}")
# reindex appropriately
nblast_sim = nblast_sim.reindex(
    index=left_intersect_index, columns=left_intersect_index
)
sim = preprocess_nblast(
    nblast_sim.values, symmetrize_mode=symmetrize_mode, transform=transform
)
left_sim = pd.DataFrame(data=sim, index=nblast_sim.index, columns=nblast_sim.index)

side = "right"
nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-{nblast_type}.csv", index_col=0)
nblast_sim.columns = nblast_sim.columns.values.astype(int)
print(f"{len(nblast_sim)} neurons in NBLAST data on {side}")
# get neurons that are in both
right_intersect_index = np.intersect1d(meta.index, nblast_sim.index)
print(f"{len(right_intersect_index)} neurons in intersection on {side}")
# reindex appropriately
nblast_sim = nblast_sim.reindex(
    index=right_intersect_index, columns=right_intersect_index
)
sim = preprocess_nblast(
    nblast_sim.values, symmetrize_mode=symmetrize_mode, transform=transform
)
right_sim = pd.DataFrame(data=sim, index=nblast_sim.index, columns=nblast_sim.index)


from giskard.stats import calc_discriminability_statistic

level_key = f"lvl{level}_labels"

left_meta = meta.loc[left_intersect_index]
left_clustering = left_meta[level_key].values
left_sim = left_sim.reindex(index=left_meta.index, columns=left_meta.index)
left_total_discrim, left_cluster_discrim = calc_discriminability_statistic(
    1 - left_sim.values, left_clustering
)

right_meta = meta.loc[right_intersect_index]
right_clustering = right_meta[level_key].values
right_sim = right_sim.reindex(index=right_meta.index, columns=right_meta.index)
right_total_discrim, right_cluster_discrim = calc_discriminability_statistic(
    1 - right_sim.values, right_clustering
)

uni_clusters = np.unique(meta[level_key])

mean_cluster_discrim = {}
for cluster_label in uni_clusters:
    mean_cluster_discrim[cluster_label] = (
        left_cluster_discrim[cluster_label] + right_cluster_discrim[cluster_label]
    ) / 2


# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# left_meta
# meta["new_inds"] = np.arange(len(meta))

ax, divider, top, _ = adjplot(
    left_sim.values,
    ax=ax,
    plot_type="heatmap",
    sizes=(0.5, 0.5),
    sort_class=group[:line_level],
    item_order="new_inds",
    class_order=group_order,
    meta=left_meta,
    palette=CLASS_COLOR_DICT,
    colors=CLASS_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),
    cbar_kws=dict(shrink=0.7),
    title="NBLAST similarity",
)

node_map = {}
for i in range(lowest_level, -1, -1):
    level_labels = meta[f"lvl{i}_labels"].unique()
    for label in level_labels:
        node = make_node(label, node_map)
        node.meta = meta[meta[f"lvl{i}_labels"] == label]
        parent_label = get_parent_label(label)
        if parent_label is not None:
            parent = make_node(parent_label, node_map)
            node.parent = parent

stashfig("nblast-sim-grouped")

# root = node
# root.hierarchical_mean("new_inds")


# left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
# plot_dendrogram(left_ax, root, orientation="h", cut=cut)

# top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
# plot_dendrogram(top_ax, root, orientation="v", cut=cut)
