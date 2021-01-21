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


CLASS_KEY = "merge_class"
group_order = "median_node_visits"
FORMAT = "pdf"


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


def sort_mg(mg, group, group_order=group_order, ascending=True):
    """Required sorting prior to plotting the dendrograms

    Parameters
    ----------
    mg : [type]
        [description]
    group : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    meta = mg.meta
    sort_class = group + ["merge_class"]
    group_order = [group_order]
    total_sort_by = []
    for sc in sort_class:
        for co in group_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    mg = mg.sort_values(total_sort_by, ascending=ascending)  # TODO used to be False!
    return mg


def plot_adjacencies(full_mg, axs, lowest_level=7):
    group = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
    pal = sns.color_palette("deep", 1)
    model = DCSBMEstimator
    for level in np.arange(lowest_level + 1):
        ax = axs[0, level]
        adj = binarize(full_mg.adj)
        # [f"lvl{level}_labels", f"merge_class_sf_order", "merge_class"]
        _, _, top, _ = adjplot(
            adj,
            ax=ax,
            plot_type="scattermap",
            sizes=(0.5, 0.5),
            sort_class=group[: level + 1],
            # item_order=[f"{CLASS_KEY}_{group_order}_order", CLASS_KEY, group_order],
            group_order=group_order,
            meta=full_mg.meta,
            palette=CLASS_COLOR_DICT,
            colors=CLASS_KEY,
            ticks=False,
            gridline_kws=dict(linewidth=0, color="grey", linestyle="--"),  # 0.2
            color=pal[0],
        )
        top.set_title(f"Level {level} - Data")

        labels = full_mg.meta[f"lvl{level}_labels_side"]
        estimator = model(directed=True, loops=True)
        uni_labels, inv = np.unique(labels, return_inverse=True)
        estimator.fit(adj, inv)
        sample_adj = np.squeeze(estimator.sample())
        ax = axs[1, level]
        _, _, top, _ = adjplot(
            sample_adj,
            ax=ax,
            plot_type="scattermap",
            sizes=(0.5, 0.5),
            sort_class=group[: level + 1],
            item_order=[f"{CLASS_KEY}_{group_order}_order", CLASS_KEY, group_order],
            group_order=group_order,
            meta=full_mg.meta,
            palette=CLASS_COLOR_DICT,
            colors=CLASS_KEY,
            ticks=False,
            gridline_kws=dict(linewidth=0, color="grey", linestyle="--"),  # 0.2
            color=pal[0],
        )
        top.set_title(f"Level {level} - DCSBM sample")


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

# #%%

# pal = sns.color_palette("deep", 1)
# model = DCSBMEstimator
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# level = lowest_level
# _, divider, top, _ = adjplot(
#     adj,
#     ax=ax,
#     plot_type="scattermap",
#     sizes=(0.5, 0.5),
#     sort_class=["hemisphere"] + level_names[: level + 1],
#     # item_order=[f"{CLASS_KEY}_{group_order}_order", CLASS_KEY, group_order],
#     class_order=group_order,
#     meta=meta,
#     palette=CLASS_COLOR_DICT,
#     colors=CLASS_KEY,
#     ticks=False,
#     gridline_kws=dict(linewidth=0, color="grey", linestyle="--"),  # 0.2
#     color=pal[0],
# )
# # top.set_title(f"Level {level} - Data")

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

# # bounds = sorted_meta.groupby(level_names, sort=False).first()["inds"].values
# centers = sorted_meta.groupby(level_names, sort=False).mean()["new_inds"]
# fig, ax = plt.subplots(1, 1, figsize=(4, 10))
# ax.set(xlim=(-1, 10), ylim=(-10, len(meta) + 10))
# for i in range(lowest_level, -1, -1):
#     # level_centers = centers.groupby(f"lvl{i}_labels").mean()
#     level_centers = sorted_meta.groupby(f"lvl{i}_labels", sort=False).mean()["new_inds"]
#     print(level_centers)
#     ax.scatter(np.full(len(level_centers), i), level_centers, s=1)

#%%

# loc_meta = sorted_meta[level_names + ["new_inds"]]
# loc_meta.groupby("lvl7_labels").mean()

#%%
# from matplotlib.collections import LineCollection

# centers = sorted_meta.groupby(f"lvl{lowest_level}_labels", sort=False).mean()[
#     "new_inds"
# ]
# label_map = (
#     loc_meta.groupby(f"lvl{lowest_level}_labels").first().drop(columns="new_inds")
# )
# label_map["mean_inds"] = label_map.index.map(centers)
# label_map = label_map.reset_index()
# label_map
# i = 1
# levels = level_names[: len(level_names) - i]
# label_map = label_map.groupby()["mean_inds"].mean()
# centers = label_map.reset_index("")

# fig, ax = plt.subplots(1, 1, figsize=(4, 10))
# ax.set(xlim=(-1, 10), ylim=(-10, len(meta) + 10))

# for i in range(lowest_level, 4, -1):
#     # current_means = dict(zip(label_map[f"lvl{i}_labels"], label_map["mean_inds"]))
#     current_means = label_map["mean_inds"]

#     # horizontals
#     xs = np.full(len(current_means), i)
#     ys = current_means
#     starts = list(zip(xs, ys))
#     ends = list(zip(xs - 0.5, ys))
#     lines = list(zip(starts, ends))

#     # verticals
#     xs = xs - 0.5

#     lc = LineCollection(lines, linewidth=0.5, color="black", edgecolor="black")
#     ax.add_collection(lc)
#     if i != 0:
#         next_level_means = label_map.groupby(f"lvl{i-1}_labels")["mean_inds"].mean()
#         label_map = label_map.drop(columns=f"lvl{i}_labels")
#         label_map = label_map.groupby(f"lvl{i-1}_labels").first()
#         label_map["mean_inds"] = next_level_means
#         label_map = label_map.reset_index()


#     label_map = label_map.set_index(f"lvl{i}_labels")
#
#     # label_map = label_map.reset_index()
#     centers = pd.DataFrame(label_map.groupby(f"lvl{i}_labels")["mean_inds"].mean())
#     centers["next_level"] = centers.index.map(label_map[f"lvl{i-1}_labels"])
#     centers = centers.reset_index(drop=True).set_index(f"lvl{i-1}_labels")


node_map = {}


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


def make_node(label):
    if label not in node_map:
        node = MetaNode(label)
        node_map[label] = node
    else:
        node = node_map[label]
    return node


meta = sorted_meta

for i in range(lowest_level, -1, -1):
    level_labels = meta[f"lvl{i}_labels"].unique()
    for label in level_labels:
        node = make_node(label)
        node.meta = meta[meta[f"lvl{i}_labels"] == label]
        parent_label = get_parent_label(label)
        if parent_label is not None:
            parent = make_node(parent_label)
            node.parent = parent

root = node
root.hierarchical_mean("new_inds")


#%%
cut = 5  # where to draw a dashed line on the dendrogram

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
level = lowest_level
ax, divider, top, _ = adjplot(
    sorted_adj,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    sort_class=group[:cut],
    item_order="new_inds",
    class_order=group_order,
    meta=meta,
    palette=CLASS_COLOR_DICT,
    colors=CLASS_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),  # 0.2
)


def get_x_y(xs, ys, orientation):
    if orientation == "h":
        return xs, ys
    elif orientation == "v":
        return (ys, xs)


def plot_dendrogram(ax, root, orientation="h", linewidth=0.7, cut=None):
    for node in (root.descendants) + (root,):
        y = node.hierarchical_mean("new_inds")
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
        y_max = node.meta["new_inds"].max()
        y_min = node.meta["new_inds"].min()
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


left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, root, orientation="h", cut=cut)

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, root, orientation="v", cut=cut)

stashfig(f"heatmap-w-dendrogram-cut={cut}", dpi=300)
