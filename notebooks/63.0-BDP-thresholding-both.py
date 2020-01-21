# %% [markdown]
# # Imports
import json
import os
import pickle
import random
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob, invert_permutation
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True
BRAIN_VERSION = "2020-01-16"

sns.set_context("talk")

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


def r():
    return random.randint(0, 255)


def extract_ids(lod):
    out_list = []
    for d in lod:
        skel_id = d["skeleton_id"]
        out_list.append(skel_id)
    return out_list


def get_edges(adj):
    adj = adj.copy()
    all_edges = []
    for i in range(adj.shape[0]):
        row = adj[i, :]
        col = adj[:, i]
        col = np.delete(col, i)
        edges = np.concatenate((row, col))
        all_edges.append(edges)
    return all_edges


# %% [markdown]
# # Play with getting edge df representatino
graph_type = "Gad"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)

g = mg.g
meta = mg.meta
edgelist_df = mg.to_edgelist()

max_pair_edge_df = edgelist_df.groupby("edge pair ID").max()
edge_max_weight_map = dict(
    zip(max_pair_edge_df.index.values, max_pair_edge_df["weight"])
)
edgelist_df["max_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
    edge_max_weight_map
)

# %% [markdown]
# #


def stable_rank(A):
    f_norm = np.linalg.norm(A, ord="fro")
    spec_norm = np.linalg.norm(A, ord=2)
    return (f_norm ** 2) / (spec_norm ** 2)


n_edges = len(edgelist_df)
g_raw = nx.from_pandas_edgelist(edgelist_df, edge_attr=True, create_using=nx.DiGraph)
raw_adj = nx.to_numpy_array(g_raw)
edge_inds = np.nonzero(raw_adj)

rows = []
for t in range(10):
    print(t)
    thresh_df = edgelist_df[edgelist_df["max_weight"] > t]
    g_thresh = nx.from_pandas_edgelist(
        thresh_df, edge_attr=True, create_using=nx.DiGraph
    )
    thresh_adj = nx.to_numpy_array(g_thresh)
    sr = stable_rank(thresh_adj)
    n_edges_thresh = len(thresh_df)
    n_removed = n_edges - n_edges_thresh
    rows.append({"stable rank": sr, "type": "true", "threshold": t})
    for i in range(10):
        remove_inds = np.random.choice(len(edge_inds[0]), size=n_removed, replace=False)
        x_inds = edge_inds[0][remove_inds]
        y_inds = edge_inds[1][remove_inds]
        rand_thresh_adj = raw_adj.copy()
        rand_thresh_adj[x_inds, y_inds] = 0
        sr = stable_rank(rand_thresh_adj)
        rows.append({"stable rank": sr, "type": "random", "threshold": t})
# %% [markdown]
# #
plt.figure(figsize=(10, 5))
df = pd.DataFrame(rows)
sns.stripplot(data=df, x="threshold", y="stable rank", hue="type")
# %% [markdown]
# #
heatmap(raw_adj, transform="binarize", figsize=(20, 20))
heatmap(thresh_adj, transform="binarize", figsize=(20, 20))
print(stable_rank(raw_adj))
print(stable_rank(thresh_adj))


# %% [markdown]
# #
A = np.ones((10, 10))
stable_rank(A)

# %% [markdown]
# # Try thresholding in this new format
props = []
prop_edges = []
prop_syns = []
threshs = np.linspace(0, 0.3, 20)
for threshold in threshs:
    thresh_df = max_pair_edge_df[max_pair_edge_df["weight"] > threshold]
    prop = len(thresh_df[thresh_df["edge pair counts"] == 2]) / len(thresh_df)
    props.append(prop)
    prop_edges_left = (
        thresh_df["edge pair counts"].sum() / max_pair_edge_df["edge pair counts"].sum()
    )
    prop_edges.append(prop_edges_left)
    temp_df = edgelist_df[edgelist_df["max_weight"] > threshold]
    p_syns = temp_df["weight"].sum() / edgelist_df["weight"].sum()
    prop_syns.append(p_syns)
# %% [markdown]
# #

plot_df = pd.DataFrame()
plot_df["Threshold"] = threshs
plot_df["P(mirror edge present)"] = props
plot_df["Proportion of edges left"] = prop_edges
plot_df["Proportion of synapses left"] = prop_syns

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

sns.lineplot(data=plot_df, x="Threshold", y="P(mirror edge present)", ax=ax)
ax_right = ax.twinx()
sns.lineplot(
    data=plot_df,
    x="Threshold",
    y="Proportion of synapses left",
    ax=ax_right,
    color="orange",
)
remove_spines(ax_right)
remove_spines(ax)
ax.set_ylim((0, 1))
ax_right.set_ylim((0, 1))
ax.set_title(f"{graph_type}")
stashfig(f"thresh-sweep-{graph_type}-brain-syns")

# %% [markdown]
# # Plot these against each other
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=plot_df, x="P(mirror edge present)", y="Proportion of edges left")
ax.set_xlim((0, 1.1))
ax.set_ylim((0, 1.1))
remove_spines(ax)
stashfig(f"thresh-sweep-paired-{graph_type}-brain")
# %% [markdown]
# # Look at IOU
threshold = 0.05
thresh_df = max_pair_edge_df[max_pair_edge_df["weight"] > threshold]

source_pair_ids = np.unique(max_pair_edge_df["source Pair ID"])
target_pair_ids = np.unique(max_pair_edge_df["target Pair ID"])
pair_ids = np.union1d(source_pair_ids, target_pair_ids)

ious = []
kept_pairs = []
removed_pairs = []
edge_dfs = []
keep_cols = [
    "source",
    "target",
    "weight",
    "source Pair ID",
    "target Pair ID",
    "edge pair counts",
]

for pid in pair_ids:
    temp_df = thresh_df[
        (thresh_df["source Pair ID"] == pid) | (thresh_df["target Pair ID"] == pid)
    ]

    if len(temp_df) > 0:
        iou = len(temp_df[temp_df["edge pair counts"] == 2]) / len(temp_df)
        ious.append(iou)
        kept_pairs.append(pid)
        edge_dfs.append(temp_df[keep_cols])
    else:
        removed_pairs.append(pid)

# inds = np.argsort(ious)

# edge_dfs[inds]

# %% [markdown]
# #
plot_df = pd.DataFrame()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot(ious, norm_hist=False, kde=False)
remove_spines(ax)
ax.set_xlabel("IOU Score")
ax.set_xlim((0, 1))
ax.set_title(f"{graph_type}, threshold = {threshold:0.2f}")
ax.set_ylabel("Pair count")

# %% [markdown]
# #
iou_df = pd.DataFrame()
iou_df["IOU Score"] = ious
iou_df["Pair ID"] = kept_pairs
iou_df["Left ID"] = -1
iou_df["Right ID"] = -1
for i, pid in enumerate(kept_pairs):
    temp_df = meta[meta["Pair ID"] == pid]
    iou_df.loc[i, "Left ID"] = temp_df[temp_df["Hemisphere"] == "L"].index[0]
    iou_df.loc[i, "Right ID"] = temp_df[temp_df["Hemisphere"] == "R"].index[0]

# %% [markdown]
# #
iou_df.sort_values("IOU Score", ascending=True, inplace=True)
iou_df.index = range(len(iou_df))

colors = []
ids = []
for i in range(100):
    left_id = iou_df.loc[i, "Left ID"]
    right_id = iou_df.loc[i, "Right ID"]
    iou = iou_df.loc[i, "IOU Score"]
    c = int(iou * 255.0)
    print(c)
    print(iou)
    hex_color = "#%02X%02X%02X" % (c, c, c)
    colors.append(hex_color)
    colors.append(hex_color)
    ids.append(left_id)
    ids.append(right_id)

stashskel("low-iou-sorted-pairs", ids, colors, colors=colors, palette=None)

#%%
left_id = 12121795
pair_id = meta.loc[left_id, "Pair ID"]
ind = np.where(kept_pairs == pair_id)[0][0]
edge_dfs[ind]
# %% [markdown]
# #

iou_df.sort_values("IOU Score", ascending=False, inplace=True)
iou_df.index = range(len(iou_df))

colors = []
ids = []
for i in range(100):
    left_id = iou_df.loc[i, "Left ID"]
    right_id = iou_df.loc[i, "Right ID"]
    iou = iou_df.loc[i, "IOU Score"]
    c = int(iou * 255.0)
    print(c)
    print(iou)
    hex_color = "#%02X%02X%02X" % (c, c, c)
    colors.append(hex_color)
    colors.append(hex_color)
    ids.append(left_id)
    ids.append(right_id)

stashskel("high-iou-sorted-pairs", ids, colors, colors=colors, palette=None)

# %% [markdown]
# #

# Get rid of unpaired cells for now

mg.meta["is_paired"] = False
pairs = mg["Pair"]
is_paired = pairs != -1
mg.reindex(is_paired)
mg.sort_values(["Hemisphere", "Pair ID"])

plot_adj = False
if plot_adj:
    heatmap(
        mg.adj,
        sort_nodes=False,
        inner_hier_labels=mg["Class 1"],
        outer_hier_labels=mg["Hemisphere"],
        transform="binarize",
        cbar=False,
        figsize=(30, 30),
        hier_label_fontsize=10,
    )
    heatmap(
        mg.adj, sort_nodes=False, transform="binarize", cbar=False, figsize=(30, 30)
    )

n_pairs = mg.adj.shape[0] // 2

mg.verify()


# %% [markdown]
# # Extract subgraphs


extract_fb = True
if extract_fb:
    from_classes = ["MBON", "FAN", "FBN", "FB2N"]
    to_classes = ["MBIN"]
    sub_mg = MetaGraph(mg.adj, mg.meta)
    from_class_inds = sub_mg.meta["Class 1"].isin(from_classes)
    to_class_inds = sub_mg.meta["Class 1"].isin(to_classes)
    any_inds = np.logical_or(from_class_inds, to_class_inds)
    sub_mg.reindex(any_inds)
    sub_mg.sort_values(["Hemisphere", "Class 1", "Pair ID"])
    # meta = sub_mg.meta
    # meta["Original index"] = range(meta.shape[0])
    # meta.sort_values(
    #     ["Hemisphere", "Class 1", "Pair ID"],
    #     inplace=True,
    #     kind="mergesort",
    #     ascending=False,
    # )
    # temp_inds = meta["Original index"]
    # sub_mg = sub_mg.reindex(temp_inds)
    # sub_mg = MetaGraph(sub_mg.adj, meta)

    adj = sub_mg.adj.copy()
    from_class_inds = np.where(sub_mg.meta["Class 1"].isin(from_classes).values)[0]
    left_inds = np.where(sub_mg.meta["Hemisphere"] == "L")[0]
    right_inds = np.where(sub_mg.meta["Hemisphere"] == "R")[0]
    to_class_inds = np.where(sub_mg.meta["Class 1"].isin(to_classes).values)[0]
    from_left = np.intersect1d(left_inds, from_class_inds)
    from_right = np.intersect1d(right_inds, from_class_inds)
    to_left = np.intersect1d(left_inds, to_class_inds)
    to_right = np.intersect1d(right_inds, to_class_inds)

    left_left_adj = adj[np.ix_(from_left, to_left)]
    left_right_adj = adj[np.ix_(from_left, to_right)]
    right_right_adj = adj[np.ix_(from_right, to_right)]
    right_left_adj = adj[np.ix_(from_right, to_left)]
else:
    adj = mg.adj.copy()
    left_left_adj = adj[:n_pairs, :n_pairs]
    left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
    right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
    right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

if plot_adj:
    heatmap(
        mg.adj,
        inner_hier_labels=mg["Class 1"],
        outer_hier_labels=mg["Hemisphere"],
        transform="binarize",
        figsize=(10, 10),
    )

# %% [markdown]
# # Plot all edges, weights

ll_edges = left_left_adj.ravel()
lr_edges = left_right_adj.ravel()
rr_edges = right_right_adj.ravel()
rl_edges = right_left_adj.ravel()


left_edges = np.concatenate((ll_edges, lr_edges))
right_edges = np.concatenate((rr_edges, rl_edges))


props = []
threshs = np.linspace(0, 0.3, 200)
for threshold in threshs:
    keep_mask = np.logical_or(left_edges >= threshold, right_edges >= threshold)
    thresh_left_edges = left_edges.copy()[keep_mask]
    thresh_right_edges = right_edges.copy()[keep_mask]
    n_consistent = np.logical_and(thresh_left_edges != 0, thresh_right_edges != 0).sum()
    prop_consistent = n_consistent / len(thresh_left_edges)
    props.append(prop_consistent)

if extract_fb:
    first_ind = np.where(np.array(props) > 0.95)[0][0]

plot_df = pd.DataFrame()
plot_df["Threshold"] = threshs
plot_df["P(mirror edge present)"] = props

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# if extract_fb:
#     ax.axvline(threshs[first_ind], c="r")

sns.lineplot(data=plot_df, x="Threshold", y="P(mirror edge present)", ax=ax)
remove_spines(ax)

if extract_fb:
    ax.set_title(f"{graph_type}, 0.95 at thresh = {threshs[first_ind]:0.2f}")
    stashfig(f"thresh-sweep-{graph_type}-fbpaper")
else:
    ax.set_title(f"{graph_type}")
    stashfig(f"thresh-sweep-{graph_type}-brain")

# %% [markdown]
# #
# left_edges = ll_edges + lr_edges
# right_edges = rr_edges + rl_edges

nonzero_inds = np.where(np.logical_or(left_edges > 0, right_edges > 0))[0]
left_edges = left_edges[nonzero_inds]
right_edges = right_edges[nonzero_inds]

sns.set_context("talk", font_scale=1.25)
plot_log = False
if plot_log:
    left_edges = np.log10(left_edges + 0.01)
    right_edges = np.log10(right_edges + 0.01)
    xlabel = "Log10(Left edge weight)"
    ylabel = "Log10(Right edge weight)"
else:
    xlabel = "Left edge weight"
    ylabel = "Right edge weight"

pad = 1
xmin = left_edges.min() - pad
ymin = right_edges.min() - pad
xmax = left_edges.max() + pad
ymax = right_edges.max() + pad

edge_plot_df = pd.DataFrame()
scale = 0.15
edge_plot_df["y"] = left_edges + np.random.normal(scale=scale, size=left_edges.size)
edge_plot_df["x"] = right_edges + np.random.normal(scale=scale, size=right_edges.size)
fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
sns.scatterplot(
    data=edge_plot_df, x="x", y="y", ax=axs[0], s=1, alpha=0.2, linewidth=False
)
max_val = max(xmax, ymax)
axs[0].plot((0, max_val), (0, max_val), color="red", linewidth=1, linestyle="--")
axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].axis("square")

axs[1].hexbin(
    left_edges,
    right_edges,
    bins="log",
    gridsize=(int(left_edges.max()), int(right_edges.max())),
    cmap="Blues",
)
axs[1].axis("square")
axs[1].set_xlabel(xlabel)
for ax in axs:
    remove_spines(ax, keep_corner=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

axs[1].set_ylim((ymin, ymax))
axs[1].set_xlim((xmin, xmax))

plt.tight_layout()

corr = np.corrcoef(left_edges, right_edges)[0, 1]
plt.suptitle(f"{graph_type}, all paired edges, correlation = {corr:0.2f}", y=0.99)
stashfig(f"all-paired-edge-corr-{graph_type}")
axs[1].set_ylim((ymin, 30))
axs[1].set_xlim((xmin, 30))
stashfig(f"all-paired-edge-corr-zoom-{graph_type}")

# %% [markdown]
# # Get pairs into proper format
ll_edges = get_edges(left_left_adj)
lr_edges = get_edges(left_right_adj)
rr_edges = get_edges(right_right_adj)
rl_edges = get_edges(right_left_adj)

left_edge_vecs = []
right_edge_vecs = []
for i in range(n_pairs):
    left_edge_vecs.append(np.concatenate((ll_edges[i], lr_edges[i])))
    right_edge_vecs.append(np.concatenate((rr_edges[i], rl_edges[i])))

# %% [markdown]
# # Now add back in the super nodes - Janky but ok for now

graph_type = "G"
mg_for_unpaired = load_metagraph(graph_type, version=BRAIN_VERSION)
mg_for_unpaired.meta.head()

keep_classes = ["KC", "sens"]


class_label = mg_for_unpaired["Class 1"]
is_keep_class = [c in keep_classes for c in class_label]

is_keep = np.logical_or(is_paired, is_keep_class)

mg_for_unpaired.reindex(is_keep)
# print(np.unique(mg["Pair"]))
meta = mg_for_unpaired.meta
meta["Original index"] = range(meta.shape[0])
meta.sort_values(
    ["Hemisphere", "Pair ID", "Merge Class"],
    inplace=True,
    kind="mergesort",
    ascending=False,
)
temp_inds = meta["Original index"]
mg_for_unpaired = mg_for_unpaired.reindex(temp_inds)
mg_for_unpaired = MetaGraph(mg_for_unpaired.adj, meta)

adj = mg_for_unpaired.adj
for c in keep_classes:
    for side in ["L", "R"]:  # this is the side of the paired neurons
        for direction in ["ipsi", "contra"]:
            if direction == "ipsi":
                class_inds = np.where(
                    np.logical_and(
                        mg_for_unpaired["Hemisphere"] == side,
                        mg_for_unpaired["Class 1"] == c,
                    )
                )[0]
                unpaired_inds = np.where(
                    np.logical_and(
                        mg_for_unpaired["Hemisphere"] == side,
                        mg_for_unpaired["Pair ID"] == -1,
                    )
                )[0]
            elif direction == "contra":
                class_inds = np.where(
                    np.logical_and(
                        mg_for_unpaired["Hemisphere"] != side,
                        mg_for_unpaired["Class 1"] == c,
                    )
                )[0]
                unpaired_inds = np.where(
                    np.logical_and(
                        mg_for_unpaired["Hemisphere"] != side,
                        mg_for_unpaired["Pair ID"] == -1,
                    )
                )[0]
            unpaired_inds = np.intersect1d(class_inds, unpaired_inds)
            paired_inds = np.where(
                np.logical_and(
                    mg_for_unpaired["Hemisphere"] == side,
                    mg_for_unpaired["Pair ID"] != -1,
                )
            )[0]

            # from
            from_edges = adj[np.ix_(unpaired_inds, paired_inds)]
            pair_super_edges = from_edges.sum(axis=0)

            # to
            to_edges = adj[np.ix_(paired_inds, unpaired_inds)]
            pair_super_edges = to_edges.sum(axis=1)


# %% [markdown]
# # Plot, by pair, the IOU score

threshold = 8

iou_scores = []
rand_iou_scores = []
for i in range(n_pairs):
    left_edge_vec = left_edge_vecs[i].copy()
    left_edge_vec[left_edge_vec <= threshold] = 0
    right_edge_vec = right_edge_vecs[i].copy()
    right_edge_vec[right_edge_vec <= threshold] = 0
    ind = np.random.choice(len(right_edge_vecs))
    rand_vec = right_edge_vecs[ind].copy()
    rand_vec[rand_vec <= threshold] = 0

    left_mask = left_edge_vec > 0
    right_mask = right_edge_vec > 0
    or_mask = np.logical_or(left_mask, right_mask)  # anything that is an edge in either
    and_mask = np.logical_and(left_mask, right_mask)  # edge in both
    iou_score = and_mask.sum() / or_mask.sum()
    if not np.isnan(iou_score):
        iou_scores.append(iou_score)
    else:
        left_id = mg.meta.index[i]
        right_id = mg.meta.index[i + n_pairs]
        print(f"Nan IOU score for pair {left_id}, {right_id}")
    rand_mask = rand_vec > 0
    or_mask = np.logical_or(left_mask, rand_mask)  # anything that is an edge in either
    and_mask = np.logical_and(left_mask, rand_mask)  # edge in both
    iou_score = and_mask.sum() / or_mask.sum()
    if not np.isnan(iou_score):
        rand_iou_scores.append(iou_score)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot(iou_scores, ax=ax, norm_hist=True, label="True pair")
sns.distplot(rand_iou_scores, ax=ax, norm_hist=True, label="Random pair")
ax.set_xlabel("IOU score (binary edges)")
ax.set_title(f"Pair edge agreement, {graph_type}, theshold = {threshold}")
ax.yaxis.set_major_locator(plt.NullLocator())
remove_spines(ax)
ax.set_xlim((0, 1))
stashfig(f"pair-IOU-random-{graph_type}-t{threshold}")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot(iou_scores, ax=ax, norm_hist=True)
ax.set_xlabel("IOU score (binary edges)")
ax.set_title(f"Pair edge agreement, {graph_type}, theshold = {threshold}")
ax.yaxis.set_major_locator(plt.NullLocator())
remove_spines(ax)
ax.set_xlim((0, 1))
stashfig(f"pair-IOU-{graph_type}-t{threshold}")

# %% [markdown]
# # Plot, for a subset of random pairs, the edge weights
n_plots = 5
for i in range(n_plots):
    fig, axs = plt.subplots(4, 6, figsize=(30, 20))
    axs = axs.ravel()
    rand = np.random.randint(1e8)
    np.random.seed(rand)
    for ax in axs:
        ind = np.random.choice(len(right_edge_vecs))
        left_edge_vec = left_edge_vecs[ind].copy()
        right_edge_vec = right_edge_vecs[ind].copy()
        left_mask = left_edge_vec > 0
        right_mask = right_edge_vec > 0
        or_mask = np.logical_or(left_mask, right_mask)
        remove_spines(ax)
        if or_mask.sum() > 0:
            left_id = mg.meta.index[ind]
            right_id = mg.meta.index[ind + n_pairs]
            left_edge_vec = left_edge_vec[or_mask]
            right_edge_vec = right_edge_vec[or_mask]
            corr = np.corrcoef(left_edge_vec, right_edge_vec)[0, 1]
            sns.scatterplot(left_edge_vec, right_edge_vec, ax=ax)
            ax.axis("equal")
            ax.set_title(f"Corr = {corr:0.2f}")
            max_val = max(left_edge_vec.max(), right_edge_vec.max())
            ax.plot(
                (0, max_val), (0, max_val), color="red", linewidth=1, linestyle="--"
            )
            ax.text(max_val, max_val, max_val)
            # ax.xaxis.set_major_locator(plt.MaxNLocator(1))
            # xticks = ax.get_xticks()
            # print(xticks)
            # ax.set_yticks(xticks)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.set_xlabel(left_id)
            ax.set_ylabel(right_id)

    plt.suptitle(f"{graph_type}, edge weights by pair", y=1.02)
    plt.tight_layout()
    stashfig(f"edge-weights-by-pair-{graph_type}-r{rand}")
# %% [markdown]
# # For some random cells plot the pair correlations

# %% [markdown]
# # Look into a few of the low x or y edges

# %% [markdown]
# #
# Extract edges
ll_edges = get_edges(left_left_adj)
lr_edges = get_edges(left_right_adj)
rr_edges = get_edges(right_right_adj)
rl_edges = get_edges(right_left_adj)

pair_corrs = []
for i in range(n_pairs):
    left_edge_vec = np.concatenate((ll_edges[i], lr_edges[i]))
    right_edge_vec = np.concatenate((rr_edges[i], rl_edges[i]))
    both_edges = np.stack((left_edge_vec, right_edge_vec), axis=-1)
    avg_edges = np.mean(both_edges, axis=-1)
    # inds = np.where(avg_edges > threshold)[0]

    # check that left and right edge both have
    inds = np.where(
        np.logical_and(left_edge_vec > threshold, right_edge_vec > threshold)
    )[0]
    if len(inds) > 0:
        left_edge_vec = left_edge_vec[inds]
        # left_edge_vec[left_edge_vec > 0] = 1
        right_edge_vec = right_edge_vec[inds]
        # print(left_edge_vec)
        # print(right_edge_vec)
        # right_edge_vec[right_edge_vec > 0] = 1
        R = np.corrcoef(left_edge_vec, right_edge_vec)
        corr = R[0, 1]
        # print(corr)
        # print()
        if i < 40:
            plt.figure()
            plt.scatter(left_edge_vec, right_edge_vec)
            plt.title(corr)
            plt.axis("square")
        # corr = np.count_nonzero(left_edge_vec - right_edge_vec) / len(left_edge_vec)
    else:
        corr = 0
    if np.isnan(corr):
        corr = 0
    pair_corrs.append(corr)
pair_corrs = np.array(pair_corrs)


ground_truth_file = (
    base_path / "neuron-groups/GroundTruth_NeuronPairs_Brain-2019-07-29.csv"
)

ground_truth_df = pd.read_csv(ground_truth_file)
ground_truth_df.set_index("ID", inplace=True)
ground_truth_df.head()
ground_truth_ids = ground_truth_df.index.values
ground_truth_sides = ground_truth_df["Hemisphere"].values

known_inds = []
for cell_id, side in zip(ground_truth_ids, ground_truth_sides):
    if side == " left":
        ind = np.where(mg.meta.index.values == cell_id)[0]
        if len(ind) > 0:
            known_inds.append(ind[0])

not_known_inds = np.setdiff1d(range(n_pairs), known_inds)
new_pair_corrs = pair_corrs[not_known_inds]
truth_pair_corrs = pair_corrs[known_inds]

sns.set_context("talk")
plt.figure(figsize=(10, 5))
sns.distplot(new_pair_corrs, label="New pairs")
sns.distplot(truth_pair_corrs, label="Ground truth")
plt.legend()
plt.title(threshold)
stashfig(f"both-t{threshold}-corr-nodewise")

out_pair_df = pd.DataFrame()
out_pair_df

# %% Look at correlation vs degree
deg_df = mg.calculate_degrees()
plot_df = pd.DataFrame()
total_degree = deg_df["Total degree"].values
plot_df["Mean total degree"] = (
    total_degree[:n_pairs] + total_degree[n_pairs : 2 * n_pairs]
) / 2
plot_df["Correlation"] = pair_corrs
plt.figure(figsize=(10, 5))
sns.scatterplot(data=plot_df, x="Mean total degree", y="Correlation")
stashfig("corr-vs-degree")

sns.jointplot(
    data=plot_df, x="Mean total degree", y="Correlation", kind="hex", height=10
)
stashfig("corr-vs-degree-hex")

# %% [markdown]
# # Find the cells where correlation is < 0
skeleton_labels = mg.meta.index.values[: 2 * n_pairs]
side_labels = mg["Hemisphere"][: 2 * n_pairs]
left_right_pairs = zip(
    skeleton_labels[:n_pairs], skeleton_labels[n_pairs : 2 * n_pairs]
)

colors = []
ids = []

for i, (left, right) in enumerate(left_right_pairs):
    if pair_corrs[i] < 0:
        hex_color = "#%02X%02X%02X" % (r(), r(), r())
        colors.append(hex_color)
        colors.append(hex_color)
        ids.append(left)
        ids.append(right)

stashskel("pairs-low-corr", ids, colors, colors=colors, palette=None)

# %% [markdown]
# # Look at number of disagreements vs degree
prop_disagreements = []
for i in range(n_pairs):
    left_edge_vec = np.concatenate((ll_edges[i], lr_edges[i]))
    right_edge_vec = np.concatenate((rr_edges[i], rl_edges[i]))
    left_edge_vec[left_edge_vec > 0] = 1
    right_edge_vec[right_edge_vec > 0] = 1
    n_disagreement = np.count_nonzero(left_edge_vec - right_edge_vec)
    prop_disagreement = n_disagreement / len(left_edge_vec)
    prop_disagreements.append(prop_disagreement)
prop_disagreements = np.array(prop_disagreements)
plot_df["Prop. disagreements"] = prop_disagreements

sns.jointplot(
    data=plot_df, x="Mean total degree", y="Prop. disagreements", kind="hex", height=10
)
#%%
pair_corrs = []
for i in range(n_pairs):
    left_edge_vec = np.concatenate((ll_edges[i], lr_edges[i]))
    right_edge_vec = np.concatenate((rr_edges[i], rl_edges[i]))
    both_edges = np.stack((left_edge_vec, right_edge_vec), axis=-1)
    avg_edges = np.mean(both_edges, axis=-1)

#%%
left_edges = np.concatenate((left_left_adj.ravel(), left_right_adj.ravel()))
right_edges = np.concatenate((right_right_adj.ravel(), right_left_adj.ravel()))
all_edges = np.stack((left_edges, right_edges), axis=1)
all_edges_sum = np.sum(all_edges, axis=1)
edge_mask = all_edges_sum > 0
all_edges = all_edges[edge_mask]
left_edges = left_edges[edge_mask]
right_edges = right_edges[edge_mask]
mean_edges = np.mean(all_edges, axis=-1)
diff_edges = np.abs(left_edges - right_edges)
plot_df = pd.DataFrame()
plot_df["Mean (L/R) edge"] = mean_edges
plot_df["Diff (L/R) edge"] = diff_edges
plt.figure(figsize=(10, 10))
sns.scatterplot(mean_edges, diff_edges)
sns.jointplot(data=plot_df, x="Mean (L/R) edge", y="Diff (L/R) edge", kind="hex")
plt.figure(figsize=(10, 5))
bins = np.linspace(-1, 40, 41)
sns.distplot(diff_edges, kde=False, norm_hist=False, bins=bins)
sns.jointplot(
    data=plot_df,
    x="Mean (L/R) edge",
    y="Diff (L/R) edge",
    kind="hex",
    xlim=(0, 5),
    ylim=(0, 5),
)

# %% [markdown]
# # this works, would be to plot a heatmap

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
edge_plot_df = pd.DataFrame()
edge_plot_df["y"] = left_edges
edge_plot_df["x"] = right_edges
count_df = edge_plot_df.groupby(["x", "y"]).size().unstack(fill_value=0)
count_df = count_df.combine_first(count_df.T).fillna(0.0)
sns.heatmap(
    count_df + 1,
    cmap="Blues",
    cbar=True,
    ax=ax,
    square=True,
    norm=LogNorm(vmin=count_df.values.min(), vmax=count_df.values.max()),
)
ylims = ax.get_ylim()
ax.set_ylim((ylims[1], ylims[0]))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
