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
BRAIN_VERSION = "2020-01-14"

sns.set_context("talk")


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
# # Load data

graph_type = "G"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")
skeleton_labels = mg.meta.index.values
mg.meta.head()
print(np.unique(mg["Pair"]))
ind = np.where(mg.index == )

# %% [markdown]
# # Get rid of unpaired cells for now

mg.meta["is_paired"] = False
pairs = mg["Pair"]
is_paired = pairs != -1
mg.reindex(is_paired)
print(np.unique(mg["Pair"]))
meta = mg.meta
meta["Original index"] = range(meta.shape[0])
meta.sort_values(
    ["Hemisphere", "Pair ID"], inplace=True, kind="mergesort", ascending=False
)
temp_inds = meta["Original index"]
mg = mg.reindex(temp_inds)
mg = MetaGraph(mg.adj, meta)

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
    stashfig("heatmap")

    heatmap(
        mg.adj,
        sort_nodes=False,
        inner_hier_labels=mg["Hemisphere"],
        transform="binarize",
        cbar=False,
        figsize=(30, 30),
        hier_label_fontsize=10,
    )

n_pairs = mg.adj.shape[0] // 2

mg.verify()

# %% [markdown]
# # Extract subgraphs

adj = mg.adj.copy()

threshold = 0

left_left_adj = adj[:n_pairs, :n_pairs]
left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

# %% [markdown]
# # Plot all edges, weights

ll_edges = left_left_adj.ravel()
lr_edges = left_right_adj.ravel()
rr_edges = right_right_adj.ravel()
rl_edges = right_left_adj.ravel()

left_edges = np.concatenate((ll_edges, lr_edges))
right_edges = np.concatenate((rr_edges, rl_edges))
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

### LESS GOOD BELOW HEREs
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
