# %% [markdown]
# # THE MIND OF A MAGGOT

# %% [markdown]
# ## Imports
import os
import time
import warnings
from itertools import chain

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, NodeMixin
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    select_dimension,
    selectSVD,
)
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import rdpg
from graspy.utils import augment_diagonal, binarize, pass_to_ranks
from src.cluster import get_paired_inds
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.traverse import (
    Cascade,
    RandomWalk,
    TraverseDispatcher,
    to_markov_matrix,
    to_path_graph,
    to_transmission_matrix,
)
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    screeplot,
    set_axes_equal,
    stacked_barplot,
)

# from tqdm import tqdm


warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1.25, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name)


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    """
    p = np.asarray(p)
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


# %% [markdown]
# ##

from graspy.simulations import sbm


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


low_p = 0.01
diag_p = 0.1
feedforward_p = 0.3
n_blocks = 6

max_hops = 15
n_init = 100

basename = f"-{feedforward_p}-{diag_p}-{low_p}-{n_blocks}-{max_hops}-{n_init}"


block_probs = get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=n_blocks)
block_probs[1, 4] = 0.3
block_probs[2, 3] = 0.01
block_probs[2, 5] = 0.3
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False, ax=axs[0], square=True)
axs[0].xaxis.tick_top()
axs[0].set_title("Block probability matrix", pad=25)

community_sizes = np.empty(2 * n_blocks, dtype=int)
n_per_block = 100
community_sizes = n_blocks * [n_per_block]

np.random.seed(88)
adj, labels = sbm(
    community_sizes, block_probs, directed=True, loops=False, return_labels=True
)
n_verts = adj.shape[0]

matrixplot(
    adj,
    row_sort_class=labels,
    col_sort_class=labels,
    cbar=False,
    ax=axs[1],
    square=True,
)
axs[1].set_title("Adjacency matrix", pad=25)
plt.tight_layout()
stashfig("sbm" + basename)


# %% [markdown]
# ## Run paths
print(f"Running {n_init} random walks from each source node...")

transition_probs = to_markov_matrix(adj)

out_inds = np.where(labels == n_blocks - 1)[0]
source_inds = np.where(labels == 0)[0]


def rw_from_node(s):
    paths = []
    rw = RandomWalk(
        transition_probs, stop_nodes=out_inds, max_hops=10, allow_loops=False
    )
    for n in range(n_init):
        rw.start(s)
        paths.append(rw.traversal_)
    return paths


par = Parallel(n_jobs=-1, verbose=10)
paths_by_node = par(delayed(rw_from_node)(s) for s in source_inds)
paths = []
for p in paths_by_node:
    paths += p
print(len(paths))

# %% [markdown]
# ## Look at distribution of path lengths
path_lens = []
for p in paths:
    path_lens.append(len(p))

sns.distplot(path_lens, kde=False)

paths_by_len = {i: [] for i in range(1, max_hops + 1)}
for p in paths:
    paths_by_len[len(p)].append(p)

# %% [markdown]
# ## Subsampling and selecting paths
path_len = 4
paths = paths_by_len[path_len]
np.random.seed(8888)

pal = sns.color_palette("tab10", n_colors=10)
pal = [pal[0], pal[2], pal[4], pal[6], pal[1], pal[3], pal[5], pal[7], (0, 0, 0)]
pal = pal[:path_len]

new_paths = []
for p in paths:
    # select paths that got to a stop node
    if p[-1] in out_inds:
        new_paths.append(p)
paths = new_paths
print(f"Number of paths of length {path_len}: {len(paths)}")

subsample = min(2 ** 13, len(paths))


if subsample != -1:
    inds = np.random.choice(len(paths), size=subsample, replace=False)
    new_paths = []
    for i, p in enumerate(paths):
        if i in inds:
            new_paths.append(p)
    paths = new_paths

print(f"Number of paths after subsampling: {len(paths)}")


# %% [markdown]
# ##

embedder = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
embed = embedder.fit_transform(pass_to_ranks(adj))
embed = np.concatenate(embed, axis=-1)
pairplot(embed, labels=labels)

# %% [markdown]
# ## Show 2 dimensions of pairwise cosine embedding

pdist = pairwise_distances(embed, metric="cosine")

# %% [markdown]
# ##
manifold = TSNE(metric="precomputed")
# manifold = ClassicalMDS(n_components=2, dissimilarity="precomputed")
cos_embed = manifold.fit_transform(pdist)

# %% [markdown]
# ##
plot_df = pd.DataFrame(data=cos_embed)
plot_df["labels"] = labels

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="labels",
    palette="Set1",
    # legend="full",
    ax=ax,
    s=25,
    linewidth=0.5,
    alpha=0.8,
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.get_legend().get_texts()[0].set_text("Block")
ax.axis("off")

for b in np.unique(labels):
    mean_series = plot_df[plot_df["labels"] == b].mean()
    x = mean_series[0] + 4
    y = mean_series[1] + 4
    ax.text(x, y, b, fontsize=20)

ax = axs[1]
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    # hue="labels",
    color="grey",
    palette="Set1",
    ax=ax,
    s=25,
    linewidth=0.5,
    alpha=0.8,
)

# ax.get_legend().remove()
# ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
# ax.get_legend().get_texts()[0].set_text("Block")
# ax.axis("equal")
ax.axis("off")


for b in np.unique(labels):
    mean_series = plot_df[plot_df["labels"] == b].mean()
    x = mean_series[0] + 4
    y = mean_series[1] + 4
    ax.text(x, y, b, fontsize=20)

pal = sns.color_palette("husl", n_colors=path_len)
# pal = [pal[0], pal[2], pal[4], pal[6], pal[1], pal[3], pal[5], pal[7], (0, 0, 0)]
# pal = pal[:path_len]

plot_path_inds = np.random.choice(len(paths), size=500, replace=False)
for i, p in enumerate(paths):
    if i in plot_path_inds:
        for t, (start, end) in enumerate(nx.utils.pairwise(p)):
            x1, y1 = plot_df.loc[start, [0, 1]]
            x2, y2 = plot_df.loc[end, [0, 1]]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=pal[t],
                linewidth=0.2,
                alpha=0.6,
                label=t + 1 if i == plot_path_inds[0] else "",
            )

leg = ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Link order")
for lh in leg.legendHandles:
    lh.set_alpha(1)
    lh.set_linewidth(3)

stashfig("embed-sbm-nodes")

# %% [markdown]
# ## Compute distances between paths
print("Computing pairwise distances between paths...")
print(len(paths))
path_dist_func = "squared"

path_dist_mat = np.zeros((len(paths), len(paths)))
for i in range(len(paths)):
    for j in range(len(paths)):
        p1 = paths[i]
        p2 = paths[j]
        dist_sum = 0
        for t in range(path_len):
            dist = pdist[p1[t], p2[t]]
            if path_dist_func == "squared":
                dist = dist ** 2
            dist_sum += dist
        path_dist_mat[i, j] = dist_sum

if path_dist_func == "squared":
    path_dist_mat = path_dist_mat ** (1 / 2)

path_indicator_mat = np.zeros((len(paths), len(adj)), dtype=int)
for i, p in enumerate(paths):
    for j, visit in enumerate(p):
        path_indicator_mat[i, visit] = j + 1

# %% [markdown]
# ##

Z = linkage(squareform(path_dist_mat), method="average")
sns.clustermap(
    path_dist_mat,
    figsize=(20, 20),
    row_linkage=Z,
    col_linkage=Z,
    xticklabels=False,
    yticklabels=False,
)
stashfig("agglomerative-path-dist-mat")


# %% [markdown]
# ##

# %% [markdown]
# ##

# manifold = ClassicalMDS(n_components=2, dissimilarity="precomputed")
manifold = TSNE(metric="precomputed")

path_embed = manifold.fit_transform(path_dist_mat)
path_embed = path_embed + np.random.normal(0, 0.01, size=path_embed.shape)

agmm = AutoGMMCluster(max_components=5, n_jobs=-2)
pred = agmm.fit_predict(path_embed)

plot_df = pd.DataFrame(data=path_embed)
plot_df["labels"] = pred
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="labels",
    palette="Set1",
    # legend="full",
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
# ax.get_legend().remove()
# ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
# ax.axis("equal")
# ax.axis("off")

# %% [markdown]
# ##


# %% [markdown]
# ##


node_meta = pd.DataFrame()
# plot_meta = meta.loc[visited]
node_meta["labels"] = labels

# visited = path_indicator_mat.sum(axis=0)
# visited = visited > 0
plot_path_indicator_mat = path_indicator_mat
# node_meta = node_meta.iloc[visited]

path_meta = pd.DataFrame()
path_meta["cluster"] = pred

Z = linkage(squareform(path_dist_mat), method="average", optimal_ordering=False)
R = dendrogram(
    Z,
    truncate_mode=None,
    get_leaves=True,
    no_plot=True,
    color_threshold=0,
    distance_sort="descending",
    above_threshold_color="k",
    orientation="left",
)
order = invert_permutation(R["leaves"])

path_meta["dend_order"] = -order

context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)
fig, axs = plt.subplots(
    1, 2, figsize=(15, 10), gridspec_kw=dict(width_ratios=[0.95, 0.02], wspace=0.02)
)
ax = axs[0]
ax, div, top, left = matrixplot(
    plot_path_indicator_mat,
    ax=ax,
    plot_type="scattermap",
    col_sort_class=["labels"],
    col_ticks=True,
    col_meta=node_meta,
    col_colors="labels",
    col_palette="Set1",
    row_meta=path_meta,
    row_sort_class="cluster",
    row_item_order="dend_order",
    row_ticks=True,
    gridline_kws=dict(linewidth=1, color="grey", linestyle="--"),
    sizes=(4, 4),
    hue="weight",
    palette=pal,
)
ax.set_ylabel("Paths (grouped by cluster)")
top.set_xlabel("Nodes (grouped by block)")
ax = axs[1]
palplot(pal, ax=ax)
ax.yaxis.tick_right()
ax.set_title("Hop")
ax.set_yticklabels(np.arange(1, path_len + 1))

from src.visualization import remove_shared_ax, remove_spines

# dend_ax = div.append_axes("left", size="8%", pad=0, sharey=ax)
# remove_shared_ax(dend_ax)
# remove_spines(dend_ax)


# from matplotlib.collections import LineCollection

# dcoord = R["icoord"]
# icoord = R["dcoord"]
# coords = zip(icoord, dcoord)
# tree_kws = {"linewidths": 0.3, "colors": ".2"}
# lines = LineCollection([list(zip(x, y)) for x, y in coords], **tree_kws)
# dend_ax.add_collection(lines)
# number_of_leaves = len(order)
# max_coord = max(map(max, icoord))
# dend_ax.set_ylim(0, number_of_leaves * 10)
# dend_ax.set_xlim(max_coord * 1.05, 0)
stashfig("path-indcator-cluster-path" + basename)


# %%
