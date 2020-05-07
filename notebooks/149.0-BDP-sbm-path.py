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
# ##

n_blocks = 15
B = np.zeros((n_blocks, n_blocks))
B[0, 6] = 0.3
B[1, 4] = 0.3
B[2, 7] = 0.3
B[3, 5] = 0.3
B[5, 8] = 0.3
B[4, 7] = 0.3
B[4, 6] = 0.3
B[6, 9] = 0.3
B[6, 10] = 0.3
B[8, 10] = 0.3
B[9, 13] = 0.3
B[9, 11] = 0.3
B[11, 12] = 0.3
B[10, 14] = 0.3
B[10, 13] = 0.3


fig, axs = plt.subplots(1, 2, figsize=(20, 10))
# sns.set_context("talk", font_scale=0.7)
sns.heatmap(B, annot=False, cmap="Reds", cbar=False, ax=axs[0], square=True)
axs[0].xaxis.tick_top()
axs[0].set_title("Block probability matrix", pad=25)

n_per_block = 50
community_sizes = n_blocks * [n_per_block]

np.random.seed(88)
adj, labels = sbm(community_sizes, B, directed=True, loops=False, return_labels=True)
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
stashfig("sbm")
# %% [markdown]
# ##


def add_noise(B):
    B += np.diag(np.full(n_blocks, 0.1))
    B += 0.005
    return B


n_blocks = 14

P0 = np.zeros((n_blocks, n_blocks))
P0[0, 1] = 0.5
P0[1, 2] = 0.5
P0[2, 3] = 0.5

P1 = np.zeros((n_blocks, n_blocks))
P1[4, 5] = 0.5
P1[5, 1] = 0.5
P1[1, 6] = 0.5
P1[6, 7] = 0.5

P2 = np.zeros((n_blocks, n_blocks))
P2[8, 9] = 0.5
P2[9, 10] = 0.5
P2[10, 6] = 0.5
P2[6, 7] = 0.5

P3 = np.zeros((n_blocks, n_blocks))
P3[11, 12] = 0.5
P3[12, 6] = 0.5
P3[6, 13] = 0.5

B = 0.25 * P0 + 0.25 * P1 + 0.25 * P2 + 0.25 * P3
B = add_noise(B)
sf = -signal_flow(B)
perm = np.argsort(sf)
B = B[np.ix_(perm, perm)]
P0 = P0[np.ix_(perm, perm)]
P1 = P1[np.ix_(perm, perm)]
P2 = P2[np.ix_(perm, perm)]
P3 = P3[np.ix_(perm, perm)]

n_row = 2
n_col = 6
scale = 5
fig = plt.figure(figsize=(n_col * scale, n_row * scale))
from matplotlib.gridspec import GridSpec

gs = GridSpec(n_row, n_col, figure=fig)


def quick_heatmap(A, ax):
    sns.heatmap(A, annot=False, cmap="Reds", cbar=False, ax=ax, square=True)


ax = fig.add_subplot(gs[0, 0])
quick_heatmap(P1, ax)
ax.set_xticks([])
name0 = r"$P_0: 0 \to 4 \to 7 \to 10 \to 13$"
ax.set_title(name0)

ax = fig.add_subplot(gs[0, 1])
quick_heatmap(P2, ax)
ax.set_yticks([])
ax.set_xticks([])
name1 = r"$P_1: 1 \to 5 \to 8 \to 10 \to 13$"
ax.set_title(name1)

ax = fig.add_subplot(gs[1, 0])
quick_heatmap(P3, ax)
name2 = r"$P_2: 2 \to 6 \to 10 \to 12$"
ax.set_title(name2)

ax = fig.add_subplot(gs[1, 1])
quick_heatmap(P0, ax)
ax.set_yticks([])
name3 = r"$P_3: 3 \to 7 \to 9 \to 11$"
ax.set_title(name3)

ax = fig.add_subplot(gs[:, 2:4])
quick_heatmap(B, ax)
ax.set_title(r"$B: \frac{1}{4}(P_0 + P_1 + P_2 + P_3) + assortative + noise$")

np.random.seed(88)
n_per_block = 50
community_sizes = n_blocks * [n_per_block]
adj, labels = sbm(community_sizes, B, directed=True, loops=False, return_labels=True)
n_verts = adj.shape[0]
ax = fig.add_subplot(gs[:, 4:])
matrixplot(
    adj, row_sort_class=labels, col_sort_class=labels, cbar=False, ax=ax, square=True
)
ax.xaxis.tick_bottom()
# ax.xaxis.set_label_position("bottom")
ax.set_xticklabels(ax.get_xticklabels(), va="top")
ax.set_title(r"$A:$ (50 vertices per block)")
plt.tight_layout()
stashfig("sbm-problem-statement")

# %% [markdown]
# ##

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
sns.heatmap(B, annot=False, cmap="Reds", cbar=False, ax=axs[0], square=True)
axs[0].xaxis.tick_top()
axs[0].set_title("Block probability matrix", pad=25)


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

# %% [markdown]
# ##
D_inv = np.diag(1 / np.sum(adj, axis=1))
M = D_inv @ adj

# %% [markdown]
# ##

n_init = 2 ** 10
all_inds = np.arange(len(adj))
out_blocks = [11, 12, 13]
start_blocks = [0, 1, 2, 3]
out_inds = all_inds[np.isin(labels, out_blocks)]
start_inds = all_inds[np.isin(labels, start_blocks)]


def rw_from_node(s):
    paths = []
    rw = RandomWalk(M, stop_nodes=out_inds, max_hops=10, allow_loops=False)
    for n in range(n_init):
        rw.start(s)
        paths.append(rw.traversal_)
    return paths


paths = []
for s in start_inds:
    paths += rw_from_node(s)

start = []
path_indicator_mat = np.zeros((len(paths), len(adj)), dtype=int)
for i, p in enumerate(paths):
    start.append(labels[p[0]])
    for j, visit in enumerate(p):
        path_indicator_mat[i, visit] = j + 1

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# path_indicator_mat[path_indicator_mat != 1] = 0
matrixplot(
    path_indicator_mat,
    plot_type="scattermap",
    row_sort_class=start,
    col_sort_class=labels,
    # row_item_order=np.arange(len(path_indicator_mat)),
    # col_item_order=np.arange(path_indicator_mat.shape[1]),
    sizes=(0.2, 0.2),
    hue="weight",
    palette=sns.color_palette("tab10", n_colors=10),
    ax=ax,
)
ax.set_yticklabels([name0, name1, name2, name3])
# %% [markdown]
# ##
matrixplot(
    path_indicator_mat[:50, :50],
    plot_type="scattermap",
    sizes=(0.2, 0.2),
    hue="weight",
    palette=sns.color_palette("husl", n_colors=10),
    ax=ax,
)


# %% [markdown]
# ##

embedder = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
embed = embedder.fit_transform(adj)
embed = np.concatenate(embed, axis=-1)
pairplot(embed, labels=labels, palette="tab20")


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
