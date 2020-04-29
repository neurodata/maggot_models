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
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
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


graph_type = "Gad"
mg = load_metagraph(graph_type, version="2020-04-01")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

# plot where we are cutting out nodes based on degree
degrees = mg.calculate_degrees()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sns.distplot(np.log10(degrees["Total edgesum"]), ax=ax)
q = np.quantile(degrees["Total edgesum"], 0.05)
ax.axvline(np.log10(q), linestyle="--", color="r")
ax.set_xlabel("log10(total synapses)")

# remove low degree neurons
idx = meta[degrees["Total edgesum"] > q].index
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)
meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj

# %% [markdown]
# ## Setup for paths

out_groups = [
    ("dVNC", "dVNC;CN", "dVNC;RG", "dSEZ;dVNC"),
    ("dSEZ", "dSEZ;CN", "dSEZ;LHN", "dSEZ;dVNC"),
    ("motor-PaN", "motor-MN", "motor-VAN", "motor-AN"),
    ("RG", "RG-IPC", "RG-ITP", "RG-CA-LP", "dVNC;RG"),
    ("dUnk",),
]
out_group_names = ["VNC", "SEZ" "motor", "RG", "dUnk"]
source_groups = [
    ("sens-ORN",),
    ("sens-MN",),
    ("sens-photoRh5", "sens-photoRh6"),
    ("sens-thermo",),
    ("sens-vtd",),
    ("sens-AN",),
]
source_group_names = ["Odor", "MN", "Photo", "Temp", "VTD", "AN"]
class_key = "merge_class"

sg = list(chain.from_iterable(source_groups))
og = list(chain.from_iterable(out_groups))
sg_name = "All"
og_name = "All"

np.random.seed(888)
max_hops = 10
n_init = 2 ** 11
transition_probs = to_markov_matrix(adj)

source_inds = meta[meta[class_key].isin(sg)]["inds"].values
out_inds = meta[meta[class_key].isin(og)]["inds"].values


# %% [markdown]
# ## Run paths
print(f"Running {n_init} random walks from each source node...")


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
stashfig(f"path-length-dist-graph={graph_type}")

paths_by_len = {i: [] for i in range(1, max_hops + 1)}
for p in paths:
    paths_by_len[len(p)].append(p)

# %% [markdown]
# ## Subsampling and selecting paths
path_len = 6
paths = paths_by_len[path_len]
subsample = min(2 ** 13, len(paths))
np.random.seed(8888)


new_paths = []
for p in paths:
    # select paths that got to a stop node
    if p[-1] in out_inds:
        new_paths.append(p)
paths = new_paths
print(f"Number of paths of length {path_len}: {len(paths)}")

if subsample != -1:
    inds = np.random.choice(len(paths), size=subsample, replace=False)
    new_paths = []
    for i, p in enumerate(paths):
        if i in inds:
            new_paths.append(p)
    paths = new_paths

print(f"Number of paths after subsampling: {len(paths)}")


# %% [markdown]
# ## Embed for a dissimilarity measure


def add_connections(x1, x2, y1, y2, color="black", alpha=0.3, linewidth=0.3, ax=None):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    if ax is None:
        ax = plt.gca()
    for i in range(len(x1)):
        ax.plot(
            [x1[i], x2[i]],
            [y1[i], y2[i]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


print("Embedding graph...")
embedder = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
embed = embedder.fit_transform(pass_to_ranks(adj))
embed = np.concatenate(embed, axis=-1)

lp_inds, rp_inds = get_paired_inds(meta)
R, _, = orthogonal_procrustes(embed[lp_inds], embed[rp_inds])

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
plot_df = pd.DataFrame(data=embed[:, [0, 1]])
plot_df["merge_class"] = meta["merge_class"].values
ax = axs[0]
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
# ax.axis("off")
remove_spines(ax)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.spines["right"].set_visible(True)
ax.set_title("Before Procrustes")
add_connections(
    plot_df.iloc[lp_inds, 0],
    plot_df.iloc[rp_inds, 0],
    plot_df.iloc[lp_inds, 1],
    plot_df.iloc[rp_inds, 1],
    ax=ax,
)

left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]
embed[left_inds] = embed[left_inds] @ R
plot_df[0] = embed[:, 0]
plot_df[1] = embed[:, 1]
ax = axs[1]
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
ax.axis("off")
add_connections(
    plot_df.iloc[lp_inds, 0],
    plot_df.iloc[rp_inds, 0],
    plot_df.iloc[lp_inds, 1],
    plot_df.iloc[rp_inds, 1],
    ax=ax,
)
ax.set_title("After Procrustes")

plt.tight_layout()
stashfig("procrustes-ase")


# %% [markdown]
# ## Use cosine distance as dissimilarity
print("Finding pairwise distances...")
pdist = pairwise_distances(embed, metric="cosine")

manifold = TSNE(metric="precomputed")
cos_embed = manifold.fit_transform(pdist)
plot_df = pd.DataFrame(data=cos_embed)
plot_df["merge_class"] = meta["merge_class"].values
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
ax.axis("equal")
ax.axis("off")

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
# ## Decide on an embedding method for distance matrix
dim_reduce = "cmds"

basename = (
    f"-dim_red={dim_reduce}-subsample={subsample}-plen={path_len}-graph={graph_type}"
)

# %% [markdown]
# ## Agglomerative cluster and look at path distance mat


Z = linkage(squareform(path_dist_mat), method="average")
sns.clustermap(
    path_dist_mat,
    figsize=(20, 20),
    row_linkage=Z,
    col_linkage=Z,
    xticklabels=False,
    yticklabels=False,
)
stashfig("agglomerative-path-dist-mat" + basename)

# %% [markdown]
# ##
print("Running dimensionality reduction on path dissimilarity...")


X = path_dist_mat
max_dim = int(np.ceil(np.log2(np.min(X.shape))))

if dim_reduce == "cmds":
    cmds = ClassicalMDS(dissimilarity="precomputed", n_components=max_dim)
    path_embed = cmds.fit_transform(X)
    sing_vals = cmds.singular_values_
elif dim_reduce == "iso":
    iso = Isomap(n_components=max_dim, metric="precomputed")
    path_embed = iso.fit_transform(path_dist_mat)
    sing_vals = iso.kernel_pca_.lambdas_

elbows, elbow_vals = select_dimension(sing_vals, n_elbows=3)
elbows = np.array(elbows)
rng = np.arange(1, len(sing_vals) + 1)

# screeplot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(rng, sing_vals, "o-")
pc = ax.scatter(elbows, elbow_vals, color="red", label="ZG")
pc.set_zorder(100)  # put red above
ax.legend()
stashfig(f"screeplot" + basename)

# pairplot of full embedding
pairplot(path_embed, alpha=0.02)
stashfig(f"pairs-all" + basename)

# %% [markdown]
# ## Cluster and plot on the embedding
print("Running AGMM on path embedding")
n_components = elbows[0]
print(f"Using {n_components} dimensions")

agmm = AutoGMMCluster(max_components=50, n_jobs=-2)
pred = agmm.fit_predict(path_embed[:, :n_components])

print(f"Number of clusters: {agmm.n_components_}")


pg = pairplot(
    path_embed[:, :n_components],
    alpha=0.05,
    labels=pred,
    palette=cc.glasbey_light,
    legend_name="Cluster",
)
leg = pg._legend
for lh in leg.legendHandles:
    lh.set_alpha(1)
stashfig(f"pairplot-agmm" + basename)

# %% [markdown]
# ## Plot the dissimilarity sorted by this clustering
color_dict = dict(zip(np.unique(pred), cc.glasbey_light))
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    path_dist_mat,
    sort_class=pred,
    cmap=None,
    center=None,
    ax=ax,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle="--"),
    ticks=False,
    colors=pred,
    palette=color_dict,
    cbar=False,
)
stashfig(f"dissim-agmm" + basename)

# %% [markdown]
# ## Plot the path indicator mat

Z = linkage(squareform(path_dist_mat), method="average", optimal_ordering=False)
R = dendrogram(
    Z, truncate_mode=None, get_leaves=True, no_plot=True, color_threshold=-np.inf
)
order = invert_permutation(R["leaves"])

path_meta = pd.DataFrame()
path_meta["cluster"] = pred
path_meta["dend_order"] = order

Z = linkage(squareform(pdist), method="average", optimal_ordering=False)
R = dendrogram(
    Z, truncate_mode=None, get_leaves=True, no_plot=True, color_threshold=-np.inf
)
order = invert_permutation(R["leaves"])

meta["dend_order"] = order
meta["signal_flow"] = -signal_flow(adj)
meta["class2"].fillna(" ", inplace=True)

mean_visits = []
for i in range(path_indicator_mat.shape[1]):
    mean_visit = np.mean(path_indicator_mat[:, i][path_indicator_mat[:, i] != 0])
    # print(mean_visit)
    if np.isnan(mean_visit):
        mean_visit = path_len + 1
    mean_visits.append(mean_visit)

meta["mean_visit"] = mean_visits
meta["int_mean_visit"] = meta["mean_visit"].astype(int)

pseudo_classes = [
    "LHN",
    "LHN2",
    "CN",
    "CX",
    "CN2",
    "CN;LHN",
    "FB2N",
    "FAN",
    "FBN",
    "FFN",
    "unk",
]
class1 = meta["class1"].values
class2 = meta["class2"].values
new_class1 = []
new_class2 = []

for i, mc in enumerate(class1):
    if mc in pseudo_classes:
        new_class1.append("u")
        new_class2.append(str(int(np.round(mean_visits[i]))))
        # print(str(int(mean_visits[i])))
    else:
        new_class1.append(mc)
        new_class2.append(class2[i])

meta["pseudo_class1"] = new_class1
meta["pseudo_class2"] = new_class2

visited = path_indicator_mat.sum(axis=0)
visited = visited > 0
plot_path_indicator_mat = path_indicator_mat[:, visited]
plot_meta = meta.loc[visited]

pal = sns.color_palette("tab10", n_colors=10)
pal = [pal[0], pal[2], pal[4], pal[6], pal[1], pal[3], pal[5], pal[7], (0, 0, 0)]
pal = pal[:path_len]

fig, axs = plt.subplots(
    1, 2, figsize=(30, 20), gridspec_kw=dict(width_ratios=[0.95, 0.02], wspace=0.02)
)
ax = axs[0]
ax, div, top, left = matrixplot(
    plot_path_indicator_mat,
    ax=ax,
    plot_type="scattermap",
    # col_sort_class=["pseudo_class1", "pseudo_class2"],
    # col_class_order="mean_visit",
    col_ticks=True,
    tick_rot=90,
    col_meta=plot_meta,
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_item_order="dend_order",
    col_tick_pad=[0.5, 1.5],
    # col_ticks=False,
    row_meta=path_meta,
    # row_sort_class="cluster",
    row_item_order="dend_order",
    row_ticks=True,
    gridline_kws=dict(linewidth=0.3, color="grey", linestyle="--"),
    sizes=(2, 2),
    hue="weight",
    palette=pal,
)
# ax.set_ylabel("Cluster")
ax.set_ylabel("Path")
top.set_xlabel("Neuron")
ax = axs[1]
palplot(pal, ax=ax)
ax.yaxis.tick_right()
ax.set_title("Hop")
ax.set_yticklabels(np.arange(1, path_len + 1))

stashfig("path-indcator-dendrosort" + basename)


fig, axs = plt.subplots(
    1, 2, figsize=(30, 20), gridspec_kw=dict(width_ratios=[0.95, 0.02], wspace=0.02)
)
ax = axs[0]
ax, div, top, left = matrixplot(
    plot_path_indicator_mat,
    ax=ax,
    plot_type="scattermap",
    col_sort_class=["pseudo_class1", "pseudo_class2"],
    col_class_order="mean_visit",
    col_ticks=True,
    tick_rot=90,
    col_meta=plot_meta,
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_item_order="dend_order",
    col_tick_pad=[0.5, 1.5],
    # col_ticks=False,
    row_meta=path_meta,
    row_sort_class="cluster",
    row_item_order="dend_order",
    row_ticks=True,
    gridline_kws=dict(linewidth=0.3, color="grey", linestyle="--"),
    sizes=(2, 2),
    hue="weight",
    palette=pal,
)
ax.set_ylabel("Cluster")
# ax.set_ylabel("Path")
# top.set_xlabel("Neuron")
ax = axs[1]
palplot(pal, ax=ax)
ax.yaxis.tick_right()
ax.set_title("Hop")
ax.set_yticklabels(np.arange(1, path_len + 1))

stashfig("path-indcator-class" + basename)

# %% [markdown]
# ##
# TODO show the dendrogram here?
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

fig, axs = plt.subplots(
    1, 2, figsize=(30, 20), gridspec_kw=dict(width_ratios=[0.95, 0.02], wspace=0.02)
)
ax = axs[0]
ax, div, top, left = matrixplot(
    plot_path_indicator_mat,
    ax=ax,
    plot_type="scattermap",
    # col_sort_class=["pseudo_class1", "pseudo_class2"],
    # col_class_order="mean_visit",
    col_ticks=True,
    tick_rot=90,
    col_meta=plot_meta,
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_item_order="mean_visit",
    # col_item_order="dend_order",
    col_tick_pad=[0.5, 1.5],
    # col_ticks=False,
    row_meta=path_meta,
    # row_sort_class="cluster",
    row_item_order="dend_order",
    row_ticks=True,
    gridline_kws=dict(linewidth=0.3, color="grey", linestyle="--"),
    sizes=(2, 2),
    hue="weight",
    palette=pal,
)
# ax.set_ylabel("Cluster")
ax.set_ylabel("Path")
top.set_xlabel("Neuron")
ax = axs[1]
palplot(pal, ax=ax)
ax.yaxis.tick_right()
ax.set_title("Hop")
ax.set_yticklabels(np.arange(1, path_len + 1))

from src.visualization import remove_shared_ax, remove_spines

dend_ax = div.append_axes("left", size="8%", pad=0, sharey=ax)
remove_shared_ax(dend_ax)
remove_spines(dend_ax)


from matplotlib.collections import LineCollection

dcoord = R["icoord"]
icoord = R["dcoord"]
coords = zip(icoord, dcoord)
tree_kws = {"linewidths": 0.3, "colors": ".2"}
lines = LineCollection([list(zip(x, y)) for x, y in coords], **tree_kws)
dend_ax.add_collection(lines)
number_of_leaves = len(order)
max_coord = max(map(max, icoord))
dend_ax.set_ylim(0, number_of_leaves * 10)
dend_ax.set_xlim(max_coord * 1.05, 0)
stashfig("path-indcator-dendro-path" + basename)
# %% [markdown]
# ## Plot each cluster separately.
save_all = False

uni_pred = np.unique(pred)

if save_all:
    for up in uni_pred:
        mask = pred == up
        hop_hist = np.zeros((path_len, len(meta)))
        sub_path_mat = path_indicator_mat[mask]
        for i in range(path_len):
            num_at_visit = np.count_nonzero(sub_path_mat == i + 1, axis=0)
            hop_hist[i, :] = num_at_visit

        n_visits = hop_hist.sum(axis=0)
        sub_hop_hist = hop_hist[:, n_visits > 0]
        sub_meta = meta[n_visits > 0].copy()
        sum_visit = (sub_hop_hist * np.arange(1, path_len + 1)[:, None]).sum(axis=0)
        sub_n_visits = sub_hop_hist.sum(axis=0)
        mean_visit = sum_visit / sub_n_visits
        sub_meta["mean_visit_int"] = mean_visit.astype(int)
        sub_meta["mean_visit"] = mean_visit

        plot_mat = np.log10(sub_hop_hist + 1)

        Z = linkage(plot_mat.T, metric="cosine", method="average")
        R = dendrogram(Z, no_plot=True, color_threshold=-np.inf)
        order = R["leaves"]
        sub_meta["dend_order"] = invert_permutation(order)

        title = f"Cluster {up}, {len(sub_path_mat) / len(paths):0.2f} paths, {len(sub_meta)} neurons"

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax, div, top_ax, left_ax = matrixplot(
            plot_mat,
            ax=ax,
            col_sort_class=["mean_visit_int"],
            # col_class_order="mean_visit",
            col_ticks=False,
            col_meta=sub_meta,
            col_colors="merge_class",
            col_palette=CLASS_COLOR_DICT,
            col_item_order=["merge_class", "dend_order"],
            cbar=False,
            gridline_kws=dict(linewidth=0.3, color="grey", linestyle="--"),
        )
        ax.set_yticks(np.arange(1, path_len + 1) - 0.5)
        ax.set_yticklabels(np.arange(1, path_len + 1))
        ax.set_ylabel("Hops")
        top_ax.set_title(title)
        stashfig(f"hop_hist-class-cluster={up}" + basename)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax, div, top_ax, left_ax = matrixplot(
            plot_mat,
            ax=ax,
            col_sort_class=["mean_visit_int"],
            # col_class_order="mean_visit",
            col_ticks=False,
            col_meta=sub_meta,
            col_colors="merge_class",
            col_palette=CLASS_COLOR_DICT,
            col_item_order=["dend_order"],
            cbar=False,
            gridline_kws=dict(linewidth=0.3, color="grey", linestyle="--"),
        )
        ax.set_yticks(np.arange(1, path_len + 1) - 0.5)
        ax.set_yticklabels(np.arange(1, path_len + 1))
        ax.set_ylabel("Hops")
        top_ax.set_title(title)
        stashfig(f"hop_hist-cluster={up}" + basename)
        plt.close()

        sub_meta["colors"] = sub_meta["merge_class"].map(CLASS_COLOR_DICT)
        sub_meta_dict = sub_meta.set_index("inds").to_dict(orient="index")

        cluster_paths = []
        for i, p in enumerate(paths):
            if mask[i]:
                cluster_paths.append(p)
        path_graph = to_path_graph(cluster_paths)
        nx.set_node_attributes(path_graph, sub_meta_dict)

        for n, d in path_graph.degree():
            path_graph.nodes[n]["degree"] = d

        ax = draw_networkx_nice(
            path_graph,
            "dend_order",
            "mean_visit",
            colors="colors",
            sizes="degree",
            draw_labels=False,
            size_scale=2,
            weight_scale=0.25,
        )
        ax.invert_yaxis()
        ax.set_title(title)
        stashfig(f"path-graph-cluster={up}" + basename)
        plt.close()


# %% [markdown]
# ## enbed the node dissimilarity matrix

manifold = TSNE(metric="precomputed")
cos_embed = manifold.fit_transform(pdist)
plot_df = pd.DataFrame(data=cos_embed)
plot_df["merge_class"] = meta["merge_class"].values

# %% [markdown]
# ## plot by cluster
mask = pred == 9
cluster_paths = []
for i, p in enumerate(paths):
    if mask[i]:
        cluster_paths.append(p)


fig, axs = plt.subplots(
    1, 2, figsize=(10, 10), gridspec_kw=dict(width_ratios=[0.95, 0.03], wspace=0)
)

ax = axs[0]
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

for p in cluster_paths:
    for t, (start, end) in enumerate(nx.utils.pairwise(p)):
        x1, y1 = plot_df.loc[start, [0, 1]]
        x2, y2 = plot_df.loc[end, [0, 1]]

        lc = ax.plot([x1, x2], [y1, y2], color=pal[t], linewidth=0.1, alpha=0.5)

sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=10,
)
ax.axis("equal")

ax = axs[1]
palplot(pal, ax=ax)
ax.yaxis.tick_right()
ax.set_title("Hop")
ax.set_yticklabels(np.arange(1, path_len + 1))

# %% [markdown]
# ## plot by pairs of path

triu_inds = np.triu_indices_from(path_dist_mat, k=1)
all_path_dists = path_dist_mat[triu_inds]


fig, axs = plt.subplots(
    1, 3, figsize=(20, 10), gridspec_kw=dict(width_ratios=[0.48, 0.48, 0.02], wspace=0)
)

ax = axs[0]

q = np.quantile(all_path_dists, 0.05)
close_to_q = np.isclose(all_path_dists, q)
close_paths = (triu_inds[0][close_to_q], triu_inds[1][close_to_q])

i = 4
p1_ind = close_paths[0][i]
p2_ind = close_paths[1][i]

p = paths[p1_ind]
for t, (start, end) in enumerate(nx.utils.pairwise(p)):
    x1, y1 = plot_df.loc[start, [0, 1]]
    x2, y2 = plot_df.loc[end, [0, 1]]
    ax.plot([x1, x2], [y1, y2], color=pal[t], linewidth=2, alpha=0.5)

p = paths[p2_ind]
for t, (start, end) in enumerate(nx.utils.pairwise(p)):
    x1, y1 = plot_df.loc[start, [0, 1]]
    x2, y2 = plot_df.loc[end, [0, 1]]
    ax.plot([x1, x2], [y1, y2], color=pal[t], linewidth=2, alpha=0.5)

sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=10,
)
# ax.axis("off")
# ax.spines["left"].set_visible(True)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(True)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.axis("equal")
ax.set_title("Similar paths (5th percentile)")

##
ax = axs[1]

q = np.quantile(all_path_dists, 0.95)
close_to_q = np.isclose(all_path_dists, q)
close_paths = (triu_inds[0][close_to_q], triu_inds[1][close_to_q])

i = 2
p1_ind = close_paths[0][i]
p2_ind = close_paths[1][i]

p = paths[p1_ind]
for t, (start, end) in enumerate(nx.utils.pairwise(p)):
    x1, y1 = plot_df.loc[start, [0, 1]]
    x2, y2 = plot_df.loc[end, [0, 1]]
    ax.plot([x1, x2], [y1, y2], color=pal[t], linewidth=2, alpha=0.5)

p = paths[p2_ind]
for t, (start, end) in enumerate(nx.utils.pairwise(p)):
    x1, y1 = plot_df.loc[start, [0, 1]]
    x2, y2 = plot_df.loc[end, [0, 1]]
    ax.plot([x1, x2], [y1, y2], color=pal[t], linewidth=2, alpha=0.5)

sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=10,
)
ax.axis("off")
ax.axis("equal")
ax.set_title("Dissimilar paths (95th percentile)")

##
ax = axs[2]
palplot(pal, ax=ax)
ax.yaxis.tick_right()
ax.set_title("Hop")
ax.set_yticklabels(np.arange(1, path_len + 1))

stashfig("example-paths-tsne" + basename)
# %%
