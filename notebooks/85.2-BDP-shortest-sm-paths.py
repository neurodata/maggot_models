# %% [markdown]
# #
import os
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, ClassicalMDS, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import get_lcc, symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph, preprocess
from src.io import savefig, saveskels, savecsv
from src.visualization import remove_spines, screeplot

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def pairwise_sparse_jaccard_distance(X, Y=None):
    """
    Computes the Jaccard distance between two sparse matrices or between all pairs in
    one sparse matrix.

    Args:
        X (scipy.sparse.csr_matrix): A sparse matrix.
        Y (scipy.sparse.csr_matrix, optional): A sparse matrix.

    Returns:
        numpy.ndarray: A similarity matrix.

    REF https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix
    """

    if Y is None:
        Y = X

    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)

    intersect = X.dot(Y.T)

    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = (xx + yy).T - intersect

    return (1 - intersect / union).A


VERSION = "2020-01-29"

graph_type = "Gad"
threshold = 1
weight = "weight"
mg = load_metagraph("Gad", VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=True,
    remove_pdiff=False,
    binarize=False,
    weight=weight,
)
print(mg.meta["Class 1"].unique())

out_classes = ["O_dVNC"]
sens_classes = ["sens"]


sparse_adj = nx.to_scipy_sparse_matrix(
    mg.g, weight=weight, nodelist=mg.meta.index.values
)

print("Running Dijkstra's")
# TODO need to also save the path order here
dists, predecessors = dijkstra(sparse_adj, return_predecessors=True)

meta = mg.meta.copy()
meta["idx"] = range(len(meta))
from_inds = meta[meta["Class 1"].isin(sens_classes)]["idx"].values
out_inds = meta[meta["Class 1"].isin(out_classes)]["idx"].values

path_mat = lil_matrix((len(from_inds) * len(out_inds), len(mg)), dtype=bool)
order_mat = lil_matrix((len(from_inds) * len(out_inds), len(mg)), dtype=int)

include_end = True
include_start = True

paths = []
orders = []
endpoints = []
for i, from_ind in enumerate(from_inds):
    for j, out_ind in enumerate(out_inds):
        curr_ind = predecessors[from_ind, out_ind]
        if curr_ind != -9999:
            path = []
            order = []
            loc = 0
            while curr_ind != from_ind:
                loc += 1
                path.append(curr_ind)
                order.append(loc)
                curr_ind = predecessors[from_ind, curr_ind]
            path.reverse()
            if include_start:
                path.insert(0, from_ind)
                order.append(order[-1] + 1)
            if include_end:
                path.append(out_ind)
                order.append(order[-1] + 1)
            paths.append(path)
            orders.append(order)
            endpoints.append((from_ind, out_ind))

order_mat = csr_matrix((orders, (range(len(paths), paths))))

# %% [markdown]
# #
# mins = np.min(order_mat, axis=1)
# order_mat[order_mat > 0] += mins[:, np.newaxis]

# if subsample paths
path_inds = np.random.randint(path_mat.shape[0], size=10000)
path_mat = path_mat[path_inds]
order_mat = order_mat[path_inds]


# remove not connected paths
# seems crazy that this is possible
row_sums = path_mat.sum(axis=1).A1
row_mask = row_sums != 0
path_mat = path_mat[row_mask]
order_mat = order_mat[row_mask]

# remove nodes that are never visited
col_sums = path_mat.sum(axis=0).A1
col_mask = col_sums != 0
path_mat = path_mat[:, col_mask]
order_mat = order_mat[:, col_mask]

meta = meta.iloc[col_mask, :]


# def metaheatmap(data, meta, sortby_classes=None, sortby_nodes=None, ascending=True):
#     meta = meta.copy()
#     meta.index = range(len(meta))
#     if not isinstance(sortby_classes, (list, tuple, np.ndarray)):
#         sortby_classes = [sortby_classes]
#     if not isinstance(sortby_nodes, (list, tuple, np.ndarray)):
#         sortby_nodes = [sortby_nodes]
#     sortby = sortby_classes + sortby_nodes
#     sort_meta = meta.sort_values(sortby, inplace=False, ascending=ascending)
#     first_df = sort_meta.groupby(sortby_classes).first()
#     first_inds = list(first_df.index.values)
#     first_inds.append(len(meta) + 1)


# metaheatmap(path_mat, meta, sortby_classes=["class_rank"], sortby_nodes=["mean_rank"])

# %% [markdown]
# #

from sklearn.manifold import MDS

path_mat = path_mat.tocsr()  # for fast mult

print("Finding pairwise jaccard distances")
pdist_sparse = pairwise_sparse_jaccard_distance(path_mat)

print(pdist_sparse.shape)

print("Embedding with MDS")
mds = ClassicalMDS(dissimilarity="precomputed")
# mds = MDS(dissimilarity="precomputed", n_components=6, n_init=16, n_jobs=-2)
jaccard_embedding = mds.fit_transform(pdist_sparse)

# %% [markdown]
# #

print("Clustering embedding")
agmm = AutoGMMCluster(
    min_components=10, max_components=40, affinity="euclidean", linkage="single"
)
labels = agmm.fit_predict(jaccard_embedding)

pairplot(
    jaccard_embedding, title="AGMM o CMDS o Jaccard o Sensorimotor Paths", labels=labels
)
savefig("AGMM-CMDS-jaccard-sm-path")

print("Finding mean paths")
mean_paths = []
uni_labels = np.unique(labels)
for ul in uni_labels:
    inds = np.where(labels == ul)[0]
    paths = path_mat[inds, :]
    mean_path = np.array(np.mean(paths, axis=0))
    mean_paths.append(mean_path)
mean_paths = np.squeeze(np.array(mean_paths))

# TODO remove sensory and motor indices from the matrix

# %% [markdown]
# #
print("Plotting clustermaps")
sns.set_context("talk")
clustergrid = sns.clustermap(
    mean_paths, cmap="Reds", figsize=(20, 10), row_cluster=True
)
clustergrid.fig.suptitle("Sensorimotor path clusters")
clustergrid.ax_heatmap.set_ylabel("Path cluster")
clustergrid.ax_heatmap.set_xlabel("Node")
stashfig("smpath-clustermap")


col_sums = mean_paths.sum(axis=0)
col_sums[col_sums == 0] = 1
std_mean_paths = mean_paths / col_sums[np.newaxis, :]
clustergrid = sns.clustermap(
    std_mean_paths, cmap="Reds", figsize=(20, 10), row_cluster=True, xticklabels=False
)
clustergrid.fig.suptitle("Sensoritmotor path clusters (column-normalized)")
clustergrid.ax_heatmap.set_ylabel("Path cluster")
clustergrid.ax_heatmap.set_xlabel("Node")
stashfig("smpath-clustermap-normalized")

# %% [markdown]
# #
clustergrid = sns.clustermap(
    mean_paths, cmap="Reds", figsize=(20, 10), col_cluster=False, xticklabels=False
)
clustergrid.fig.suptitle("Sensoritmotor path clusters (column-normalized)")
clustergrid.ax_heatmap.set_ylabel("Path cluster")
clustergrid.ax_heatmap.set_xlabel("Node")
stashfig("smpath-clustermap-normalized")

#%%

sort_class = "Class 1"

nnz = order_mat.getnnz(axis=0)
nnz[nnz == 0] = 1
mean_rank = np.squeeze(np.array(order_mat.sum(axis=0) / nnz))
meta["mean_rank"] = mean_rank
class_rank = meta.groupby(sort_class)["mean_rank"].median()
class_rank = meta[sort_class].map(class_rank)
class_rank.name = "class_rank"
meta = pd.concat((meta, class_rank), ignore_index=False, axis=1)
meta["idx"] = range(len(meta))
sort_meta = meta.sort_values(["class_rank", sort_class, "mean_rank"], inplace=False)
perm_inds = sort_meta.idx.values
# path_mat = path_mat[:, perm_inds]

# data = path_mat.todense().astype(float)
data = mean_paths.copy()
data = np.log10(data)
data[~np.isfinite(data)] = 0
data[data != 0] -= data.min()
data = data[:, perm_inds]

sort_meta["idx"] = range(len(sort_meta))
first_df = sort_meta.groupby([sort_class], sort=False).first()
first_inds = list(first_df["idx"].values)
first_inds.append(len(meta) + 1)
# for the tic locs
middle_df = sort_meta.groupby([sort_class], sort=False).mean()
middle_inds = list(middle_df["idx"].values)
middle_labels = list(middle_df.index)

# draw the plot, could be heatmap or gridmap here
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
sns.heatmap(data, cmap="RdBu_r", ax=ax, vmin=0, center=0)
ax.set_xticks(middle_inds)
ax.set_xticklabels(middle_labels)
for t in first_inds:
    ax.axvline(t - 0.5, color="grey", linestyle="--", alpha=0.5, linewidth=1)

axis = ax.xaxis
axis.set_major_locator(plt.FixedLocator(middle_inds[0::2]))
axis.set_minor_locator(plt.FixedLocator(middle_inds[1::2]))
axis.set_minor_formatter(plt.FormatStrFormatter("%s"))
ax.tick_params(which="minor", pad=40, length=10)
ax.tick_params(which="major", length=5)
ax.set_xticklabels(middle_labels[0::2])
ax.set_xticklabels(middle_labels[1::2], minor=True)
ax.xaxis.tick_top()
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
    tick.set_fontsize(10)
for tick in ax.get_xticklabels(minor=True):
    tick.set_rotation(45)
    tick.set_fontsize(10)

ax.set_ylabel("Cluster index")
ax.set_xlabel("Node index")
ax.set_title("AGMM o CMDS o Jaccard o Shortest Sensorimotor Paths")

stashfig("ssmpath-custom-heatmap-log")
