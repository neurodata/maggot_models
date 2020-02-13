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
from src.block import run_minimize_blockmodel
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
dists, predecessors = dijkstra(sparse_adj, return_predecessors=True)

mg.meta["idx"] = range(len(mg))
from_inds = mg.meta[mg.meta["Class 1"].isin(sens_classes)]["idx"].values
out_inds = mg.meta[mg.meta["Class 1"].isin(out_classes)]["idx"].values

path_mat = lil_matrix((len(from_inds) * len(out_inds), len(mg)), dtype=bool)

print("Reconstructing paths")
for i, from_ind in enumerate(from_inds):
    for j, out_ind in enumerate(out_inds):
        # the below is ignoring the actual start and end node in the path, hopefully
        curr_ind = predecessors[from_ind, out_ind]
        if curr_ind != -9999:
            while curr_ind != from_ind:  # key for no path in scipy
                path_mat[i * len(out_inds) + j, curr_ind] = True
                curr_ind = predecessors[from_ind, curr_ind]

# if subsample paths
# path_mat = path_mat[np.random.randint(path_mat.shape[0], size=10000)]

# remove not connected paths
# seems crazy that this is possible
row_sums = path_mat.sum(axis=1).A1
path_mat = path_mat[row_sums != 0]


path_mat = path_mat.tocsr()  # for fast mult

print("Finding pairwise jaccard distances")
pdist_sparse = pairwise_sparse_jaccard_distance(path_mat)

print(pdist_sparse.shape)

print("Embedding with MDS")
cmds = ClassicalMDS(dissimilarity="precomputed")
jaccard_embedding = cmds.fit_transform(pdist_sparse)

print("Clustering embedding")
agmm = AutoGMMCluster(
    min_components=2, max_components=20, affinity="euclidean", linkage="average"
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
