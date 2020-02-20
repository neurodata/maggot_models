#%%
import os
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
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
from src.io import savecsv, savefig, saveskels
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


run_name = "86.1-BDP-prob-path-cluster"
embed_name = "87.0-BDP-read-prob-paths"
cluster_name = "88.0-BDP-cluster-embedding"
threshold = 1
weight = "weight"
graph_type = "Gad"
cutoff = 8
base = f"-c{cutoff}-t{threshold}-{graph_type}"

base_path = Path(f"./maggot_models/notebooks/outs/{run_name}/csvs")
meta = pd.read_csv(base_path / str("meta" + base + ".csv"), index_col=0)
path_mat = pd.read_csv(base_path / str("prob-path-mat" + base + ".csv"), index_col=0)

base_path = Path(f"./maggot_models/notebooks/outs/{embed_name}/csvs")
embed_mat = pd.read_csv(base_path / str("euclid-mds-embed.csv"), index_col=0)

base_path = Path(f"./maggot_models/notebooks/outs/{cluster_name}/csvs")
mean_path_mat = pd.read_csv(base_path / str("mean-paths.csv"), index_col=0)
mean_paths = mean_path_mat.values
# %% [markdown]
# #

sort_class = "Class 1"
# nnz = order_mat.getnnz(axis=0)
# nnz[nnz == 0] = 1
# mean_rank = np.squeeze(np.array(order_mat.sum(axis=0) / nnz))
# meta["mean_rank"] = mean_rank
# class_rank = meta.groupby(sort_class)["mean_rank"].median()
# class_rank = meta[sort_class].map(class_rank)
# class_rank.name = "class_rank"
# meta = pd.concat((meta, class_rank), ignore_index=False, axis=1)
meta["idx"] = range(len(meta))
sort_meta = meta.sort_values([sort_class], inplace=False)
perm_inds = sort_meta.idx.values
# path_mat = path_mat[:, perm_inds]

# data = path_mat.todense().astype(float)
data = mean_paths.copy()
col_sums = data.sum(axis=0)
data = data / col_sums
# data = np.log10(data)
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
