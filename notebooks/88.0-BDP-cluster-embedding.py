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
threshold = 1
weight = "weight"
graph_type = "Gad"
cutoff = 8
base = f"-c{cutoff}-t{threshold}-{graph_type}"

base_path = Path(f"./maggot_models/notebooks/outs/{run_name}/csvs")
meta = pd.read_csv(base_path / str("meta" + base + ".csv"), index_col=0)
path_mat = pd.read_csv(
    base_path / str("prob-path-mat" + base + ".csv"), index_col=0
).values

base_path = Path(f"./maggot_models/notebooks/outs/{embed_name}/csvs")
embed_mat = pd.read_csv(base_path / str("euclid-mds-embed.csv"), index_col=0)

gmm = AutoGMMCluster(
    min_components=10,
    max_components=50,
    affinity="all",
    linkage="all",
    covariance_type="all",
    n_jobs=-2,
    verbose=30,
)
labels = gmm.fit_predict(embed_mat.values)

label_df = pd.DataFrame(data=labels)
stashcsv(label_df, "labels")

print("Finding mean paths")
mean_paths = []
uni_labels = np.unique(labels)
for ul in uni_labels:
    inds = np.where(labels == ul)[0]
    paths = path_mat[inds, :]
    mean_path = np.array(np.mean(paths, axis=0))
    mean_paths.append(mean_path)
mean_paths = np.squeeze(np.array(mean_paths))

mean_path_df = pd.DataFrame(data=mean_paths, columns=meta.index.values)
stashcsv(mean_path_df, "mean-paths")

