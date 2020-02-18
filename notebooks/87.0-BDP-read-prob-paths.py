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
threshold = 1
weight = "weight"
graph_type = "Gad"
cutoff = 8
base = f"-c{cutoff}-t{threshold}-{graph_type}"


base_path = Path(f"./maggot_models/notebooks/outs/{run_name}/csvs")
meta = pd.read_csv(base_path / str("meta" + base + ".csv"), index_col=0)
path_mat = pd.read_csv(base_path / str("prob-path-mat" + base + ".csv"), index_col=0)

sparse_path = csr_matrix(path_mat.values)

euclid_dists = pairwise_distances(sparse_path, metric="euclidean")

mds = ClassicalMDS(dissimilarity="precomputed")
mds_embed = mds.fit_transform(euclid_dists)
embed_df = pd.DataFrame(data=mds_embed)

stashcsv(embed_df, "euclid-mds-embed")
