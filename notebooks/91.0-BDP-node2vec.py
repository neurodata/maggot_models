# %% [markdown]
# #
import itertools
import os
import time
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import textdistance
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, ClassicalMDS, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import get_lcc, symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels
from src.visualization import (
    CLASS_COLOR_DICT,
    barplot_text,
    draw_networkx_nice,
    remove_spines,
    screeplot,
    stacked_barplot,
)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


#%% Load and preprocess the data

VERSION = "2020-01-29"
print(f"Using version {VERSION}")

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
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")

out_classes = [
    "O_dVNC",
    "O_dSEZ",
    "O_IPC",
    "O_ITP",
    "O_dSEZ;FFN",
    "O_CA-LP",
    "O_dSEZ;FB2N",
]
sens_classes = ["sens"]

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
prob_mat = adj.copy()
row_sums = prob_mat.sum(axis=1)
dead_inds = np.where(row_sums == 0)[0]
row_sums[row_sums == 0] = 1
prob_mat = prob_mat / row_sums[:, np.newaxis]

n_verts = len(prob_mat)
meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))
from_inds = meta[meta["Class 1"].isin(sens_classes)]["idx"].values
out_inds = meta[meta["Class 1"].isin(out_classes)]["idx"].values

ind_map = dict(zip(meta.index, meta["idx"]))
g = nx.relabel_nodes(g, ind_map, copy=True)

out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# #
from topologic.embedding import node2vec_embedding

# start_nodes = list(meta[meta["Class 1"].isin(sens_classes)].index.values)
from_inds = list(meta[meta["Class 1"].isin(sens_classes)]["idx"].values)


embed = node2vec_embedding(
    g,
    num_walks=100,
    walk_length=50,
    dimensions=8,
    window_size=1,
    iterations=3,
    inout_hyperparameter=1,
    return_hyperparameter=0.1,
    start_nodes=from_inds,
)
latent = embed[0]
node_labels = embed[1]
node_labels = np.vectorize(int)(node_labels)
from graspy.plot import pairplot

class_labels = meta.iloc[node_labels, meta.columns.get_loc("Merge Class")].values

pairplot(latent, labels=class_labels, palette=CLASS_COLOR_DICT)
