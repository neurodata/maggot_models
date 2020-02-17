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

out_classes = ["O_dVNC"]
sens_classes = ["sens"]
print(f"Finding paths from {sens_classes} to {out_classes}")

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
prob_mat = adj.copy()
row_sums = prob_mat.sum(axis=1)
row_sums[row_sums == 0] = 1
prob_mat = prob_mat / row_sums[:, np.newaxis]

meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))
from_inds = meta[meta["Class 1"].isin(sens_classes)]["idx"].values
out_inds = meta[meta["Class 1"].isin(out_classes)]["idx"].values

ind_map = dict(zip(meta.index, meta["idx"]))
g = nx.relabel_nodes(g, ind_map, copy=True)
#%%

out_ind_map = dict(zip(out_inds, range(len(out_inds))))
weighted_pair_paths = []
cutoff = 6
for i, from_ind in enumerate(from_inds[:1]):
    paths = nx.all_simple_paths(g, from_ind, out_inds, cutoff=cutoff)
    path_probs = np.zeros((len(out_inds), len(g)))
    for path, pairpath in zip(paths, map(nx.utils.pairwise, paths)):
        path_prob = 1
        out_ind = path[-1]
        path_inds = path[:-1]  # don't include the last in the matrix
        # though, the probability of that edge is included...
        for pair in pairpath:
            path_prob *= prob_mat[pair]
        path_probs[out_ind_map[out_ind], path_inds] += path_prob

    # normalize
    max_probs = path_probs.max(axis=1)
    max_probs[max_probs == 0] = 1
    path_probs = path_probs / max_probs[:, np.newaxis]
    path_probs[from_ind, :] = 0  # don't count the start node, not interesting

    # TODO what do I normalize this by...
    # path probs is now the pmf of probability of visiting node k on a simple path
    # from node i to node j of length at most = to cutoff
    # could just normalize by the max prob here...
    # I think this makes sense because the probability of visiting i,j should be 1
    weighted_pair_paths.append(path_probs)

weighted_pair_paths = np.concatenate(weighted_pair_paths, axis=0)
path_mat = csr_matrix(weighted_pair_paths)
print(path_mat.shape)

# %% [markdown]
# # write the above in parallel


def prob_path_search(from_ind):
    paths = nx.all_simple_paths(g, from_ind, out_inds, cutoff=cutoff)
    path_probs = np.zeros((len(out_inds), len(g)))
    for path, pairpath in zip(paths, map(nx.utils.pairwise, paths)):
        path_prob = 1
        out_ind = path[-1]
        path_inds = path[:-1]  # don't include the last in the matrix
        # though, the probability of that edge is included...
        for pair in pairpath:
            path_prob *= prob_mat[pair]
        path_probs[out_ind_map[out_ind], path_inds] += path_prob
    # normalize
    max_probs = path_probs.max(axis=1)
    max_probs[max_probs == 0] = 1
    path_probs = path_probs / max_probs[:, np.newaxis]
    path_probs[from_ind, :] = 0  # don't count the start node, not interesting
    return path_probs


# %% [markdown]
# #


outs = Parallel(n_jobs=-2, verbose=10)(
    delayed(prob_path_search)(i) for i in from_inds[:10]
)
weighted_pair_paths = np.concatenate(outs, axis=0)
path_mat = csr_matrix(weighted_pair_paths)
print(path_mat.shape)
