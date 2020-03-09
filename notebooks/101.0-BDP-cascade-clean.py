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

from graspy.embed import AdjacencySpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import get_lcc, symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels
from src.traverse import (
    generate_random_cascade,
    generate_random_walks,
    path_to_visits,
    to_markov_matrix,
    to_path_graph,
)
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

VERSION = "2020-03-09"
print(f"Using version {VERSION}")

graph_type = "G"
threshold = 0
weight = "weight"
all_out = False
mg = load_metagraph(graph_type, VERSION)
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
    "O_dSEZ",
    "O_dSEZ;CN",
    "O_dSEZ;LHN",
    "O_dVNC",
    "O_dVNC;O_RG",
    "O_dVNC;CN",
    "O_RG",
    "O_dUnk",
    "O_RG-IPC",
    "O_RG-ITP",
    "O_RG-CA-LP",
]

sens_classes = [
    "sens-ORN",
    "sens-photoRh5",
    "sens-photoRh6",
    "sens-PaN",
    "sens-MN",
    "sens-thermo",
    "sens-vtn",
]
class_key = "Merge Class"

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
n_verts = len(adj)
meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))

from_inds = meta[meta[class_key].isin(sens_classes)]["idx"].values
out_inds = meta[meta[class_key].isin(out_classes)]["idx"].values
ind_map = dict(zip(meta.index, meta["idx"]))
g = nx.relabel_nodes(g, ind_map, copy=True)
out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# # Try the propogation thing

p = 0.01
not_probs = (1 - p) ** adj  # probability of none of the synapses causing postsynaptic
probs = 1 - not_probs  # probability of ANY of the synapses firing onto next


# %% [markdown]
# #
from anytree import Node, RenderTree, LevelOrderGroupIter

max_depth = 5
n_bins = 5

start_ind = 2
n_sims = 1000


def cascades_from_node(
    start_ind, probs, stop_inds=[], max_depth=10, n_sims=1000, seed=None
):
    np.random.seed(seed)
    node_hist_mat = np.zeros((n_verts, n_bins * n_verts), dtype=int)
    for n in range(n_sims):
        root = Node(start_ind)
        root = generate_random_cascade(
            root, probs, 0, stop_inds=stop_inds, visited=[], max_depth=max_depth
        )
        for level, children in enumerate(LevelOrderGroupIter(root)):
            for node in children:
                node_hist_mat[node.name, level] += 1
    return node_hist_mat


seeds = np.random.choice(int(1e8), size=len(from_inds), replace=False)
outs = Parallel(n_jobs=-2, verbose=10)(
    delayed(cascades_from_node)(fi, probs, out_inds, max_depth, n_sims, seed)
    for fi, seed in zip(from_inds, seeds)
)


# %% [markdown]
# #
fig, ax = plt.subplots(figsize=(5, 25))
# log_mat = np.log10(hist_mat)
mini_hist_mat = hist_mat[:, start_ind * n_bins : (start_ind + 1) * n_bins]
# sums = mini_hist_mat.sum(axis=0)
# sums[sums == 0] = 1
# log_mat = mini_hist_mat / sums
log_mat = np.log10(mini_hist_mat + 1)
sns.heatmap(log_mat, cmap="RdBu_r", center=0)
stashfig("cascade-example")
# %% [markdown]
# #
sums = np.sum(mini_hist_mat, axis=0)
# %% [markdown]
# #
p = probs.ravel()
p = p[p != 0]
sns.distplot(p)
