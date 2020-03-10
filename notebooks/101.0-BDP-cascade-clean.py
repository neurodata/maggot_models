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

from anytree import LevelOrderGroupIter, Node, RenderTree
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from graspy.embed import AdjacencySpectralEmbed, selectSVD
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
    matrixplot,
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

graph_type = "Gad"
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
# # Use a method to generate visits

p = 0.01
not_probs = (1 - p) ** adj  # probability of none of the synapses causing postsynaptic
probs = 1 - not_probs  # probability of ANY of the synapses firing onto next

max_depth = 5
n_bins = 5

n_sims = 100


def cascades_from_node(
    start_ind, probs, stop_inds=[], max_depth=10, n_sims=1000, seed=None
):
    np.random.seed(seed)
    node_hist_mat = np.zeros((n_verts, n_bins), dtype=int)
    for n in range(n_sims):
        root = Node(start_ind)
        root = generate_random_cascade(
            root, probs, 1, stop_inds=stop_inds, visited=[], max_depth=max_depth
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
hist_mat = np.concatenate(outs, axis=-1)
log_mat = np.log10(hist_mat + 1)
# %% [markdown]
# # Sort metadata

# row metadata
ids = pd.Series(index=meta["idx"], data=meta.index, name="id")
to_class = ids.map(meta["Merge Class"])
to_class.name = "to_class"
row_df = pd.concat([ids, to_class], axis=1)

# col metadata
orders = pd.Series(data=len(from_inds) * list(range(n_bins)), name="order")
from_idx = pd.Series(data=np.repeat(from_inds, n_bins), name="from_idx")
from_ids = from_idx.map(ids)
from_ids.name = "from_id"
from_class = from_ids.map(meta["Merge Class"])
from_class.name = "from_class"
col_df = pd.concat([orders, from_idx, from_ids, from_class], axis=1)

# %% [markdown]
# #
shape = log_mat.shape
figsize = tuple(i / 40 for i in shape)
fig, ax = plt.subplots(1, 1, figsize=figsize)
matrixplot(
    log_mat,
    ax=ax,
    col_meta=col_df,
    col_sort_class=["from_class"],
    row_meta=row_df,
    row_sort_class=["to_class"],
    plot_type="scattermap",
    sizes=(0.5, 0.5),
)
stashfig("first-matrixplot-scatter")

fig, ax = plt.subplots(1, 1, figsize=figsize)
matrixplot(
    log_mat,
    ax=ax,
    col_meta=col_df,
    col_sort_class=["from_class"],
    row_meta=row_df,
    row_sort_class=["to_class"],
    plot_type="heatmap",
    sizes=(0.5, 0.5),
)
stashfig("first-matrixplot-heatmap")

# %% [markdown]
# # Screeplots
screeplot(hist_mat.astype(float), title="Raw hist mat (full)")
stashfig("scree-hist-mat")
screeplot(log_mat, title="Log hist mat (full)")
stashfig("scree-log-mat")

# %% [markdown]
# # Pairplots

pca = PCA(n_components=6)
embed = pca.fit_transform(log_mat)
loadings = pca.components_.T
pg = pairplot(
    embed,
    labels=to_class.values,
    palette=CLASS_COLOR_DICT,
    height=5,
    title="Node response embedding (log)",
)
pg._legend.remove()
stashfig("node-pca-log")
pg = pairplot(
    loadings, labels=from_class.values, height=5, title="Source class embedding (log)"
)
stashfig("source-pca-log")

pca = PCA(n_components=6)
embed = pca.fit_transform(hist_mat.astype(float))
loadings = pca.components_.T
pg = pairplot(
    embed,
    labels=to_class.values,
    palette=CLASS_COLOR_DICT,
    height=5,
    title="Node response embedding (raw)",
)
pg._legend.remove()
stashfig("node-pca-log")
pg = pairplot(
    loadings, labels=from_class.values, height=5, title="Source class embedding (raw)"
)
stashfig("source-pca-log")

# %% [markdown]
# # Collapse that matrix
collapsed_hist = []
collapsed_col_df = []
for fc in from_class.unique():
    from_df = col_df[col_df["from_class"] == fc]
    for order in from_df["order"].unique():
        inds = from_df[from_df["order"] == order].index
        col = hist_mat[:, inds].sum(axis=1)
        collapsed_hist.append(col)
        row = {"order": order, "from_class": fc}
        collapsed_col_df.append(row)
collapsed_col_df = pd.DataFrame(collapsed_col_df)
collapsed_hist = np.array(collapsed_hist).T
log_collapsed_hist = np.log10(collapsed_hist)

fig, ax = plt.subplots(1, 1, figsize=(10, 20))
matrixplot(
    log_collapsed_hist,
    ax=ax,
    col_meta=collapsed_col_df,
    col_sort_class=["from_class"],
    row_meta=row_df,
    row_sort_class=["to_class"],
    plot_type="heatmap",
    sizes=(0.5, 0.5),
)
stashfig("collapsed-matrixplot-heatmap")
