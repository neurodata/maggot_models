# %% [markdown]
# #
import itertools
import os
import time
from itertools import chain

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, Node, RenderTree
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from graspy.plot import heatmap, pairplot
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels
from src.traverse import (
    cascades_from_node,
    generate_cascade_tree,
    generate_random_walks,
    path_to_visits,
    to_markov_matrix,
    to_path_graph,
)
from src.visualization import (
    CLASS_COLOR_DICT,
    _draw_seperators,
    barplot_text,
    draw_networkx_nice,
    matrixplot,
    remove_spines,
    screeplot,
    sort_meta,
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
mg = load_metagraph(graph_type, VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=False,
    remove_pdiff=False,
    binarize=False,
    weight=weight,
)
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")

# TODO update this with the mixed groups
# TODO make these functional for selecting proper paths
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
# out_classes = []

from_groups = [
    ("sens-ORN",),
    ("sens-photoRh5", "sens-photoRh6"),
    ("sens-MN",),
    ("sens-thermo",),
    ("sens-vtd",),
    ("sens-AN",),
]
# from_groups = [("O_dVNC",)]
from_group_names = ["Odor", "Photo", "MN", "Temp", "VTD", "AN"]
from_classes = list(chain.from_iterable(from_groups))  # make this a flat list

class_key = "Merge Class"

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
n_verts = len(adj)
meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))

from_inds = meta[meta[class_key].isin(from_classes)]["idx"].values
out_inds = meta[meta[class_key].isin(out_classes)]["idx"].values
ind_map = dict(zip(meta.index, meta["idx"]))
g = nx.relabel_nodes(g, ind_map, copy=True)
out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# # Use a method to generate visits

path_type = "cascade"
p = 0.01
not_probs = (1 - p) ** adj  # probability of none of the synapses causing postsynaptic
probs = 1 - not_probs  # probability of ANY of the synapses firing onto next
seed = 8888
max_depth = 10
n_bins = 10
n_sims = 100
method = "path"
normalize_n_source = False


basename = f"-{graph_type}-t{threshold}-pt{path_type}-b{n_bins}-n{n_sims}-m{method}"
basename += f"-norm{normalize_n_source}"

np.random.seed(seed)
if method == "tree":
    seeds = np.random.choice(int(1e8), size=len(from_inds), replace=False)
    outs = Parallel(n_jobs=-2, verbose=10)(
        delayed(cascades_from_node)(
            fi, probs, out_inds, max_depth, n_sims, seed, n_bins, method
        )
        for fi, seed in zip(from_inds, seeds)
    )
elif method == "path":
    outs = []
    for start_ind in from_inds:
        temp_hist = cascades_from_node(
            start_ind, probs, out_inds, max_depth, n_sims, seed, n_bins, method
        )
        outs.append(temp_hist)
hist_mat = np.concatenate(outs, axis=-1)


# generate_cascade_paths(start_ind, probs, 1, stop_inds=out_inds, max_depth=10)
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
log_mat = np.log10(hist_mat + 1)
shape = log_mat.shape
# figsize = tuple(i / 40 for i in shape)
figsize = (10, 20)
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
    tick_rot=45,
)
stashfig("log-full-scatter" + basename)

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
    tick_rot=45,
)
stashfig("log-full-heat" + basename)

# %% [markdown]
# # Screeplots
screeplot(hist_mat.astype(float), title="Raw hist mat (full)")
stashfig("scree-raw-mat" + basename)
screeplot(log_mat, title="Log hist mat (full)")
stashfig("scree-log-mat" + basename)

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
stashfig("node-pca-log" + basename)
pg = pairplot(
    loadings, labels=from_class.values, height=5, title="Source class embedding (log)"
)
stashfig("source-pca-log" + basename)

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
stashfig("node-pca-log" + basename)
pg = pairplot(
    loadings, labels=from_class.values, height=5, title="Source class embedding (raw)"
)
stashfig("source-pca-log" + basename)

# %% [markdown]
# # Collapse that matrix
collapsed_hist = []
collapsed_col_df = []

for fg, fg_name in zip(from_groups, from_group_names):
    from_df = col_df[col_df["from_class"].isin(fg)]
    n_in_group = len(from_df)
    for order in from_df["order"].unique():
        inds = from_df[from_df["order"] == order].index
        col = hist_mat[:, inds].sum(axis=1)
        if normalize_n_source:
            col = col.astype(float)
            col /= n_in_group
        collapsed_hist.append(col)
        row = {"order": order, "from_class": fg_name}
        collapsed_col_df.append(row)


collapsed_col_df = pd.DataFrame(collapsed_col_df)
collapsed_hist = np.array(collapsed_hist).T
log_collapsed_hist = np.log10(collapsed_hist + 1)

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
    tick_rot=0,
)
stashfig("collapsed-log-heat" + basename)

# %% [markdown]
# # clustermap the matrix


sns.set_context("talk", font_scale=1)
linkage = "average"
metric = "euclidean"
colors = np.vectorize(CLASS_COLOR_DICT.get)(row_df["to_class"])

perm_inds, sort_collapsed_col_df = sort_meta(
    collapsed_col_df, sort_class=["from_class"]
)
sort_log_collapsed_hist = log_collapsed_hist[:, perm_inds]


cg = sns.clustermap(
    data=sort_log_collapsed_hist,
    col_cluster=False,
    row_colors=colors,
    cmap="RdBu_r",
    center=0,
    cbar_pos=None,
    method=linkage,
    metric=metric,
)
ax = cg.ax_heatmap
_draw_seperators(
    ax, col_meta=sort_collapsed_col_df, col_sort_class=["from_class"], tick_rot=0
)
ax.yaxis.set_ticks([])
ax.set_xlabel(r"Visits over time $\to$")
ax.set_ylabel("Neuron")
stashfig("collapsed-log-clustermap" + basename)
stashfig("collapsed-log-clustermap" + basename, fmt="pdf")

# %% [markdown]
# #

# %% [markdown]
# # Do some plotting for illustration only
# from src.traverse import generate_cascade_paths
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


def remove_shared_ax(ax):
    shax = ax.get_shared_x_axes()
    shay = ax.get_shared_y_axes()
    shax.remove(ax)
    shay.remove(ax)
    for axis in [ax.xaxis, ax.yaxis]:
        ticker = mpl.axis.Ticker()
        axis.major = ticker
        axis.minor = ticker
        loc = mpl.ticker.NullLocator()
        fmt = mpl.ticker.NullFormatter()
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(fmt)


sns.set_context("talk")
# uPN 742
target_ind = 605
# target_ind = 743
# target_ind = 2282
# target_ind = 596
# target_ind = 2367
row = collapsed_hist[target_ind, :]
perm_inds, sort_col_df = sort_meta(collapsed_col_df, sort_class=["from_class"])
sort_row = row[perm_inds]

fig, ax = plt.subplots(1, 1)
xs = np.arange(len(sort_row)) + 0.5
divider = make_axes_locatable(ax)
bot_cax = divider.append_axes("bottom", size="3%", pad=0.02, sharex=ax)
remove_shared_ax(bot_cax)
sns.set_palette("Set1")
ax.bar(x=xs, height=sort_row, width=0.8)
_draw_seperators(ax, col_meta=sort_col_df, col_sort_class=["from_class"], tick_rot=0)
ax.set_xlim(0, len(xs))
ax.set_ylabel("# hits @ time")

sns.heatmap(
    collapsed_col_df["order"][None, :], ax=bot_cax, cbar=False, cmap="RdBu", center=0
)
bot_cax.set_xticks([])
bot_cax.set_yticks([])
bot_cax.set_xlabel(r"Time $\to$", x=0.1, ha="left", labelpad=-22)
bot_cax.set_xticks([20.5, 24.5, 28.5])
bot_cax.set_xticklabels([1, 5, 9])

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_title(
    f"Response for cell {target_ind} ({meta[meta['idx'] == target_ind]['Merge Class'].values[0]})"
)

stashfig(f"{target_ind}-response-hist" + basename)
