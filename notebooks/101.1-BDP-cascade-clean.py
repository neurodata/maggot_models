# %% [markdown]
# #
import itertools
import os
import time
from itertools import chain

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, Node, RenderTree
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    barplot_text,
    draw_networkx_nice,
    draw_separators,
    matrixplot,
    remove_shared_ax,
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

plot_examples = False
plot_embed = False
plot_full_mat = False
graph_type = "Gad"
threshold = 0
weight = "weight"
mg = load_metagraph(graph_type, VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight=weight,
)
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")

# TODO update this with the mixed groups
# TODO make these functional for selecting proper paths

inout = "sensory_to_out"
if inout == "sensory_to_out":
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
    from_groups = [
        ("sens-ORN",),
        ("sens-photoRh5", "sens-photoRh6"),
        ("sens-MN",),
        ("sens-thermo",),
        ("sens-vtd",),
        ("sens-AN",),
    ]
    from_group_names = ["Odor", "Photo", "MN", "Temp", "VTD", "AN"]

if inout == "out_to_sensory":
    from_groups = [
        ("motor-mAN", "motormVAN", "motor-mPaN"),
        ("O_dSEZ", "O_dVNC;O_dSEZ", "O_dSEZ;CN", "LHN;O_dSEZ"),
        ("O_dVNC", "O_dVNC;CN", "O_RG;O_dVNC", "O_dVNC;O_dSEZ"),
        ("O_RG", "O_RG-IPC", "O_RG-ITP", "O_RG-CA-LP", "O_RG;O_dVNC"),
        ("O_dUnk",),
    ]
    from_group_names = ["Motor", "SEZ", "VNC", "RG", "dUnk"]
    out_classes = [
        "sens-ORN",
        "sens-photoRh5",
        "sens-photoRh6",
        "sens-MN",
        "sens-thermo",
        "sens-vtd",
        "sens-AN",
    ]

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
if path_type == "cascade":
    p = 0.005
    # p = 0.05
    not_probs = (
        1 - p
    ) ** adj  # probability of none of the synapses causing postsynaptic
    probs = 1 - not_probs  # probability of ANY of the synapses firing onto next
elif path_type == "fancy-cascade":
    alpha = 0.5
    flat = np.full(adj.shape, alpha)
    deg = meta["dendrite_input"].values
    deg[deg == 0] = 1
    flat = flat / deg[None, :]
    not_probs = np.power((1 - flat), adj)
    probs = 1 - not_probs

#%%
seed = 8888
max_depth = 10
n_bins = 10
n_sims = 100
method = "tree"
normalize_n_source = False


basename = f"-{graph_type}-p{p}-pt{path_type}-b{n_bins}-n{n_sims}-m{method}"
basename += f"-norm{normalize_n_source}"
basename += f"-{inout}"


np.random.seed(seed)
if method == "tree":
    seeds = np.random.choice(int(1e8), size=len(from_inds), replace=False)
    outs = Parallel(n_jobs=1, verbose=10)(
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
if plot_full_mat:
    shape = log_mat.shape
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
        row_colors=CLASS_COLOR_DICT,
        row_meta=row_df,
        row_sort_class=["to_class"],
        plot_type="heatmap",
        sizes=(0.5, 0.5),
        tick_rot=45,
    )
    stashfig("log-full-heat" + basename)

# %% [markdown]
# # Screeplots

if plot_embed:
    screeplot(hist_mat.astype(float), title="Raw hist mat (full)")
    stashfig("scree-raw-mat" + basename)
    screeplot(log_mat, title="Log hist mat (full)")
    stashfig("scree-log-mat" + basename)

# %% [markdown]
# # Pairplots
if plot_embed:
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
        loadings,
        labels=from_class.values,
        height=5,
        title="Source class embedding (log)",
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
        loadings,
        labels=from_class.values,
        height=5,
        title="Source class embedding (raw)",
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

# %% [markdown]
# #
if plot_embed:
    pca = PCA(n_components=6)
    embed = pca.fit_transform(log_collapsed_hist)
    loadings = pca.components_.T
    pg = pairplot(
        embed,
        labels=to_class.values,
        palette=CLASS_COLOR_DICT,
        height=5,
        title="Collapsed node response embedding (log)",
    )
    pg._legend.remove()
    stashfig("coll-node-pca-log" + basename)
    pg = pairplot(
        loadings,
        labels=collapsed_col_df["from_class"].values,
        height=5,
        title="Collapsed source class embedding (log)",
    )
    stashfig("coll-source-pca-log" + basename)

    pca = PCA(n_components=6)
    embed = pca.fit_transform(collapsed_hist.astype(float))
    loadings = pca.components_.T
    pg = pairplot(
        embed,
        labels=to_class.values,
        palette=CLASS_COLOR_DICT,
        height=5,
        title="Collapsed node response embedding (raw)",
    )
    pg._legend.remove()
    stashfig("coll-node-pca-log" + basename)
    pg = pairplot(
        loadings,
        labels=collapsed_col_df["from_class"].values,
        height=5,
        title="Collapsed source class embedding (raw)",
    )
    stashfig("coll-source-pca-log" + basename)

# %% [markdown]
# # Compute mean visit over all sources, for plotting
def mean_visit(row):
    n_groups = len(row) // n_bins
    s = 0
    for i in range(n_groups):
        group = row[i * n_bins : (i + 1) * n_bins]
        for j, val in enumerate(group):
            s += j * val
    s /= row.sum()
    return s


visits = []
for r in collapsed_hist:
    mv = mean_visit(r)
    visits.append(mv)
visits = np.array(visits)
visits[np.isnan(visits)] = n_bins + 1
row_df["visit_order"] = visits
mean_visit_order = row_df.groupby(["to_class"])["visit_order"].mean()
row_df["group_visit_order"] = row_df["to_class"].map(mean_visit_order)
row_df["n_visit"] = collapsed_hist.sum(axis=1)


# %% [markdown]
# #
sns.set_context("talk", font_scale=1)
gridline_kws = dict(color="grey", linestyle="--", alpha=0.7, linewidth=0.3)

fig, ax = plt.subplots(1, 1, figsize=(25, 15))
ax, divider, top_cax, left_cax = matrixplot(
    log_collapsed_hist.T,
    ax=ax,
    row_meta=collapsed_col_df,
    row_sort_class=["from_class"],
    col_meta=row_df,
    col_sort_class=["to_class"],
    col_colors=CLASS_COLOR_DICT,
    col_class_order="group_visit_order",
    col_item_order=["visit_order"],
    plot_type="heatmap",
    tick_rot=45,
    col_ticks=False,
    gridline_kws=gridline_kws,
)
cax = divider.append_axes("right", size="1%", pad=0.02, sharey=ax)
remove_shared_ax(cax)
sns.heatmap(
    collapsed_col_df["order"][:, None], ax=cax, cbar=False, cmap="RdBu", center=0
)
cax.set_xticks([])
cax.set_yticks([])
cax.set_ylabel(r"Hops $\to$", rotation=-90, ha="center", va="center", labelpad=20)
cax.yaxis.set_label_position("right")
top_cax.set_yticks([0.5])
top_cax.set_yticklabels(["Class"], va="center")
ax.set_xlabel("Neuron")
ax.set_ylabel("Source class")
stashfig("collapsed-log-heat-transpose" + basename, dpi=200)

fig, ax = plt.subplots(1, 1, figsize=(25, 15))
ax, divider, top_cax, left_cax = matrixplot(
    log_collapsed_hist.T,
    ax=ax,
    row_meta=collapsed_col_df,
    row_sort_class=["from_class"],
    col_meta=row_df,
    col_sort_class=["to_class"],
    col_colors=CLASS_COLOR_DICT,
    col_class_order="group_visit_order",
    col_item_order=["visit_order"],
    plot_type="heatmap",
    tick_rot=45,
    col_ticks=True,
    gridline_kws=gridline_kws,
)
cax = divider.append_axes("right", size="1%", pad=0.02, sharey=ax)
remove_shared_ax(cax)
sns.heatmap(
    collapsed_col_df["order"][:, None], ax=cax, cbar=False, cmap="RdBu", center=0
)
cax.set_xticks([])
cax.set_yticks([])
cax.set_ylabel(r"Hops $\to$", rotation=-90, ha="center", va="center", labelpad=20)
cax.yaxis.set_label_position("right")
top_cax.set_yticks([0.5])
top_cax.set_yticklabels(["Class"], va="center")
ax.set_xlabel("Neuron")
ax.set_ylabel("Source class")
stashfig("collapsed-log-heat-transpose-labeled" + basename, dpi=200)

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
    data=sort_log_collapsed_hist.T,
    col_cluster=True,
    row_cluster=False,
    col_colors=colors,
    cmap="RdBu_r",
    center=0,
    cbar_pos=None,
    method=linkage,
    metric=metric,
)
ax = cg.ax_heatmap
draw_separators(
    ax,
    ax_type="y",
    sort_meta=sort_collapsed_col_df,
    sort_class=["from_class"],
    tick_rot=0,
)
ax.xaxis.set_ticks([])
# ax.set_ylabel(r"Visits over time $\to$")
ax.set_xlabel("Neuron")
ax.yaxis.tick_left()
# ax.set_yticklabels(ax.get_yticklabels(), ha="left")
stashfig("collapsed-log-clustermap" + basename)
# stashfig("collapsed-log-clustermap" + basename, fmt="pdf")


# %% [markdown]
# # Do some plotting for illustration only


if plot_examples:
    sns.set_context("talk")
    sns.set_palette("Set1")
    examples = [742, 605, 743, 2282, 596, 2367, 1690, 2313]
    for target_ind in examples:
        row = collapsed_hist[target_ind, :]
        perm_inds, sort_col_df = sort_meta(collapsed_col_df, sort_class=["from_class"])
        sort_row = row[perm_inds]

        fig, ax = plt.subplots(1, 1)
        xs = np.arange(len(sort_row)) + 0.5
        divider = make_axes_locatable(ax)
        bot_cax = divider.append_axes("bottom", size="3%", pad=0.02, sharex=ax)
        remove_shared_ax(bot_cax)

        ax.bar(x=xs, height=sort_row, width=0.8)
        draw_separators(
            ax, sort_meta=sort_col_df, sort_class=["from_class"], tick_rot=0
        )
        ax.set_xlim(0, len(xs))
        ax.set_ylabel("# hits @ time")

        sns.heatmap(
            collapsed_col_df["order"][None, :],
            ax=bot_cax,
            cbar=False,
            cmap="RdBu",
            center=0,
        )
        bot_cax.set_xticks([])
        bot_cax.set_yticks([])
        bot_cax.set_xlabel(r"Hops $\to$", x=0.1, ha="left", labelpad=-22)
        bot_cax.set_xticks([20.5, 24.5, 28.5])
        bot_cax.set_xticklabels([1, 5, 9], rotation=0)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        target_skid = meta.iloc[target_ind, :].name
        ax.set_title(
            f"Response for cell {target_skid} ({meta[meta['idx'] == target_ind]['Merge Class'].values[0]})"
        )

        stashfig(f"{target_skid}-response-hist" + basename)

