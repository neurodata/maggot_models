# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.visualization import remove_shared_ax

from scipy.cluster.hierarchy import linkage
from scipy.integrate import tplquad
from scipy.spatial.distance import squareform
from scipy.special import comb
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances

import pymaid
from graspy.utils import is_symmetric, pass_to_ranks, symmetrize
from hyppo.ksample import KSample
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    draw_leaf_dendrogram,
    get_mid_map,
    gridmap,
    matrixplot,
    plot_single_dendrogram,
    remove_axis,
    remove_spines,
    set_axes_equal,
    stacked_barplot,
)


np.random.seed(8888)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# params

level = 7
class_key = f"lvl{level}_labels"

metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "color_iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.3-BDP-omni-clust"


# load data
pair_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
pair_meta["lvl0_labels"] = pair_meta["lvl0_labels"].astype(str)
pair_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
pair_adj = pair_adj.values
mg = MetaGraph(pair_adj, pair_meta)
meta = mg.meta


def sort_mg(mg, level_names):
    meta = mg.meta
    sort_class = level_names + ["merge_class"]
    class_order = ["sf"]
    total_sort_by = []
    for sc in sort_class:
        for co in class_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    mg = mg.sort_values(total_sort_by, ascending=False)
    return mg


level = 7
lowest_level = level
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
mg = sort_mg(mg, level_names)


fig, axs = plt.subplots(
    1,
    2,
    figsize=(20, 15),
    sharey=True,
    gridspec_kw=dict(width_ratios=[0.25, 0.75], wspace=0),
)
mid_map = draw_leaf_dendrogram(
    mg.meta, axs[0], lowest_level=lowest_level, draw_labels=False
)
key_order = list(mid_map.keys())


compartment = "axon"
direction = "presynaptic"
foldername = "160.1-BDP-morpho-dcorr"
n_sub = 48
filename = f"test-stats-lvl={level}-compartment={compartment}-direction={direction }-method=subsample-n_sub={n_sub}-max_samp=500"
stat_df = readcsv(filename, foldername=foldername, index_col=0)
sym_vals = symmetrize(stat_df.values, method="triu")
stat_df = pd.DataFrame(data=sym_vals, index=stat_df.index, columns=stat_df.index)
ordered_stat_df = stat_df.loc[key_order, key_order]
sns.set_context("talk")
sns.heatmap(ordered_stat_df, ax=axs[1], cbar=False, cmap="RdBu_r", center=0)
axs[1].invert_yaxis()
axs[1].invert_xaxis()
axs[1].set_xticklabels([])

remove_shared_ax(axs[0])
remove_shared_ax(axs[1])
axs[1].set_yticks(np.arange(len(key_order)) + 0.5)
axs[1].set_yticklabels(key_order)
axs[1].yaxis.tick_right()
plt.tick_params(labelright="on", rotation=0, color="grey", labelsize=8)
axs[1].set_xticks([])
axs[1].set_title(
    f"DCorr test statistic, compartment = {compartment}, direction = {direction}"
)
# plt.tight_layout()
basename = f"-test-stats-lvl={level}-compartment={compartment}-direction={direction}"
stashfig("dcorr-heatmap-bar-dendrogram" + basename)

# %% [markdown]
# ##
sym_vals[~np.isfinite(sym_vals)] = 1
pdist = squareform(sym_vals)
Z = linkage(pdist, method="average", metric="euclidean")

cg = sns.clustermap(
    1 - sym_vals,
    row_linkage=Z,
    col_linkage=Z,
    xticklabels=False,
    yticklabels=False,
    cmap="Reds",
    vmin=0,
    # center=0,
    cbar_pos=None,
)
inds = cg.dendrogram_col.reordered_ind
stashfig("test-stat-clustered")

from src.visualization import adjplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot_mat = 1 - sym_vals.copy()
plot_mat = plot_mat[np.ix_(inds, inds)]
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(plot_mat, cbar=False, ax=ax)

divider = make_axes_locatable(ax)
top_ax = divider.append_axes("top", size="5%", pad=0, sharex=ax)
left_ax = divider.append_axes("left", size="5%", pad=0, sharey=ax)
remove_shared_ax(top_ax)
remove_shared_ax(left_ax)

mids = np.arange(len(plot_mat)) + 0.5

labels = stat_df.index.values.copy()
labels = labels[inds]


def calc_bar_params(sizes, label, mid, palette=None):
    if palette is None:
        palette = CLASS_COLOR_DICT
    heights = sizes.loc[label]
    n_in_bar = heights.sum()
    offset = mid - n_in_bar / 2
    starts = heights.cumsum() - heights + offset
    colors = np.vectorize(palette.get)(heights.index)
    return heights, starts, colors


def plot_bar(meta, mid, ax, orientation="horizontal"):
    if orientation == "horizontal":
        method = ax.barh
        ax.xaxis.set_visible(False)
        remove_spines(ax)
    elif orientation == "vertical":
        method = ax.bar
        ax.yaxis.set_visible(False)
        remove_spines(ax)
    sizes = meta.groupby("merge_class").size()
    sizes /= sizes.sum()
    starts = sizes.cumsum() - sizes
    colors = np.vectorize(CLASS_COLOR_DICT.get)(starts.index)
    for i in range(len(sizes)):
        method(mid, sizes[i], 0.7, starts[i], color=colors[i])


for i, label in enumerate(labels):
    temp_meta = meta[meta[f"lvl{level}_labels"] == label]
    plot_bar(temp_meta, mids[i], left_ax)
    plot_bar(temp_meta, mids[i], top_ax, orientation="vertical")


top_dend_ax = divider.append_axes("top", size="15%", pad=0, sharex=ax)
left_dend_ax = divider.append_axes("left", size="15%", pad=0, sharey=ax)
remove_shared_ax(top_dend_ax)
remove_shared_ax(left_dend_ax)


from scipy.cluster.hierarchy import dendrogram
from matplotlib.collections import LineCollection

sort_sym_vals = sym_vals[np.ix_(inds, inds)]


def draw_dendrogram(
    X, orientation="horizontal", ax=None, linewidth=1, method="average"
):
    pdist = squareform(X)
    Z = linkage(pdist, method=method)
    R = dendrogram(Z, no_plot=True)
    if orientation == "horizontal":
        dcoord = R["icoord"]
        icoord = R["dcoord"]
    elif orientation == "vertical":
        dcoord = R["dcoord"]
        icoord = R["icoord"]
    coords = zip(icoord, dcoord)
    tree_kws = {"linewidths": linewidth, "colors": ".2"}
    lines = LineCollection([list(zip(x, y)) for x, y in coords], **tree_kws)
    ax.add_collection(lines)
    number_of_leaves = len(X)
    if orientation == "horizontal":
        max_coord = max(map(max, icoord))
        ax.set_ylim(number_of_leaves * 10, 0)  # reversed for y
        ax.set_xlim(max_coord * 1.05, 0)
    elif orientation == "vertical":
        max_coord = max(map(max, dcoord))
        ax.set_xlim(0, number_of_leaves * 10)
        ax.set_ylim(0, max_coord * 1.05)
    remove_spines(ax)


draw_dendrogram(sort_sym_vals, orientation="horizontal", ax=left_dend_ax)
draw_dendrogram(sort_sym_vals, orientation="vertical", ax=top_dend_ax)
stashfig("test_stat_clustered_bars" + basename)

pdist = squareform(sym_vals)
Z = linkage(pdist, method="average")
from scipy.cluster.hierarchy import fcluster

flat_labels = fcluster(Z, 0.1, criterion="distance")
cluster_labels = stat_df.index.values.copy()


# %% [markdown]
# ##

# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)


def filter_connectors(connectors, ids, direction, compartment):
    label_connectors = connectors[connectors[f"{direction}_to"].isin(ids)]
    label_connectors = label_connectors[
        label_connectors[f"{direction}_type"] == compartment
    ]
    label_connectors = label_connectors[
        ~label_connectors["connector_id"].duplicated(keep="first")
    ]
    return label_connectors


from src.visualization import plot_3view
from src.pymaid import start_instance

start_instance()
skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)

for fl in np.unique(flat_labels):
    label_names = cluster_labels[flat_labels == fl]
    skids = meta[meta[f"lvl{level}_labels"].isin(label_names)].index
    temp_connectors = filter_connectors(connectors, skids, direction, compartment)
    if len(temp_connectors) > 5000:
        inds = np.random.choice(len(temp_connectors), size=5000, replace=False)
        temp_connectors = temp_connectors.iloc[inds]
    fig = plt.figure(figsize=(15, 5))
    axs = np.empty((3), dtype="O")
    gs = plt.GridSpec(1, 3, figure=fig, wspace=0, hspace=0)
    for j in range(3):
        ax = fig.add_subplot(gs[j], projection="3d")
        axs[j] = ax
        ax.axis("off")
    plot_3view(
        temp_connectors,
        axs,
        palette=skeleton_color_dict,
        label_by=f"{direction}_to",
        alpha=0.6,
        s=2,
    )
    stashfig(f"{fl}_labeled" + basename)

