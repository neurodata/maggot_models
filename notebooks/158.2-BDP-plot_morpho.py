# %% [markdown]
# ##
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import tplquad
from scipy.stats import gaussian_kde

import pymaid
from graspy.utils import pass_to_ranks
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    get_mid_map,
    gridmap,
    matrixplot,
    plot_3view,
    plot_single_dendrogram,
    remove_axis,
    remove_spines,
    set_axes_equal,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


def add_subplot(row, col, axs, projection=None):

    return ax


# params

level = 3
key = f"lvl{level}_labels"

metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "color_iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.2-BDP-omni-clust"


# load data
pair_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
pair_meta["lvl0_labels"] = pair_meta["lvl0_labels"].astype(str)
pair_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
pair_adj = pair_adj.values
mg = MetaGraph(pair_adj, pair_meta)
meta = mg.meta

level_names = [f"lvl{i}_labels" for i in range(level + 1)]


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


def calc_ego_connectivity(adj, meta, label, axis=0):
    this_inds = meta[meta[key] == label]["inds"].values
    uni_cat = meta[key].unique()
    connect_mat = []
    for other_label in uni_cat:
        other_inds = meta[meta[key] == other_label]["inds"].values
        if axis == 0:
            sum_vec = adj[np.ix_(other_inds, this_inds)].sum(axis=axis)
        elif axis == 1:
            sum_vec = adj[np.ix_(this_inds, other_inds)].sum(axis=axis)
        connect_mat.append(sum_vec)
    return np.array(connect_mat)


mg = sort_mg(mg, level_names)
meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj


start_instance()

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)


# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)


# %% [markdown]
# ##

# plot params
scale = 5
n_col = 10
n_row = 3
margin = 0.01
gap = 0.02

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "grey",
    "ytick.color": "dimgrey",
    "xtick.color": "dimgrey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
}
for k, val in rc_dict.items():
    mpl.rcParams[k] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

for label in meta[key].unique()[::-1]:

    fig = plt.figure(figsize=(n_col * scale, n_row * scale))

    # for the morphology plotting
    morpho_gs = plt.GridSpec(
        n_row,
        3,
        figure=fig,
        wspace=0,
        hspace=0,
        left=margin,
        right=margin + 3 / n_col,
        top=1 - margin,
        bottom=margin,
    )

    # plot the neurons and synapses
    morpho_axs = np.empty((3, 3), dtype="O")

    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(morpho_gs[i, j], projection="3d")
            morpho_axs[i, j] = ax
            ax.axis("off")

    temp_meta = meta[meta[key] == label]
    label1_ids = temp_meta.index.values
    label1_ids = [int(i) for i in label1_ids]
    label1_inputs = connectors[connectors["postsynaptic_to"].isin(label1_ids)]
    label1_outputs = connectors[connectors["presynaptic_to"].isin(label1_ids)]
    label1_outputs = label1_outputs[
        ~label1_outputs["connector_id"].duplicated(keep="first")
    ]

    axs = morpho_axs
    plot_3view(
        label1_ids, axs[0, :], palette=skeleton_color_dict, row_title="Skeletons"
    )
    plot_3view(
        label1_inputs,
        axs[1, :],
        palette=skeleton_color_dict,
        label_by="postsynaptic_to",
        alpha=0.6,
        s=2,
        row_title="Input",
    )
    plot_3view(
        label1_outputs,
        axs[2, :],
        palette=skeleton_color_dict,
        label_by="presynaptic_to",
        alpha=0.6,
        s=2,
        row_title="Output",
    )

    mid_gs = plt.GridSpec(
        n_row,
        4,
        figure=fig,
        left=margin + 3 / n_col + gap,
        right=margin + 5 / n_col + gap,
        bottom=margin,
        top=1 - margin,
    )

    ax = fig.add_subplot(mid_gs[0, 1:])

    cat = temp_meta[f"lvl{level}_labels_side"].values
    subcat = temp_meta["merge_class"].values
    stacked_barplot(
        cat,
        subcat,
        ax=ax,
        color_dict=CLASS_COLOR_DICT,
        plot_names=True,
        text_color="dimgrey",
        bar_height=0.2,
    )
    ax.get_legend().remove()
    ax.set_title("Class membership", y=0.85)
    ax.set_ylim(-0.5, 2)
    subgraph_inds = temp_meta["inds"].values
    subgraph_adj = adj[np.ix_(subgraph_inds, subgraph_inds)]
    ax = fig.add_subplot(mid_gs[1:, :])
    _, _, top, _ = adjplot(
        pass_to_ranks(subgraph_adj),
        plot_type="heatmap",
        cbar=False,
        ax=ax,
        meta=temp_meta,
        item_order=["merge_class", "sf"],
        colors="merge_class",
        palette=CLASS_COLOR_DICT,
    )
    top.set_title("Intra-cluster connectivity")

    dend_gs = plt.GridSpec(
        1,
        4,
        figure=fig,
        left=margin + 5 / n_col + 2 * gap,
        right=1 - margin,
        bottom=margin,
        top=1 - margin,
        wspace=0.02,
        hspace=0,
        width_ratios=[0.3, 1, 1, 1],
    )

    # barplot
    ax = fig.add_subplot(dend_gs[0, 1])
    uni_cat = meta[key].unique()
    # hack to make the figures line up
    divider = make_axes_locatable(ax)
    tick_ax = divider.append_axes("top", size="3%", pad=0)
    remove_axis(tick_ax)
    tick_ax.set_title("Source/target cluster")
    # plot the actual bars
    stacked_barplot(
        meta[key],
        meta["merge_class"],
        category_order=uni_cat,
        subcategory_order=meta["merge_class"].unique(),
        ax=ax,
        color_dict=CLASS_COLOR_DICT,
        text_color="dimgrey",
    )
    ax.get_legend().remove()
    ax.set_ylim(-0.5, len(uni_cat) - 0.5)
    # make the current cluster bold
    for t in ax.get_yticklabels():
        text = t.get_text()
        text = text.split(" ")[0]
        t.set_fontsize(12)
        if text == label:
            t.set_color("black")

    this_inds = meta[meta[key] == label]["inds"].values

    ax = fig.add_subplot(dend_gs[0, 2])

    input_mat = calc_ego_connectivity(adj, meta, label, axis=0)

    _, _, top, _ = matrixplot(
        pass_to_ranks(input_mat[::-1]),
        col_meta=temp_meta,
        col_item_order=["merge_class", "sf"],
        col_sort_class=["merge_class"],
        col_ticks=False,
        col_colors="merge_class",
        col_palette=CLASS_COLOR_DICT,
        cbar=False,
        row_ticks=False,
        ax=ax,
    )
    ax.set_yticks([])
    top.set_title("Input")

    # output
    ax = fig.add_subplot(dend_gs[0, 3])

    output_mat = calc_ego_connectivity(adj, meta, label, axis=1)

    _, _, top, _ = matrixplot(
        pass_to_ranks(output_mat[::-1]),
        col_meta=temp_meta,
        col_item_order=["merge_class", "sf"],
        col_sort_class=["merge_class"],
        col_ticks=False,
        col_colors="merge_class",
        col_palette=CLASS_COLOR_DICT,
        row_ticks=False,
        cbar=False,
        ax=ax,
    )
    ax.set_yticks([])
    top.set_title("Output")

    fig.suptitle("Cluster " + label, y=1.03, color="black", fontsize=30)
    stashfig(f"lvl{level}_{label}" + basename)
