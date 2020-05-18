# %% [markdown]
# ##
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.integrate import tplquad
from scipy.stats import gaussian_kde

import pymaid
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance


from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
    remove_axis,
    remove_spines,
    set_axes_equal,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

from src.io import readcsv
from src.graph import MetaGraph


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# mg = load_metagraph("G")
# mg = preprocess(
#     mg,
#     threshold=0,
#     sym_threshold=False,
#     remove_pdiff=True,
#     binarize=False,
#     weight="weight",
# )
# meta = mg.meta

metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}-G"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.1-BDP-omni-clust"

# load data
pair_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
pair_meta["lvl0_labels"] = pair_meta["lvl0_labels"].astype(str)
pair_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
pair_mg = MetaGraph(pair_adj.values, pair_meta)
meta = pair_mg.meta

start_instance()

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)


# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)

from matplotlib.patches import Circle

# params

level = 6
key = f"lvl{level}_labels"
volume_names = ["PS_Neuropil_manual"]

show_neurons = True
scale = 5
n_col = 1 * 3
n_row = 3


# %% [markdown]
# ##


sns.set_context("talk", font_scale=1.5)


connection_types = ["axon", "dendrite", "unsplit"]
pal = sns.color_palette("deep", 5)
colors = [pal[3], pal[0], pal[4]]  # red, blue, purple
connection_colors = dict(zip(connection_types, colors))




def set_view_params(ax, azim=-90, elev=0, dist=5):
    ax.azim = azim
    ax.elev = elev
    ax.dist = dist
    set_axes_equal(ax)


def plot_volumes(ax):
    pymaid.plot2d(volumes, ax=ax, method="3d")
    for c in ax.collections:
        if isinstance(c, Poly3DCollection):
            c.set_alpha(0.03)


def add_subplot(row, col, projection=None):
    ax = fig.add_subplot(gs[row, col], projection=projection)
    axs[row, col] = ax
    return ax


def plot_neuron_morphology(ids, inputs, outputs, axs, row_label=True):
    # plot neuron skeletons
    row = 0
    for i, view in enumerate(views):
        ax = axs[row, i]
        if show_neurons:
            pymaid.plot2d(
                ids, color=skeleton_color_dict, ax=ax, connectors=False, method="3d"
            )
        plot_volumes(ax)
        set_view_params(ax, **view_dict[view])

    # plot inputs
    row = 1
    for i, view in enumerate(views):
        ax = axs[row, i]
        for j, ct in enumerate(connection_types):
            ct_inputs = inputs[inputs["postsynaptic_type"] == ct]
            connector_locs = ct_inputs[["x", "y", "z"]].values
            pymaid.plot2d(
                connector_locs,
                ax=ax,
                method="3d",
                scatter_kws=dict(color=connection_colors[ct]),
            )
        plot_volumes(ax)
        set_view_params(ax, **view_dict[view])

    # plot outputs
    row = 2
    for i, view in enumerate(views):
        ax = axs[row, i]
        for j, ct in enumerate(connection_types):
            ct_outputs = outputs[outputs["presynaptic_type"] == ct]
            connector_locs = ct_outputs[["x", "y", "z"]].values
            pymaid.plot2d(
                connector_locs,
                ax=ax,
                method="3d",
                scatter_kws=dict(color=connection_colors[ct], label=ct, s=2, alpha=0.5),
            )
            # if i == 0:
            #     ax.legend(bbox_to_anchor=(0, 1), loc="upper left")
        plot_volumes(ax)
        set_view_params(ax, **view_dict[view])

    if row_label:
        axs[0, 0].text2D(
            x=0,
            y=0.5,
            s="Skeletons",
            ha="right",
            va="center",
            color="grey",
            rotation=90,
            transform=axs[0, 0].transAxes,
        )
        axs[1, 0].text2D(
            x=0,
            y=0.5,
            s="Inputs",
            ha="right",
            va="center",
            color="grey",
            rotation=90,
            transform=axs[1, 0].transAxes,
        )
        axs[2, 0].text2D(
            x=0,
            y=0.5,
            s="Outputs",
            ha="right",
            va="center",
            color="grey",
            rotation=90,
            transform=axs[2, 0].transAxes,
        )


labels = np.unique(meta[f"lvl{level}_labels"])

for label in labels[10:]:
    # get info
    label1_ids = meta[meta[key] == label].index.values
    label1_ids = [int(i) for i in label1_ids]
    label1_inputs = connectors[connectors["postsynaptic_to"].isin(label1_ids)]
    label1_outputs = connectors[connectors["presynaptic_to"].isin(label1_ids)]

    fig = plt.figure(figsize=(n_col * scale, n_row * scale))
    # fig.suptitle(label, y=0.93)
    gs = plt.GridSpec(n_row, n_col, figure=fig, wspace=0, hspace=0)
    axs = np.empty((n_row, n_col), dtype="O")

    for i in range(3):
        for j in range(n_col):
            ax = add_subplot(i, j, projection="3d")
            ax.axis("off")

    col1_axs = axs[:, :]
    plot_neuron_morphology(
        label1_ids, label1_inputs, label1_outputs, col1_axs, row_label=True
    )
    col1_axs[0, 1].set_title(label)

    legend_elements = []
    for ct in connection_types:
        p = Circle(
            (-1, -1), facecolor=connection_colors[ct], label=ct, linewidth=0, radius=1
        )
        legend_elements.append(p)
    col1_axs[2, 0].legend(
        handles=legend_elements, bbox_to_anchor=(0, 0), loc="upper left"
    )

    plt.tight_layout()

    stashfig(f"morpho-lvl{level}_{label}")

