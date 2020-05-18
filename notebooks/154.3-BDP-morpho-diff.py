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
    savefig(name, foldername=FNAME, save_on=True, **kws)


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

split_file = "maggot_models/data/raw/brain_acc_split_nodes.csv"
split_df = pd.read_csv(split_file)
split_df.set_index("skeleton", inplace=True)
splits = split_df.squeeze()

# params
label1 = "0-1-1-1-0-1-1"
label2 = "0-1-1-1-0-0-1"
level = 6
key = f"lvl{level}_labels"
volume_names = ["PS_Neuropil_manual"]

from src.visualization import stacked_barplot

stacked_barplot(
    meta[key].values, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT
)

# get info
label1_ids = meta[meta[key] == label1].index.values
label1_ids = [int(i) for i in label1_ids]
label2_ids = meta[meta[key] == label2].index.values
label2_ids = [int(i) for i in label2_ids]

# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)

label1_inputs = connectors[connectors["postsynaptic_to"].isin(label1_ids)]
label1_outputs = connectors[connectors["presynaptic_to"].isin(label1_ids)]

label2_inputs = connectors[connectors["postsynaptic_to"].isin(label2_ids)]
label2_outputs = connectors[connectors["presynaptic_to"].isin(label2_ids)]

# %% [markdown]
# ##
from matplotlib.patches import Circle

show_plot = True
if show_plot:
    show_neurons = True
    scale = 5
    n_col = 2 * 3
    n_row = 3

    sns.set_context("talk", font_scale=1.5)
    fig = plt.figure(figsize=(n_col * scale, n_row * scale))
    # fig.suptitle(label, y=0.93)
    gs = plt.GridSpec(n_row, n_col, figure=fig, wspace=0, hspace=0)
    axs = np.empty((n_row, n_col), dtype="O")

    connection_types = ["axon", "dendrite", "unsplittable"]
    pal = sns.color_palette("deep", 5)
    colors = [pal[3], pal[0], pal[4]]  # red, blue, purple
    connection_colors = dict(zip(connection_types, colors))

    views = ["front", "side", "top"]
    view_params = [
        dict(azim=-90, elev=0, dist=5),
        dict(azim=0, elev=0, dist=5),
        dict(azim=-90, elev=90, dist=5),
    ]
    view_dict = dict(zip(views, view_params))

    volumes = [pymaid.get_volume(v) for v in volume_names]

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
                    scatter_kws=dict(color=connection_colors[ct], label=ct),
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

    for i in range(3):
        for j in range(n_col):
            ax = add_subplot(i, j, projection="3d")
            ax.axis("off")

    col1_axs = axs[:, :3]
    plot_neuron_morphology(
        label1_ids, label1_inputs, label1_outputs, col1_axs, row_label=True
    )
    col1_axs[0, 1].set_title(label1)

    legend_elements = []
    for ct in connection_types:
        p = Circle(
            (-1, -1), facecolor=connection_colors[ct], label=ct, linewidth=0, radius=1
        )
        legend_elements.append(p)
    col1_axs[2, 0].legend(
        handles=legend_elements, bbox_to_anchor=(0, 0), loc="upper left"
    )

    col2_axs = axs[:, 3:]
    plot_neuron_morphology(
        label2_ids, label2_inputs, label2_outputs, col2_axs, row_label=True
    )
    col2_axs[0, 1].set_title(label2)

    plt.tight_layout()

    stashfig(f"morpho-compare-{label1}_{label2}")


# %% [markdown]
# # ##

from hyppo.ksample import KSample


def split_in_out(inputs, outputs):
    in_axon = inputs[inputs["postsynaptic_type"] == "axon"][["x", "y", "z"]].values
    in_dend = inputs[inputs["postsynaptic_type"] == "dendrite"][["x", "y", "z"]].values
    out_axon = outputs[outputs["presynaptic_type"] == "axon"][["x", "y", "z"]].values
    out_dend = outputs[outputs["presynaptic_type"] == "dendrite"][
        ["x", "y", "z"]
    ].values
    return (in_axon, in_dend, out_axon, out_dend)


names = ["Axon input", "Dendrite input", "Axon output", "Dendrite output"]
syn_groups1 = split_in_out(label1_inputs, label1_outputs)
syn_groups2 = split_in_out(label2_inputs, label2_outputs)

result_df = pd.DataFrame(
    index=names, columns=["pval", "stat", "n_sample1", "n_sample2"], dtype="float64"
)

run_test = True
print("Running dcorr...")
ksamp = KSample("Dcorr")
for i, n in enumerate(names):
    print(n)
    data1 = syn_groups1[i]
    print(data1.shape)
    data2 = syn_groups2[i]
    print(data2.shape)
    if run_test:
        stat, pval = ksamp.test(data1, data2, auto=True)
        result_df.loc[n, "pval"] = pval
    else:
        stat = ksamp._statistic(data1, data2)
    result_df.loc[n, "stat"] = stat
    result_df.loc[n, "n_sample1"] = len(data1)
    result_df.loc[n, "n_sample2"] = len(data2)
    print()

print(result_df)

stashcsv(result_df, f"{label1}_{label2}-test-results" + basename)
# plot_vars = np.array(["x", "y", "z"])


# def plot_connectors(data, Z, x, y, ax, mins, maxs):
#     sns.scatterplot(
#         data=data,
#         y=plot_vars[y],
#         x=plot_vars[x],
#         s=3,
#         alpha=0.05,
#         ax=ax,
#         linewidth=0,
#         color="black",
#     )
#     unused = np.setdiff1d([0, 1, 2], [x, y])[0]
#     projection = Z.sum(axis=unused)
#     if x > y:
#         projection = projection.T
#     ax.imshow(
#         np.rot90(projection),  # integrate out the unused dim
#         cmap=plt.cm.Reds,
#         extent=[mins[x], maxs[x], mins[y], maxs[y]],
#         vmin=0,
#     )
#     ax.set_xlim([mins[x], maxs[x]])
#     ax.set_ylim([maxs[y], mins[y]])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     remove_spines(ax)


# def get_joint_bounds(datas):
#     all_mins = []
#     all_maxs = []
#     for df in datas:
#         all_mins.append(df[plot_vars].min().values)
#         all_maxs.append(df[plot_vars].max().values)
#     all_mins = np.array(all_mins)
#     all_maxs = np.array(all_maxs)
#     mins = np.min(all_mins, axis=0)
#     maxs = np.max(all_maxs, axis=0)
#     scale = 0.2
#     ranges = maxs - mins
#     mins = mins - ranges * scale
#     maxs = maxs + ranges * scale
#     return mins, maxs


# def plot_kdes(datas, row=None):
#     mins, maxs = get_joint_bounds(datas)

#     # compute KDEs
#     kernels = []
#     for df in datas:
#         data_mat = df[plot_vars].values
#         kernel = gaussian_kde(data_mat.T)
#         kernels.append(kernel)

#     # evaluate KDEs
#     X, Y, Z = np.mgrid[
#         mins[0] : maxs[0] : 100j, mins[1] : maxs[1] : 100j, mins[2] : maxs[2] : 100j
#     ]
#     positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
#     Zs = []
#     for k in kernels:
#         Z = np.reshape(k(positions).T, X.shape)
#         Zs.append(Z)

#     dim_pairs = [(0, 1), (2, 1), (0, 2)]

#     # axes_names = [
#     #     r"$\leftarrow$ L $\quad$ R $\rightarrow$",
#     #     r"$\leftarrow$ D $\quad$ V $\rightarrow$",
#     #     r"$\leftarrow$ A $\quad$ P $\rightarrow$",
#     # ]

#     for i in range(2):
#         for j, dims in enumerate(dim_pairs):
#             ax = axs[row, j + i * 3]
#             x = dims[0]
#             y = dims[1]
#             plot_connectors(datas[i], Zs[i], x, y, ax, mins, maxs)
#             if j == 2:
#                 ax.invert_yaxis()
#         # ax.set_xlabel(axes_names[0])
#         # ax.set_ylabel(axes_names[1])


# # print("Input KDEs...")
# # df1 = label1_inputs
# # df2 = label2_inputs
# # datas = [df1, df2]
# # plot_kdes(datas, row=3)
# # axs[3, 0].set_ylabel("Input KDEs", color="grey")

# # print("Output KDEs...")
# # df1 = label1_outputs
# # df2 = label2_outputs
# # datas = [df1, df2]
# # plot_kdes(datas, row=4)
# # axs[4, 0].set_ylabel("Output KDEs", color="grey")

