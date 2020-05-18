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


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


mg = load_metagraph("G")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

start_instance()

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)

split_file = "maggot_models/data/raw/brain_acc_split_nodes.csv"
split_df = pd.read_csv(split_file)
split_df.set_index("skeleton", inplace=True)
splits = split_df.squeeze()


def get_connectors(nl):
    connectors = pymaid.get_connectors(nl)
    connectors.set_index("connector_id", inplace=True)
    connectors.drop(
        [
            "confidence",
            "creation_time",
            "edition_time",
            "tags",
            "creator",
            "editor",
            "type",
        ],
        inplace=True,
        axis=1,
    )
    details = pymaid.get_connector_details(connectors.index.values)
    details.set_index("connector_id", inplace=True)
    connectors = pd.concat((connectors, details), ignore_index=False, axis=1)
    connectors.reset_index(inplace=True)
    return connectors


def set_view_params(ax, azim=-90, elev=0, dist=5):
    ax.azim = azim
    ax.elev = elev
    ax.dist = dist
    set_axes_equal(ax)


def get_in_out(ids):
    nl = pymaid.get_neurons(ids)
    connectors = get_connectors(nl)
    outputs = connectors[connectors["presynaptic_to"].isin(ids)].copy()
    # I hope this is a valid assumption?
    inputs = connectors[~connectors["presynaptic_to"].isin(ids)].copy()
    return inputs, outputs


def get_split_in_out(labels, splits):
    nl = pymaid.get_neurons(labels)

    all_axon_treenodes = []
    all_dend_treenodes = []
    for i, n in enumerate(nl):
        skid = int(n.skeleton_id)
        # order of output is axon, dendrite
        fragments = pymaid.cut_neuron(n, splits[skid])

        axon_treenodes = fragments[0].nodes.treenode_id.values
        dend_treenodes = fragments[1].nodes.treenode_id.values
        all_axon_treenodes.append(axon_treenodes)
        all_dend_treenodes.append(dend_treenodes)

    all_axon_treenodes = np.concatenate(all_axon_treenodes)
    all_dend_treenodes = np.concatenate(all_dend_treenodes)
    all_treenodes = np.concatenate((all_axon_treenodes, all_dend_treenodes))

    def filter_treenodes(nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        relevant_nodes = [node for node in nodes if node in all_treenodes]
        is_axon = np.isin(relevant_nodes, all_axon_treenodes)
        is_dend = np.isin(relevant_nodes, all_dend_treenodes)
        if is_axon.all():
            return "axon"
        elif is_dend.all():
            return "dend"
        elif (is_axon | is_dend).any():
            # afaik can only happen if synapse is polyadic and some inputs to group are
            # onto dendrites and some are onto axons
            return "mixed"
        else:
            return ""

    inputs, outputs = get_in_out(labels)

    inputs["postsynaptic_type"] = inputs["postsynaptic_to_node"].map(filter_treenodes)
    outputs["presynaptic_type"] = outputs["presynaptic_to_node"].map(filter_treenodes)
    return inputs, outputs


# params
label1 = "uPN"
label2 = "KC"
volume_names = ["PS_Neuropil_manual"]

# get info
label1_ids = meta[meta["class1"] == label1].index.values
label1_ids = [int(i) for i in label1_ids]
label1_inputs, label1_outputs = get_split_in_out(label1_ids, splits)
label2_ids = meta[meta["class1"] == label2].index.values
label2_ids = [int(i) for i in label2_ids]
label2_inputs, label2_outputs = get_split_in_out(label2_ids, splits)


# %% [markdown]
# ##
show_plot = False
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

    connection_types = ["axon", "dend", "mixed"]
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
                    scatter_kws=dict(color=connection_colors[ct]),
                )
            plot_volumes(ax)
            set_view_params(ax, **view_dict[view])

        if row_label:
            axs[0, 0].text2D(
                x=0,
                y=0.5,
                s="Skeletons",
                ha="center",
                va="bottom",
                color="grey",
                rotation=90,
                transform=axs[0, 0].transAxes,
            )
            axs[1, 0].text2D(
                x=0,
                y=0.5,
                s="Inputs",
                ha="center",
                va="bottom",
                color="grey",
                rotation=90,
                transform=axs[1, 0].transAxes,
            )
            axs[2, 0].text2D(
                x=0,
                y=0.5,
                s="Outputs",
                ha="center",
                va="bottom",
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

    col2_axs = axs[:, 3:]
    plot_neuron_morphology(
        label2_ids, label2_inputs, label2_outputs, col2_axs, row_label=True
    )

    plt.tight_layout()

    stashfig(f"morpho-compare-{label1}-{label2}")


# %% [markdown]
# # ##

from hyppo.ksample import KSample


def split_in_out(inputs, outputs):
    in_axon = inputs[inputs["postsynaptic_type"] == "axon"][["x", "y", "z"]].values
    in_dend = inputs[inputs["postsynaptic_type"] == "dend"][["x", "y", "z"]].values
    out_axon = outputs[outputs["presynaptic_type"] == "axon"][["x", "y", "z"]].values
    out_dend = outputs[outputs["presynaptic_type"] == "dend"][["x", "y", "z"]].values
    return (in_axon, in_dend, out_axon, out_dend)


names = ["Axon input", "Dendrite input", "Axon output", "Dendrite output"]
syn_groups1 = split_in_out(label1_inputs, label1_outputs)
syn_groups2 = split_in_out(label2_inputs, label2_outputs)

stat_series = pd.Series(index=names)

ksamp = KSample("Dcorr")
for i, n in enumerate(names):
    data1 = syn_groups1[i]
    data2 = syn_groups2[i]
    stat = ksamp._statistic(data1, data2)
    stat_series[n] = stat

print(stat_series)
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

