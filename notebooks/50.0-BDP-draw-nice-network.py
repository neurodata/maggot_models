# %% [markdown]
# #
import os
from operator import itemgetter

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable

from graspy.embed import LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.simulations import sbm
from graspy.utils import binarize, get_lcc, is_fully_connected, symmetrize
from src.data import load_everything
from src.hierarchy import normalized_laplacian, signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVEFIGS = True
SAVESKELS = True
SAVEOBJS = True


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


def stashobj(obj, name, **kws):
    saveobj(obj, name, foldername=FNAME, save_on=SAVEOBJS, **kws)


def n_to_labels(n):
    n = np.array(n)
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


# block_probs = get_feedforward_B(low_p, diag_p, feedforward_p)
def get_recursive_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    alternating = np.ones(2 * n_blocks - 1)
    alternating[1::2] = 0
    late_alternating = np.ones(2 * n_blocks - 1)
    late_alternating[::2] = 0
    B = np.zeros((2 * n_blocks, 2 * n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B -= np.diag(np.diag(B, k=-1), k=-1)
    B += np.diag(diag_p * np.ones(2 * n_blocks))
    B += np.diag((feedforward_p - low_p) * alternating[:-1], k=2)
    B += np.diag(diag_p * alternating, k=1)
    B += np.diag(low_p * late_alternating, k=1)
    B += np.diag(diag_p * alternating, k=-1)
    B += np.diag(low_p * late_alternating, k=-1)
    return B


low_p = 0.01
diag_p = 0.2
feedforward_p = 0.3
n_blocks = 5


block_probs = get_recursive_feedforward_B(
    low_p, diag_p, feedforward_p, n_blocks=n_blocks
)
plt.figure(figsize=(10, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds")


community_sizes = np.empty(2 * n_blocks, dtype=int)
n_feedforward = 100
n_feedback = 100
community_sizes[::2] = n_feedforward
community_sizes[1::2] = n_feedback
labels = n_to_labels(community_sizes)


A = sbm(community_sizes, block_probs, directed=True, loops=False)
n_verts = A.shape[0]

heatmap(A, cbar=False)
shuffle_inds = np.random.permutation(n_verts)
A = A[np.ix_(shuffle_inds, shuffle_inds)]
labels = labels[shuffle_inds]

namelist = ["b1", "b1fb", "b2", "b2fb", "b3", "b3fb", "b4", "b4fb", "b5", "b5fb"]
name_map = dict(zip(range(2 * n_blocks), namelist))
labels = np.array(itemgetter(*labels)(name_map))
# %% [markdown]
# #

blockmodel_df = get_blockmodel_df(A, labels, use_weights=True, return_counts=False)
sns.heatmap(blockmodel_df, annot=True, cmap="Reds")

# %% [markdown]
# # make the networkx graph


from graspy.embed import AdjacencySpectralEmbed

g = nx.from_pandas_adjacency(blockmodel_df, create_using=nx.DiGraph())
uni_labels, counts = np.unique(labels, return_counts=True)
size_scaler = 5
size_map = dict(zip(uni_labels, size_scaler * counts))
nx.set_node_attributes(g, size_map, name="Size")
adj = nx.to_numpy_array(g, nodelist=uni_labels)
node_signal_flow = signal_flow(adj)
sf_map = dict(zip(uni_labels, node_signal_flow))
nx.set_node_attributes(g, sf_map, name="Signal Flow")
sym_adj = symmetrize(adj)
node_lap = AdjacencySpectralEmbed(n_components=1).fit_transform(sym_adj)
node_lap = np.squeeze(node_lap)
lap_map = dict(zip(uni_labels, node_lap))
nx.set_node_attributes(g, lap_map, name="Laplacian-2")
color_map = dict(zip(uni_labels, cc.glasbey_light))
nx.set_node_attributes(g, color_map, name="Color")
g.nodes(data=True)


def draw_networkx_nice(
    g,
    x_pos,
    y_pos,
    sizes=None,
    colors=None,
    nodelist=None,
    cmap="Blues",
    ax=None,
    x_boost=0,
    y_boost=0,
    draw_axes_arrows=False,
):
    if nodelist is None:
        nodelist = g.nodes()
    weights = nx.get_edge_attributes(g, "weight")

    x_attr_dict = nx.get_node_attributes(g, x_pos)
    y_attr_dict = nx.get_node_attributes(g, y_pos)

    pos = {}
    label_pos = {}
    for n in nodelist:
        pos[n] = (x_attr_dict[n], y_attr_dict[n])
        label_pos[n] = (x_attr_dict[n] + x_boost, y_attr_dict[n] + y_boost)

    if sizes is not None:
        size_attr_dict = nx.get_node_attributes(g, sizes)
        node_size = []
        for n in nodelist:
            node_size.append(size_attr_dict[n])

    if colors is not None:
        color_attr_dict = nx.get_node_attributes(g, colors)
        node_color = []
        for n in nodelist:
            node_color.append(color_attr_dict[n])

    weight_array = np.array(list(weights.values()))
    norm = mplc.Normalize(vmin=0.1, vmax=weight_array.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cmap = sm.to_rgba(weight_array)

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 30), frameon=False)

    node_collection = nx.draw_networkx_nodes(
        g, pos, node_color=node_color, node_size=node_size, with_labels=False, ax=ax
    )
    n_squared = len(nodelist) ** 2  # maximum z-order so far
    node_collection.set_zorder(n_squared)

    nx.draw_networkx_edges(
        g,
        pos,
        edge_color=cmap,
        connectionstyle="arc3,rad=0.2",
        arrows=True,
        width=1.5,
        ax=ax,
    )

    text_items = nx.draw_networkx_labels(g, label_pos, ax=ax)

    # make sure the labels are above all in z order
    for _, t in text_items.items():
        t.set_zorder(n_squared + 1)

    if draw_axes_arrows:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # get width and height of axes object to compute
        # matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height

        # manual arrowhead width and length
        hw = 1.0 / 50.0 * (ymax - ymin)
        hl = 1.0 / 50.0 * (xmax - xmin)
        lw = 3  # axis line width
        ohg = 0.3  # arrow overhang

        # compute matching arrowhead length and width
        yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
        yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height
        ylw = lw * (xmax - xmin) / (ymax - ymin)

        # draw x and y axis
        ax.arrow(
            xmin,
            ymin,
            xmax - xmin,
            0.0,
            fc="k",
            ec="k",
            lw=lw,
            head_width=hw,
            head_length=hl,
            overhang=ohg,
            length_includes_head=True,
            clip_on=False,
        )

        ax.arrow(
            xmin,
            ymin,
            0.0,
            ymax - ymin,
            fc="k",
            ec="k",
            lw=ylw,
            head_width=yhw,
            head_length=yhl,
            overhang=ohg,
            length_includes_head=True,
            clip_on=False,
        )
    ax.set_xlabel(x_pos)
    ax.set_ylabel(y_pos)
    # plt.box(False)
    fig.set_facecolor("w")
    return ax


draw_networkx_nice(g, "Laplacian-2", "Signal Flow", sizes="Size", colors="Color")

stashfig("try-drawing")


# %% [markdown]
# #
nx.write_graphml(
    g, "maggot_models/notebooks/outs/50.0-BDP-draw-nice-network/test_gml.graphml"
)

