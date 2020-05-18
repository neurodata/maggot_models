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
from tqdm import tqdm

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
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", dpi=400, **kws)


def plot_fragments(nl, splits, neuron_class=None, scale=8):
    n_col = len(nl)
    fig = plt.figure(figsize=(scale * n_col, scale))  # constrained_layout=True)
    for i, n in enumerate(nl):
        ax = fig.add_subplot(1, n_col, i + 1, projection="3d")
        skid = int(n.skeleton_id)
        if skid in splits.index:
            split_nodes = splits[skid]
            split_locs = pymaid.get_node_location(split_nodes)
            split_locs = split_locs[["x", "y", "z"]].values
            pymaid.plot2d(
                split_locs, ax=ax, scatter_kws=dict(color="orchid", s=30), method="3d"
            )
            # order of output is axon, dendrite
            fragments = pymaid.cut_neuron(n, split_nodes)
        else:
            fragments = [n]
        n_frag = len(fragments)
        for i, f in enumerate(fragments):
            if n_frag == 1:
                color = colors[2]  # unsplitable
            elif i == n_frag - 1:
                color = colors[1]  # dendrite
            else:
                color = colors[0]  # axon
            f.plot2d(ax=ax, color=color, method="3d")
            title = f"{neuron_class}, {n.neuron_name}, {n.skeleton_id}"
            ax.set_title(title, color="grey")
        set_axes_equal(ax)
    plt.tight_layout()


def get_savename(nl, neuron_class=None):
    savename = f"{neuron_class}"
    for n in nl:
        savename += f"-{n.skeleton_id}"
    savename += "-split"
    return savename


mg = load_metagraph("G")
meta = mg.meta

start_instance()

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)
connection_types = ["axon", "dendrite", "unsplittable"]
pal = sns.color_palette("deep", 5)
colors = [pal[1], pal[2], pal[4]]
connection_colors = dict(zip(connection_types, colors))


splits = pymaid.find_treenodes(tags="mw axon split")
splits = splits.set_index("skeleton_id")["treenode_id"].squeeze()


# plot paired neurons
pair_meta = meta[meta["pair_id"] != -1]
pairs = pair_meta["pair_id"].unique()

for p in pairs:
    temp_meta = pair_meta[pair_meta["pair_id"] == p]
    skids = temp_meta.index.values.astype(int)
    neuron_class = temp_meta["merge_class"].iloc[0]
    nl = pymaid.get_neurons(skids)
    plot_fragments(nl, splits, neuron_class=neuron_class)
    stashfig(get_savename(nl, neuron_class=neuron_class))

# plot unpaired neurons
unpair_meta = meta[meta["pair_id"] == -1]

for skid, row in unpair_meta.iterrows():
    neuron_class = row["merge_class"]
    nl = pymaid.get_neurons([skid])
    nl = pymaid.CatmaidNeuronList(nl)
    plot_fragments(nl, splits, neuron_class=neuron_class)
    stashfig(get_savename(nl, neuron_class=neuron_class))
