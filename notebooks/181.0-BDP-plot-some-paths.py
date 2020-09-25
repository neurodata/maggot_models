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
    savefig(name, foldername=FNAME, save_on=True, fmt="png", dpi=300, **kws)


mg = load_metagraph("G")

meta = mg.meta

# skeleton_color_dict = dict(
#     zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
# )

# fig = plt.figure(figsize=(15, 5))
# # for the morphology plotting
# margin = 0.01
# gap = 0.02
# n_col = 3
# morpho_gs = plt.GridSpec(
#     1,
#     3,
#     figure=fig,
#     wspace=0,
#     hspace=0,
#     left=margin,
#     right=margin + 3 / n_col,
#     top=1 - margin,
#     bottom=margin,
# )

# # plot the neurons and synapses
# morpho_axs = np.empty((1, 3), dtype="O")

# i = 0
# for j in range(3):
#     ax = fig.add_subplot(morpho_gs[i, j], projection="3d")
#     morpho_axs[i, j] = ax
#     ax.axis("off")

# select_classes = ["sens-photoRh5", "sens-photoRh6", "vPN", "LON"]
# select_inds = meta[meta["merge_class"].isin(select_classes)].index.astype(int)
# select_inds = [int(i) for i in select_inds]


# skeleton_color_dict = dict(
#     zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
# )
# colors = sns.color_palette("deep", n_colors=10, desat=0.7)


# def rgb2hex(args):
#     return "#{:02x}{:02x}{:02x}".format(
#         int(args[0] * 255), int(args[1] * 255), int(args[2] * 255)
#     )


# colors = list(map(rgb2hex, colors))

# purple = colors[4]
# red = colors[3]
# grey = colors[7]


# color_map = {"sens-photoRh5": purple, "sens-photoRh6": purple, "vPN": red, "LON": grey}
# skeleton_color_dict = dict(
#     zip(select_inds, np.vectorize(color_map.get)(meta.loc[select_inds, "merge_class"]))
# )


# start_instance()
# plot_3view(select_inds, morpho_axs[0, :], palette=skeleton_color_dict, row_title="")
# stashfig("visual-skeletons")


# %%
interesting_loc = "maggot_models/data/interesting_paths.csv"

import csv

interesting_paths = []
with open(interesting_loc, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
        interesting_paths.append([int(l) for l in line])
interesting_paths

# %% [markdown]
# ##

fig = plt.figure(figsize=(15, 5))
# for the morphology plotting
margin = 0.01
gap = 0.02
n_col = 3
morpho_gs = plt.GridSpec(
    1,
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
morpho_axs = np.empty((1, 3), dtype="O")

i = 0
for j in range(3):
    ax = fig.add_subplot(morpho_gs[i, j], projection="3d")
    morpho_axs[i, j] = ax
    ax.axis("off")

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)

from itertools import chain

interesting_inds = list(np.unique(list(chain.from_iterable(interesting_paths))))
# interesting_inds = set()
# for path in interesting_paths:
#     for p in path:
#         interesting_inds.add(p)
# interesting_paths = list(interesting_inds)

start_instance()
plot_3view(
    interesting_inds[10:], morpho_axs[0, :], palette=skeleton_color_dict, row_title=""
)
stashfig("interesting-skeletons")


# %%
