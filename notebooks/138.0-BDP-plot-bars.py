# %% [markdown]
# ##
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import poisson
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm
from umap import UMAP

from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import pairplot
from graspy.simulations import sbm
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplace,
)
from src.align import Procrustes
from src.cluster import BinaryCluster, MaggotCluster, get_paired_inds
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.io import readcsv
from src.graph import MetaGraph

from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    plot_neurons,
    screeplot,
    set_axes_equal,
    stacked_barplot,
)

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# %% [markdown]
# ##

n_levels = 10  # max # of splits
metric = "bic"
bic_ratio = 1
d = 10  # embedding dimension
method = "aniso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.0-BDP-omni-clust"


full_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
full_meta["lvl0_labels"] = full_meta["lvl0_labels"].astype(str)
full_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
full_mg = MetaGraph(full_adj.values, full_meta)

# %% [markdown]
# ##
full_meta = full_mg.meta

# parameters
lowest_level = 7

width = 0.5
gap = 10

# this determines the sorting for everybody
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
sort_class = level_names + ["merge_class"]
class_order = ["sf"]
total_sort_by = []
for sc in sort_class:
    for co in class_order:
        class_value = full_meta.groupby(sc)[co].mean()
        full_meta[f"{sc}_{co}_order"] = full_meta[sc].map(class_value)
        total_sort_by.append(f"{sc}_{co}_order")
    total_sort_by.append(sc)

full_mg = full_mg.sort_values(total_sort_by, ascending=False)
full_meta = full_mg.meta

n_leaf = full_meta[f"lvl{lowest_level}_labels"].nunique()
n_pairs = len(full_meta) // 2


# Get positions for left and right simultaneously, so they'll line up ###
# left
meta = full_meta[full_meta["hemisphere"] == "L"].copy()

level = lowest_level
labels = meta[f"lvl{level}_labels"]
classes = meta["merge_class"]
sizes = meta.groupby([f"lvl{level}_labels", "merge_class"], sort=False).size()

uni_labels = sizes.index.unique(0)

mids = []
offset = 0
for ul in uni_labels:
    x = level
    heights = sizes.loc[ul]
    starts = heights.cumsum() - heights + offset
    offset += heights.sum() + gap
    minimum = starts[0]
    maximum = starts[-1] + heights[-1]
    mid = (minimum + maximum) / 2
    mids.append(mid)

left_uni_labels = uni_labels
left_mid_map = dict(zip(uni_labels, mids))

# right
meta = full_meta[full_meta["hemisphere"] == "R"].copy()

level = lowest_level
labels = meta[f"lvl{level}_labels"]
classes = meta["merge_class"]
sizes = meta.groupby([f"lvl{level}_labels", "merge_class"], sort=False).size()

# uni_labels = np.unique(labels)
uni_labels = sizes.index.unique(0)

mids = []
offset = 0
for ul in uni_labels:
    x = level
    heights = sizes.loc[ul]
    starts = heights.cumsum() - heights + offset
    offset += heights.sum() + gap
    minimum = starts[0]
    maximum = starts[-1] + heights[-1]
    mid = (minimum + maximum) / 2
    mids.append(mid)

right_uni_labels = uni_labels
right_mid_map = dict(zip(uni_labels, mids))

keys = list(set(list(left_mid_map.keys()) + list(right_mid_map.keys())))
first_mid_map = {}
for k in keys:
    left_mid = left_mid_map[k]
    right_mid = right_mid_map[k]
    first_mid_map[k + "-"] = max(left_mid, right_mid)


def calc_bar_params(sizes, label, mid):
    heights = sizes.loc[label]
    n_in_bar = heights.sum()
    offset = mid - n_in_bar / 2
    starts = heights.cumsum() - heights + offset
    colors = np.vectorize(CLASS_COLOR_DICT.get)(heights.index)
    return heights, starts, colors


def get_last_mids(label, last_mid_map):
    last_mids = []
    if label + "-" in last_mid_map:
        last_mids.append(last_mid_map[label + "-"])
    if label + "-0" in last_mid_map:
        last_mids.append(last_mid_map[label + "-0"])
    if label + "-1" in last_mid_map:
        last_mids.append(last_mid_map[label + "-1"])
    if len(last_mids) == 0:
        print(label + " has no anchor in mid-map")
    return last_mids


def draw_bar_dendrogram(meta, ax):
    last_mid_map = first_mid_map
    line_kws = dict(linewidth=1, color="k")
    for level in np.arange(lowest_level + 1)[::-1]:
        x = level
        sizes = meta.groupby([f"lvl{level}_labels", "merge_class"], sort=False).size()

        uni_labels = sizes.index.unique(0)  # these need to be in the right order

        mids = []
        for ul in uni_labels:
            last_mids = get_last_mids(ul, last_mid_map)
            grand_mid = np.mean(last_mids)

            heights, starts, colors = calc_bar_params(sizes, ul, grand_mid)

            minimum = starts[0]
            maximum = starts[-1] + heights[-1]
            mid = (minimum + maximum) / 2
            mids.append(mid)

            # draw the bars
            for i in range(len(heights)):
                ax.bar(
                    x=x,
                    height=heights[i],
                    width=width,
                    bottom=starts[i],
                    color=colors[i],
                )

            # draw a horizontal line from the middle of this bar
            if level != 0:  # dont plot dash on the last
                ax.plot([x - 0.5 * width, x - width], [mid, mid], **line_kws)

            # line connecting to children clusters
            if level != lowest_level:  # don't plot first dash
                ax.plot(
                    [x + 0.5 * width, x + width], [grand_mid, grand_mid], **line_kws
                )

            # draw a vertical line connecting the two child clusters
            if len(last_mids) == 2:
                ax.plot([x + width, x + width], last_mids, **line_kws)

        last_mid_map = dict(zip(uni_labels, mids))


# left side
# analysis, bars, graph graph graph...
n_col = 1 + 2 + lowest_level + 1
import matplotlib.gridspec as gridspec

fig = plt.figure(constrained_layout=True, figsize=(5 * n_col, 20))
gs = gridspec.GridSpec(nrows=2, ncols=n_col)
fig, axs = plt.subplots(2, n_col)
meta = full_meta[full_meta["hemisphere"] == "L"].copy()

ax = axs[0]
ax.set_title("Left")
ax.set_ylim((-gap, (n_pairs + gap * n_leaf)))
ax.set_xlim((-0.5, lowest_level + 0.5))

draw_bar_dendrogram(meta, ax)

ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Level")
ax.set_xticks(np.arange(lowest_level + 1))
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="both", which="both", length=0)

# add a scale bar in the bottom left
ax.bar(x=0, height=100, bottom=0, width=width, color="k")
ax.text(x=0.35, y=0, s="100 neurons")

# right side
meta = full_meta[full_meta["hemisphere"] == "R"].copy()

ax = axs[1, 1]
ax.set_title("Right")
ax.set_ylim((-gap, (n_pairs + gap * n_leaf)))
ax.set_xlim((lowest_level + 0.5, -0.5))  # reversed x axis order to make them mirror

draw_bar_dendrogram(meta, ax)

ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Level")
ax.set_xticks(np.arange(lowest_level + 1))
ax.tick_params(axis="both", which="both", length=0)

plt.tight_layout()

stashfig(f"dendrobars-lowest={lowest_level}" + basename)


# %%
