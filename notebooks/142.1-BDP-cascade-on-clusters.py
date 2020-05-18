# %% [markdown]
# ##
import os
import warnings
from itertools import chain

import matplotlib as mpl
import matplotlib.gridspec as gridspec
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
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplace,
)

import matplotlib.patches as patches

from src.align import Procrustes
from src.cluster import BinaryCluster, MaggotCluster, get_paired_inds
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.pymaid import start_instance
from src.traverse import Cascade, RandomWalk, to_markov_matrix, to_transmission_matrix
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    remove_spines,
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
pair_meta = pair_mg.meta

# full_mg = load_metagraph("G")
# full_mg.meta[]
# full_meta = pair_meta
# full_adj = pair_adjs
full_meta = pair_meta
full_mg = pair_mg

# parameters
lowest_level = 8

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
full_adj = full_mg.adj

n_leaf = full_meta[f"lvl{lowest_level}_labels"].nunique()
n_pairs = len(full_meta) // 2

# %% [markdown]
# ##

from graspy.models import SBMEstimator

level = 2

n_row = 3
n_col = 7
scale = 10
fig, axs = plt.subplots(n_row, n_col, figsize=(n_row * scale, n_col * scale))

for level in range(8):
    label_name = f"lvl{level}_labels_side"
    sbm = SBMEstimator(directed=True, loops=True)
    sbm.fit(binarize(full_adj), full_meta[label_name].values)
    ax = axs[1, level]
    _, _, top, _ = adjplot(
        sbm.p_mat_,
        ax=ax,
        plot_type="heatmap",
        sort_class=["hemisphere"] + level_names[: level + 1],
        item_order=["merge_class_sf_order", "merge_class", "sf"],
        class_order="sf",
        meta=full_mg.meta,
        palette=CLASS_COLOR_DICT,
        colors="merge_class",
        ticks=False,
        gridline_kws=dict(linewidth=0.05, color="grey", linestyle="--"),
        cbar_kws=dict(shrink=0.6),
    )
stashfig("big-bhat-fig")
# %% [markdown]
# ##
# Get positions for left and right simultaneously, so they'll line up ###
def get_mid_map(full_meta, leaf_key=None, bilat=False):
    if leaf_key is None:
        leaf_key = f"lvl{lowest_level}_labels"
    # left
    if not bilat:
        meta = full_meta[full_meta["hemisphere"] == "L"].copy()
    else:
        meta = full_meta.copy()

    sizes = meta.groupby([leaf_key, "merge_class"], sort=False).size()

    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    left_mid_map = dict(zip(uni_labels, mids))
    if bilat:
        first_mid_map = {}
        for k in left_mid_map.keys():
            left_mid = left_mid_map[k]
            first_mid_map[k + "-"] = left_mid
        return first_mid_map

    # right
    meta = full_meta[full_meta["hemisphere"] == "R"].copy()

    sizes = meta.groupby([leaf_key, "merge_class"], sort=False).size()

    # uni_labels = np.unique(labels)
    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    right_mid_map = dict(zip(uni_labels, mids))

    keys = list(set(list(left_mid_map.keys()) + list(right_mid_map.keys())))
    first_mid_map = {}
    for k in keys:
        left_mid = left_mid_map[k]
        right_mid = right_mid_map[k]
        first_mid_map[k + "-"] = max(left_mid, right_mid)
    return first_mid_map


first_mid_map = get_mid_map(full_meta, bilat=True)


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


def draw_bar_dendrogram(meta, ax, orientation="vertical", width=0.5):
    last_mid_map = first_mid_map
    line_kws = dict(linewidth=1, color="k")
    for level in np.arange(lowest_level + 1)[::-1]:
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
                if orientation == "vertical":
                    ax.bar(
                        x=level,
                        height=heights[i],
                        width=width,
                        bottom=starts[i],
                        color=colors[i],
                    )
                else:
                    ax.barh(
                        y=level,
                        height=width,
                        width=heights[i],
                        left=starts[i],
                        color=colors[i],
                    )

            # draw a horizontal line from the middle of this bar
            if level != 0:  # dont plot dash on the last

                if orientation == "vertical":
                    xs = [level - 0.5 * width, level - width]
                    ys = [mid, mid]
                else:
                    ys = [level - 0.5 * width, level - width]
                    xs = [mid, mid]
                ax.plot(xs, ys, **line_kws)

            # line connecting to children clusters
            if level != lowest_level:  # don't plot first dash
                if orientation == "vertical":
                    xs = [level + 0.5 * width, level + width]
                    ys = [grand_mid, grand_mid]
                else:
                    ys = [level + 0.5 * width, level + width]
                    xs = [grand_mid, grand_mid]

                ax.plot(xs, ys, **line_kws)

            # draw a vertical line connecting the two child clusters
            if len(last_mids) == 2:
                if orientation == "vertical":
                    xs = [level + width, level + width]
                    ys = last_mids
                else:
                    xs = last_mids
                    ys = [level + width, level + width]
                ax.plot(xs, ys, **line_kws)

        last_mid_map = dict(zip(uni_labels, mids))


# %% [markdown]
# ##
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.utils import get_blockmodel_df

labels = full_meta[f"lvl{lowest_level}_labels"].values

mid_map = {}
for key, val in first_mid_map.items():
    new_key = key[:-1]
    mid_map[new_key] = val

blockmodel_df = get_blockmodel_df(
    full_adj, labels, return_counts=True, use_weights=True
)
group_sizes = full_meta.groupby([f"lvl{lowest_level}_labels"]).size()
blockmodel_df.index.name = "source"
blockmodel_df.columns.name = "target"
blockmodel_df.reset_index(inplace=True)
blockmodel_df
blockmodel_edges = blockmodel_df.melt(id_vars="source", value_name="weight")
blockmodel_edges["x"] = blockmodel_edges["target"].map(mid_map)
blockmodel_edges["y"] = blockmodel_edges["source"].map(mid_map)
blockmodel_edges["source_size"] = blockmodel_edges["source"].map(group_sizes)
blockmodel_edges["target_size"] = blockmodel_edges["target"].map(group_sizes)
blockmodel_edges["source_n_out"] = blockmodel_edges["source"].map(
    blockmodel_edges.groupby("source")["weight"].sum()
)
blockmodel_edges["out_weight"] = (
    blockmodel_edges["weight"] / blockmodel_edges["source_n_out"]
)

blockmodel_edges["target_n_in"] = blockmodel_edges["target"].map(
    blockmodel_edges.groupby("target")["weight"].sum()
)

blockmodel_edges["in_weight"] = (
    blockmodel_edges["weight"] / blockmodel_edges["target_n_in"]
)
blockmodel_edges["norm_weight"] = blockmodel_edges["weight"] / np.sqrt(
    (blockmodel_edges["source_size"] * blockmodel_edges["target_size"])
)
sns.set_context("talk")


fig, main_ax = plt.subplots(1, 1, figsize=(30, 30))
main_ax.set_ylim((-gap, (2 * n_pairs + gap * n_leaf)))
main_ax.set_xlim(((2 * n_pairs + gap * n_leaf), -gap))
# sns.scatterplot(
#     data=blockmodel_edges,
#     x="x",
#     y="y",
#     size="in_weight",
#     legend=False,
#     sizes=(0, 600),
#     # hue="out_weight",
#     # palette="Blues",
#     # marker="s",
# )

meta = full_meta.copy()
last_mid_map = first_mid_map
sizes = meta.groupby([f"lvl{lowest_level}_labels", "merge_class"], sort=False).size()
uni_labels = sizes.index.unique(0)  # these need to be in the right order
mins = []
maxs = []
for ul in uni_labels:
    last_mids = get_last_mids(ul, last_mid_map)
    grand_mid = np.mean(last_mids)

    heights, starts, colors = calc_bar_params(sizes, ul, grand_mid)

    minimum = starts[0]
    maximum = starts[-1] + heights[-1]
    xs = [minimum, maximum, maximum, minimum, minimum]
    ys = [minimum, minimum, maximum, maximum, minimum]
    #     plt.plot(xs, ys)
    mins.append(minimum)
    maxs.append(maximum)
bound_df = pd.DataFrame(data=[mins, maxs], columns=uni_labels).T
for x in range(len(bound_df)):
    for y in range(len(bound_df)):
        min_x = bound_df.iloc[x, 0]
        min_y = bound_df.iloc[y, 0]
        max_x = bound_df.iloc[x, 1]
        max_y = bound_df.iloc[y, 1]
        width = max_x - min_x
        height = max_y - min_y
        x_label = bound_df.index[x]
        y_label = bound_df.index[y]

        edge = blockmodel_edges[
            (blockmodel_edges["source"] == y_label)
            & (blockmodel_edges["target"] == x_label)
        ]
        rect = patches.Rectangle((min_x, min_y), width, height)
        main_ax.add_patch(rect)


main_ax.set_xlabel("")
main_ax.set_ylabel("")

remove_spines(main_ax)
divider = make_axes_locatable(main_ax)


meta = full_meta.copy()

left_ax = divider.append_axes("left", size="20%", pad=0, sharey=main_ax)
ax = left_ax
# ax.set_ylim((-gap, (2 * n_pairs + gap * n_leaf)))
ax.set_xlim((-0.5, lowest_level + 0.5))
draw_bar_dendrogram(meta, ax)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Level")
ax.set_xticks(np.arange(lowest_level + 1))
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="both", which="both", length=0)

# add a scale bar in the bottom left
width = 0.5
ax.bar(x=0, height=100, bottom=0, width=width, color="k")
ax.text(x=0.35, y=0, s="100 neurons")

top_ax = divider.append_axes("top", size="20%", pad=0, sharex=main_ax)
ax = top_ax
# ax.set_xlim(((2 * n_pairs + gap * n_leaf), -gap))
ax.set_ylim((lowest_level + 0.5, -0.5))
draw_bar_dendrogram(meta, ax, orientation="horizontal")
ax.set_xticks([])
ax.spines["left"].set_visible(False)
ax.set_ylabel("Level")
ax.set_yticks(np.arange(lowest_level + 1))
ax.spines["bottom"].set_visible(False)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.tick_params(axis="both", which="both", length=0)

stashfig(f"sbm-test-dendrogram-lowest={lowest_level}")

