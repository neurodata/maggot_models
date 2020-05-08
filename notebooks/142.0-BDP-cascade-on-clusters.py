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

exp = "137.0-BDP-omni-clust"

# load data
full_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
full_meta["lvl0_labels"] = full_meta["lvl0_labels"].astype(str)
full_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
full_mg = MetaGraph(full_adj.values, full_meta)

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

# %% [markdown]
# ## Random walk stuff


ad_mg = load_metagraph("Gad")
ad_mg = preprocess(ad_mg, sym_threshold=False, remove_pdiff=True, binarize=False)
ad_mg.meta["inds"] = range(len(ad_mg))
ad_adj = ad_mg.adj
meta = ad_mg.meta


source_groups = [
    ("sens-ORN",),
    ("sens-MN",),
    ("sens-photoRh5", "sens-photoRh6"),
    ("sens-thermo",),
    ("sens-vtd",),
    ("sens-AN",),
    ("dVNC", "dVNC;CN", "dVNC;RG", "dSEZ;dVNC"),
    ("dSEZ", "dSEZ;CN", "dSEZ;LHN", "dSEZ;dVNC"),
    ("motor-PaN", "motor-MN", "motor-VAN", "motor-AN"),
    ("RG", "RG-IPC", "RG-ITP", "RG-CA-LP", "dVNC;RG"),
    ("dUnk",),
]
source_group_names = [
    "Odor",
    "MN",
    "Photo",
    "Thermo",
    "VTD",
    "AN",
    "dVNC",
    "dSEZ",
    "Motor",
    "RG",
    "dUnk",
]


class_key = "merge_class"

np.random.seed(888)
max_hops = 10
n_init = 2 ** 7
p = 0.01


path_method = "cascade"

if path_method == "rw":
    transition_probs = to_markov_matrix(ad_adj)
elif path_method == "cascade":
    transition_probs = to_transmission_matrix(ad_adj, p=p)


def rw_from_node(s):
    paths = []
    rw = RandomWalk(transition_probs, max_hops=10, allow_loops=False)
    for n in range(n_init):
        rw.start(s)
        paths.append(rw.traversal_)
    return paths


def cascade_from_nodes(s):
    cascades = []
    csc = Cascade(transition_probs, max_hops=6, allow_loops=False)
    for n in range(n_init):
        csc.start(s)
        cascades.append(csc.traversal_)
    return cascades


for sg, sg_name in zip(source_groups, source_group_names):
    source_inds = meta[meta[class_key].isin(sg)]["inds"].values
    if path_method == "rw":
        # Run random walks
        par = Parallel(n_jobs=1, verbose=10)
        paths_by_node = par(delayed(rw_from_node)(s) for s in source_inds)
        paths = []
        for p in paths_by_node:
            paths += p
    if path_method == "cascade":
        paths = cascade_from_nodes(source_inds)

    # collect into hit histogram
    hit_hist = np.zeros((len(ad_mg), max_hops))
    for p in paths:
        for t, node in enumerate(p):
            hit_hist[node, t] += 1
    # apply to meta data for graph we clustered
    for i, node_id in enumerate(ad_mg.meta.index):
        if node_id in full_mg.meta.index:
            for t in range(max_hops):
                full_mg.meta.loc[node_id, f"{sg_name}_{t}_visits"] = hit_hist[i, t]


# %% [markdown]
# ##
# Get positions for left and right simultaneously, so they'll line up ###
def get_mid_map(full_meta, lowest_level=7):
    # left
    meta = full_meta[full_meta["hemisphere"] == "L"].copy()

    level = lowest_level
    sizes = meta.groupby([f"lvl{level}_labels", "merge_class"], sort=False).size()

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

    # right
    meta = full_meta[full_meta["hemisphere"] == "R"].copy()

    level = lowest_level
    sizes = meta.groupby([f"lvl{level}_labels", "merge_class"], sort=False).size()

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


first_mid_map = get_mid_map(full_meta)


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


# %% [markdown]
# ##
source_group_names = ["Odor", "MN", "AN", "Photo", "Thermo", "VTD"]
# source_group_names = ["dVNC", "dSEZ", "RG"]
# max_hops = 2
dfs = []

collapse = True
for i, sg_name in enumerate(source_group_names):
    # extract the relevant columns from the above
    cols = [f"{sg_name}_{t}_visits" for t in range(max_hops)]
    # get the mean (or sum?) visits by cluster
    visits = full_mg.meta.groupby(f"lvl{lowest_level}_labels")[cols].sum()
    cluster_names = visits.index.copy()
    dash_cluster_names = np.vectorize(lambda x: x + "-")(cluster_names)
    ys = np.vectorize(first_mid_map.get)(dash_cluster_names)
    if collapse:
        xs = np.arange(1) + i
    else:
        xs = np.arange(max_hops) + max_hops * i
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    sizes = visits.values
    if collapse:
        sizes = sizes.sum(axis=1)
    temp_df = pd.DataFrame()
    temp_df["x"] = X.ravel()
    temp_df["y"] = Y.ravel()
    temp_df["sizes"] = sizes.ravel()
    temp_df["sg"] = sg_name
    temp_df["cluster"] = cluster_names
    n_in_cluster = full_mg.meta.groupby(f"lvl{lowest_level}_labels").size()
    temp_df["n_cluster"] = cluster_names.map(n_in_cluster)
    dfs.append(temp_df)

visit_df = pd.concat(dfs, axis=0)

col_norm = False
if col_norm:
    for i, sg_name in enumerate(source_group_names):
        inds = visit_df[visit_df["sg"] == sg_name].index
        n_visits = visit_df.loc[inds, "sizes"].sum()
        visit_df.loc[inds, "sizes"] /= n_visits

row_norm = False
if row_norm:
    for i, cluster_name in enumerate(cluster_names):
        inds = visit_df[visit_df["cluster"] == cluster_name].index
        n_visits = visit_df.loc[inds, "sizes"].sum()
        visit_df.loc[inds, "sizes"] /= n_visits

count_norm = True
if count_norm:
    visit_df["sizes"] = visit_df["sizes"] / visit_df["n_cluster"]
    # f


def remove_axis(ax):
    remove_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])


sns.set_context("talk")

n_col = 3

# width_ratios = 4 * [1] + (lowest_level + 1) * [1.5]
mid_width = len(source_group_names) * 0.1
if not collapse:
    mid_width *= max_hops

width_ratios = [1, mid_width, 1]
fig = plt.figure(constrained_layout=False, figsize=(10 + mid_width * 0.2, 20))
gs = gridspec.GridSpec(
    nrows=1, ncols=n_col, figure=fig, width_ratios=width_ratios, wspace=0
)
meta = full_meta[full_meta["hemisphere"] == "L"].copy()

ax = fig.add_subplot(gs[:, 0])
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

ax = fig.add_subplot(gs[:, 2])
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

# # center fig

if collapse:
    n_inner_col = 1
else:
    n_inner_col = max_hops
n_groups = len(source_group_names)

ax = fig.add_subplot(gs[:, 1])
# remove_axis(ax)
ax.set_ylim((-gap, (n_pairs + gap * n_leaf)))
ax.set_xlim((-0.5, ((n_groups - 1) * n_inner_col) + 0.5))
# # top_ax = ax.twinx()

# top_ax.set_xticks(top_tick_locs)
# # top_ax.set_xticklabels(source_group_names)

same_size_norm = True
if same_size_norm:
    sns.scatterplot(
        data=visit_df,
        x="x",
        y="y",
        size="sizes",
        hue="sg",
        sizes=(1, 100),
        ax=ax,
        legend=False,
        palette="tab10",
    )
else:
    sns.set_palette(sns.color_palette("tab10"))
    for i, sg in enumerate(source_group_names):
        sns.scatterplot(
            data=visit_df[visit_df["sg"] == sg],
            x="x",
            y="y",
            size="sizes",
            sizes=(1, 100),
            ax=ax,
            legend=False,
        )

top_tick_locs = np.arange(start=0, stop=n_groups * (n_inner_col), step=n_inner_col)

for i, t in enumerate(top_tick_locs):
    ax.text(
        t, Y.max() + 3 * gap, source_group_names[i], rotation="vertical", ha="center"
    )

if not collapse:
    ax.set_xticks(np.arange(n_groups * n_inner_col))
    ax.set_xticklabels(np.tile(np.arange(1, n_inner_col + 1), n_groups))
    ax.set_xlabel("Hops")
else:
    ax.set_xticks([])
    ax.set_xlabel("")
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_yticks([])
ax.set_ylabel("")
ax.tick_params(axis="both", which="both", length=0)

plt.tight_layout()

basename = f"-max_hops={max_hops}-row_norm={row_norm}-col_norm={col_norm}-same_scale={same_size_norm}-count_norm={count_norm}"
stashfig("cascade-bars" + basename)
