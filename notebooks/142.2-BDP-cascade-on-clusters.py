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
from graspy.match import GraphMatch
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

# %% [markdown]
# ##


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


def diag_indices(length, k=0):
    neg = False
    if k < 0:
        neg = True
    k = np.abs(k)
    inds = (np.arange(length - k), np.arange(k, length))
    if neg:
        return (inds[1], inds[0])
    else:
        return inds


def exp_func(k, alpha, beta=1, c=0):
    return beta * np.exp(-alpha * (k - 1)) + c


def calc_mean_by_k(ks, perm_adj):
    length = len(perm_adj)
    ps = []
    for k in ks:
        p = perm_adj[diag_indices(length, k)].mean()
        ps.append(p)
    return np.array(ps)


def get_vals_by_k(ks, perm_adj):
    ys = []
    xs = []
    for k in ks:
        y = perm_adj[diag_indices(len(perm_adj), k)]
        ys.append(y)
        x = np.full(len(y), k)
        xs.append(x)
    return np.concatenate(ys), np.concatenate(xs)


def make_flat_match(length, **kws):
    match_mat = np.zeros((length, length))
    match_mat[np.triu_indices(length, k=1)] = 1
    return match_mat


def make_linear_match(length, offset=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = length - k + offset
    return match_mat


def normalize_match(graph, match_mat):
    return match_mat / match_mat.sum() * graph.sum()


def make_exp_match(length, alpha=0.5, beta=1, c=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = exp_func(k, alpha, beta, c)
    return match_mat


def fit_gm_exp(
    adj,
    alpha,
    beta=1,
    c=0,
    n_init=5,
    norm=False,
    max_iter=80,
    eps=0.05,
    n_jobs=1,
    verbose=0,
):
    gm = GraphMatch(
        n_init=1, init_method="rand", max_iter=max_iter, eps=eps, shuffle_input=True
    )
    length = len(adj)
    match_mat = make_exp_match(length, alpha=alpha)
    if norm:
        match_mat = normalize_match(adj, match_mat)

    seeds = np.random.choice(int(1e8), size=n_init)

    def _fit(seed):
        np.random.seed(seed)
        gm.fit(match_mat, adj)
        return gm.perm_inds_, gm.score_

    outs = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_fit)(s) for s in seeds)
    outs = list(zip(*outs))
    perms = np.array(outs[0])
    scores = np.array(outs[1])
    return perms, scores


def get_best_run(perms, scores, n_opts=None):
    if n_opts is None:
        n_opts = len(perms)
    opt_inds = np.random.choice(len(perms), n_opts, replace=False)
    perms = perms[opt_inds]
    scores = scores[opt_inds]
    max_ind = np.argmax(scores)
    return perms[max_ind], scores[max_ind]


# %% [markdown]
# ##

# parameters for the experiment
metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "iso"

# parameters for plotting
lowest_level = 7
width = 0.5
gap = 10

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.0-BDP-omni-clust"

# load data
pair_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
pair_meta["lvl0_labels"] = pair_meta["lvl0_labels"].astype(str)
pair_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
pair_mg = MetaGraph(pair_adj.values, pair_meta)
pair_meta = pair_mg.meta

level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
full_mg = load_metagraph("G")

# this determines the sorting for everybody
sort_class = level_names + ["merge_class"]
class_order = ["sf"]
total_sort_by = []
for sc in sort_class:
    for co in class_order:
        class_value = pair_meta.groupby(sc)[co].mean()
        pair_meta[f"{sc}_{co}_order"] = pair_meta[sc].map(class_value)
        total_sort_by.append(f"{sc}_{co}_order")
    total_sort_by.append(sc)

pair_mg = pair_mg.sort_values(total_sort_by, ascending=False)
pair_meta = pair_mg.meta
pair_adj = pair_mg.adj

n_pairs = len(pair_meta) // 2

n_leaf = pair_meta[f"lvl{lowest_level}_labels"].nunique()  # TODO change

# %% [markdown]
# ##

full_meta = full_mg.meta
motor_meta = full_meta[full_meta["class1"] == "motor"]
# kc_meta = full_meta[full_meta["class1"] == "KC"]
an_meta = full_meta[full_meta["merge_class"] == "sens-AN"]
mn_meta = full_meta[full_meta["merge_class"] == "sens-MN"]
photo_meta = full_meta[
    full_meta["merge_class"].isin(("sens-photoRh5", "sens-photoRh6"))
]
pan_meta = full_meta[full_meta["merge_class"] == "sens-PaN"]
first_meta = pd.concat(
    (pair_meta, motor_meta, an_meta, mn_meta, photo_meta, pan_meta), sort=False
)
first_meta["leaf_label"] = first_meta[f"lvl{lowest_level}_labels"]
no_cluster_inds = first_meta[first_meta[f"lvl{lowest_level}_labels"].isna()].index
first_meta.loc[no_cluster_inds, "leaf_label"] = first_meta.loc[
    no_cluster_inds, "merge_class"
]

unpaired_meta = pd.concat(
    (motor_meta, an_meta, mn_meta, photo_meta, pan_meta), sort=False
)

# %% [markdown]
# ##


# Get positions for left and right simultaneously, so they'll line up ###
def get_mid_map(meta, leaf_key=None, bilat=False):
    if leaf_key is None:
        leaf_key = f"lvl{lowest_level}_labels"
    # left
    if not bilat:
        temp_meta = meta[meta["hemisphere"] == "L"].copy()
    else:
        temp_meta = meta.copy()

    sizes = temp_meta.groupby([leaf_key, "merge_class"], sort=False).size()

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
    temp_meta = meta[meta["hemisphere"] == "R"].copy()

    sizes = temp_meta.groupby([leaf_key, "merge_class"], sort=False).size()

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


first_mid_map = get_mid_map(first_meta, "leaf_label", bilat=False)


def calc_bar_params(sizes, label, mid):
    heights = sizes.loc[label]
    n_in_bar = heights.sum()
    offset = mid - n_in_bar / 2
    starts = heights.cumsum() - heights + offset
    if not isinstance(heights, pd.Series):
        colors = [CLASS_COLOR_DICT[label]]
    else:
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


def draw_bar_dendrogram(meta, ax, orientation="vertical"):
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
source_group_names = ["Odor", "MN", "AN", "Photo", "Thermo", "VTD"]

max_hops = 3
dfs = []

collapse = True
for i, sg_name in enumerate(source_group_names):
    # extract the relevant columns from the above
    cols = [f"{sg_name}_{t}_visits" for t in range(max_hops)]
    # get the mean (or sum?) visits by cluster
    visits = full_mg.meta.groupby(f"lvl{lowest_level}_labels")[cols].sum()
    cluster_names = visits.index.copy()
    cluster_names = np.vectorize(lambda x: x + "-")(cluster_names)
    ys = np.vectorize(first_mid_map.get)(cluster_names)
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
    dfs.append(temp_df)

visit_df = pd.concat(dfs, axis=0)

col_norm = True
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


# %% [markdown]
# ##
def remove_axis(ax):
    remove_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])


n_col = 3

# width_ratios = 4 * [1] + (lowest_level + 1) * [1.5]
collapse = False
mid_width = 5 * 0.1  # len(source_group_names) * 0.1
if not collapse:
    mid_width *= max_hops

width_ratios = [1, mid_width, 1]
fig = plt.figure(constrained_layout=False, figsize=(10 + mid_width * 0.2, 20))
gs = gridspec.GridSpec(
    nrows=1, ncols=n_col, figure=fig, width_ratios=width_ratios, wspace=0
)
meta = pair_meta[pair_meta["hemisphere"] == "L"].copy()

ax = fig.add_subplot(gs[:, 0])
ax.set_title("Left")
ax.set_ylim((-gap, (len(first_meta) / 2 + gap * n_leaf)))
ax.set_xlim((-0.5, lowest_level + 0.5))

draw_bar_dendrogram(meta, ax)


sizes = unpaired_meta.groupby(["merge_class"], sort=False).size()

uni_labels = sizes.index.unique()  # these need to be in the right order

# mids = []
for ul in uni_labels:
    last_mids = get_last_mids(ul, first_mid_map)
    grand_mid = np.mean(last_mids)
    heights, starts, colors = calc_bar_params(sizes, ul, grand_mid)
    ax.bar(x=lowest_level, height=heights, bottom=starts, color=colors)

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
meta = pair_meta[pair_meta["hemisphere"] == "R"].copy()

ax = fig.add_subplot(gs[:, 2])
ax.set_title("Right")
ax.set_ylim((-gap, (len(first_meta) / 2 + gap * n_leaf)))
ax.set_xlim((lowest_level + 0.5, -0.5))  # reversed x axis order to make them mirror

draw_bar_dendrogram(meta, ax)

ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Level")
ax.set_xticks(np.arange(lowest_level + 1))
ax.tick_params(axis="both", which="both", length=0)

# %% [markdown]
# ##
# # center fig

if collapse:
    n_inner_col = 1
else:
    n_inner_col = max_hops
n_groups = 5  # len(source_group_names)

ax = fig.add_subplot(gs[:, 1])
# remove_axis(ax)
# ax.set_ylim((-gap, (n_pairs + gap * n_leaf)))
# ax.set_xlim((-0.5, ((n_groups - 1) * n_inner_col) + 0.5))
# # top_ax = ax.twinx()

# top_ax.set_xticks(top_tick_locs)
# # top_ax.set_xticklabels(source_group_names)

# same_size_norm = True
# if same_size_norm:
#     sns.scatterplot(
#         data=visit_df,
#         x="x",
#         y="y",
#         size="sizes",
#         hue="sg",
#         sizes=(1, 70),
#         ax=ax,
#         legend=False,
#         palette="tab10",
#     )
# else:
#     sns.set_palette(sns.color_palette("tab10"))
#     for i, sg in enumerate(source_group_names):
#         sns.scatterplot(
#             data=visit_df[visit_df["sg"] == sg],
#             x="x",
#             y="y",
#             size="sizes",
#             sizes=(1, 70),
#             ax=ax,
#             legend=False,
#         )

# top_tick_locs = np.arange(start=0, stop=n_groups * (n_inner_col), step=n_inner_col)

# for i, t in enumerate(top_tick_locs):
#     ax.text(
#         t, Y.max() + 3 * gap, source_group_names[i], rotation="vertical", ha="center"
#     )

# if not collapse:
#     ax.set_xticks(np.arange(n_groups * n_inner_col))
#     ax.set_xticklabels(np.tile(np.arange(1, n_inner_col + 1), n_groups))
#     ax.set_xlabel("Hops")
# else:
#     ax.set_xticks([])
#     ax.set_xlabel("")
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_yticks([])
ax.set_ylabel("")
ax.tick_params(axis="both", which="both", length=0)

plt.tight_layout()

# basename = f"-max_hops={max_hops}-row_norm={row_norm}-col_norm={col_norm}-same_scale={same_size_norm}"
stashfig("cascade-bars")  # + basename)

# %% [markdown]
# ##


# right side
# meta = full_meta[full_meta["hemisphere"] == "R"].copy()
#%%
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
        sizes=(1, 70),
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
            sizes=(1, 70),
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

basename = f"-max_hops={max_hops}-row_norm={row_norm}-col_norm={col_norm}-same_scale={same_size_norm}"
stashfig("cascade-bars" + basename)

# for t in range(max_hops):
#     pass

# ax = fig.add_subplot(gs[:, 3])
# full_sizes = full_meta.groupby(["merge_class"], sort=False).size()
# uni_class = full_sizes.index.unique()
# counts = full_sizes.values
# count_map = dict(zip(uni_class, counts))
# names = []
# colors = []
# for key, val in count_map.items():
#     names.append(f"{key} ({count_map[key]})")
#     colors.append(CLASS_COLOR_DICT[key])
# colors = colors[::-1]  # reverse because of signal flow sorting
# names = names[::-1]
# palplot(len(colors), colors, ax=ax)
# ax.yaxis.set_major_formatter(plt.FixedFormatter(names))

# %%
