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


metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.0-BDP-omni-clust"


full_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
full_meta["lvl0_labels"] = full_meta["lvl0_labels"].astype(str)
# full_meta["sf"] = -full_meta["sf"]
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
n_col = 1 + 2 + 1 + lowest_level + 1
import matplotlib.gridspec as gridspec

width_ratios = 4 * [1] + (lowest_level + 1) * [1.5]
fig = plt.figure(
    constrained_layout=False, figsize=(5 * 4 + (lowest_level + 1) * 8.5, 20)
)
gs = gridspec.GridSpec(nrows=6, ncols=n_col, figure=fig, width_ratios=width_ratios)
meta = full_meta[full_meta["hemisphere"] == "L"].copy()

ax = fig.add_subplot(gs[:, 1])
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

ax = fig.add_subplot(gs[:, 3])
full_sizes = full_meta.groupby(["merge_class"], sort=False).size()
uni_class = full_sizes.index.unique()
counts = full_sizes.values
count_map = dict(zip(uni_class, counts))
names = []
colors = []
for key, val in count_map.items():
    names.append(f"{key} ({count_map[key]})")
    colors.append(CLASS_COLOR_DICT[key])
colors = colors[::-1]  # reverse because of signal flow sorting
names = names[::-1]
palplot(len(colors), colors, ax=ax)
ax.yaxis.set_major_formatter(plt.FixedFormatter(names))

# plt.tight_layout()
model = DCSBMEstimator
for level in np.arange(lowest_level + 1):
    ax = fig.add_subplot(gs[:3, level + 4])
    adj = binarize(full_mg.adj)
    _, _, top, _ = adjplot(
        adj,
        ax=ax,
        plot_type="scattermap",
        sizes=(0.5, 0.5),
        sort_class=["hemisphere"] + level_names[: level + 1],
        item_order=["merge_class_sf_order", "merge_class", "sf"],
        class_order="sf",
        meta=full_mg.meta,
        palette=CLASS_COLOR_DICT,
        colors="merge_class",
        ticks=False,
        gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
    )
    top.set_title(f"Level {level} - Data")
    labels = full_mg.meta[f"lvl{level}_labels_side"]
    estimator = model(directed=True, loops=True)
    uni_labels, inv = np.unique(labels, return_inverse=True)
    estimator.fit(adj, inv)
    p_hat = estimator.p_mat_
    sample_adj = np.squeeze(estimator.sample())
    # p_hat[p_hat == 0] = 1 / p_hat.size
    ax = fig.add_subplot(gs[3:, level + 4])
    _, _, top, _ = adjplot(
        sample_adj,
        ax=ax,
        plot_type="scattermap",
        sizes=(0.5, 0.5),
        sort_class=["hemisphere"] + level_names[: level + 1],
        item_order=["merge_class_sf_order", "merge_class", "sf"],
        class_order="sf",
        meta=full_mg.meta,
        palette=CLASS_COLOR_DICT,
        colors="merge_class",
        ticks=False,
        gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
    )
    top.set_title(f"Level {level} - DCSBM sample")

full_mg = full_mg.sort_values(["hemisphere", "Pair ID"], ascending=True)
meta = full_mg.meta
adj = full_mg.adj
lp_inds = np.arange(n_pairs)
rp_inds = np.arange(n_pairs) + n_pairs
n_levels = 10

pairs = np.unique(meta["Pair ID"])
p_same_clusters = []
p_same_chance = []
rows = []
n_shuffles = 10
for l in range(n_levels + 1):
    n_same = 0
    pred_labels = meta[f"lvl{l}_labels"].values.copy()
    left_labels = pred_labels[lp_inds]
    right_labels = pred_labels[rp_inds]
    n_same = (left_labels == right_labels).sum()
    p_same = n_same / len(pairs)
    rows.append(dict(p_same_cluster=p_same, labels="True", level=l))

    # look at random chance
    for i in range(n_shuffles):
        np.random.shuffle(pred_labels)
        left_labels = pred_labels[lp_inds]
        right_labels = pred_labels[rp_inds]
        n_same = (left_labels == right_labels).sum()
        p_same = n_same / len(pairs)
        rows.append(dict(p_same_cluster=p_same, labels="Shuffled", level=l))

plot_df = pd.DataFrame(rows)
ax = fig.add_subplot(gs[:2, 0])
sns.lineplot(data=plot_df, x="level", y="p_same_cluster", ax=ax, hue="labels")
ax.set_ylabel("P same cluster")
ax.set_xlabel("Level")

rows = []

for l in range(n_levels + 1):
    labels = meta[f"lvl{l}_labels"].values
    left_adj = binarize(adj[np.ix_(lp_inds, lp_inds)])
    left_adj = remove_loops(left_adj)
    right_adj = binarize(adj[np.ix_(rp_inds, rp_inds)])
    right_adj = remove_loops(right_adj)
    for model, name in zip([DCSBMEstimator, SBMEstimator], ["DCSBM", "SBM"]):
        estimator = model(directed=True, loops=False)
        uni_labels, inv = np.unique(labels, return_inverse=True)
        estimator.fit(left_adj, inv[lp_inds])
        train_left_p = estimator.p_mat_
        train_left_p[train_left_p == 0] = 1 / train_left_p.size

        score = poisson.logpmf(left_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Same",
                test_side="Left",
                score=score,
                level=l,
                model=name,
                n_params=estimator._n_parameters(),
            )
        )
        score = poisson.logpmf(right_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Opposite",
                test_side="Right",
                score=score,
                level=l,
                model=name,
                n_params=estimator._n_parameters(),
            )
        )

        estimator = model(directed=True, loops=False)
        estimator.fit(right_adj, inv[rp_inds])
        train_right_p = estimator.p_mat_
        train_right_p[train_right_p == 0] = 1 / train_right_p.size

        score = poisson.logpmf(left_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Opposite",
                test_side="Left",
                score=score,
                level=l,
                model=name,
                n_params=estimator._n_parameters(),
            )
        )
        score = poisson.logpmf(right_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Same",
                test_side="Right",
                score=score,
                level=l,
                model=name,
                n_params=estimator._n_parameters(),
            )
        )

plot_df = pd.DataFrame(rows)

model_name = "DCSBM"
palette = sns.color_palette()
palette = [palette[2], palette[4]]
sns.set_palette(palette)
ax = fig.add_subplot(gs[2:4, 0])
sns.lineplot(
    data=plot_df[plot_df["model"] == model_name],
    hue="test",
    x="level",
    y="score",
    style="train_side",
)

ax.set_ylabel(f"{model_name} log lik.")

palette = sns.color_palette("deep")
palette = [palette[5]]
sns.set_palette(palette)
ax = fig.add_subplot(gs[4:6, 0])

# sns.lineplot(data=plot_df[plot_df["model"] == model_name], x="level", y="n_params")
# ax.set_yscale("log")
# ax.set_yticks([2.3e3, 1e4])
# ax.set_ylabel(f"{model_name} # parameters")

n_clusters = []
for l in range(n_levels + 1):
    n_clusters.append(meta[f"lvl{l}_labels"].nunique())
sns.lineplot(x=range(n_levels + 1), y=n_clusters, ax=ax)
ax.set_ylabel("Clusters per side")
ax.set_xlabel("Level")

plt.tight_layout()

stashfig(f"megafig-lowest={lowest_level}" + basename)
plt.close()

