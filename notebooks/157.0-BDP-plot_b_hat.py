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

n_show = 7
n_row = 3
n_col = n_show
scale = 10
fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * scale, n_row * scale))
meta = full_mg.meta
for level in range(n_show):
    # TODO show adjacency
    label_name = f"lvl{level}_labels_side"
    ax = axs[0, level]
    _, _, top, _ = adjplot(
        binarize(full_adj),
        sizes=(0.5, 0.5),
        ax=ax,
        plot_type="scattermap",
        sort_class=["hemisphere"] + level_names[: level + 1],
        item_order=["merge_class_sf_order", "merge_class", "sf"],
        class_order="sf",
        meta=meta,
        palette=CLASS_COLOR_DICT,
        colors="merge_class",
        ticks=False,
        gridline_kws=dict(linewidth=0.05, color="grey", linestyle="--"),
    )
    sbm = SBMEstimator(directed=True, loops=True)
    labels, inv = np.unique(full_meta[label_name].values, return_inverse=True)
    sbm.fit(binarize(full_adj), inv)
    ax = axs[1, level]
    _, _, top, _ = adjplot(
        sbm.p_mat_,
        ax=ax,
        plot_type="heatmap",
        sort_class=["hemisphere"] + level_names[: level + 1],
        item_order=["merge_class_sf_order", "merge_class", "sf"],
        class_order="sf",
        meta=meta,
        palette=CLASS_COLOR_DICT,
        colors="merge_class",
        ticks=False,
        gridline_kws=dict(linewidth=0.05, color="grey", linestyle="--"),
        cbar_kws=dict(shrink=0.6),
    )
    # TODO show sorted by SF for leaf nodes
    bhat = sbm.block_p_
    block_sf = -signal_flow(bhat)
    meta["block_sf"] = meta[label_name].map(dict(zip(labels, block_sf)))
    ax = axs[2, level]
    _, _, top, _ = adjplot(
        sbm.p_mat_,
        ax=ax,
        plot_type="heatmap",
        sort_class=label_name,
        # item_order=["merge_class_sf_order", "merge_class", "sf"],
        class_order="block_sf",
        meta=meta,
        palette=CLASS_COLOR_DICT,
        colors="merge_class",
        ticks=False,
        gridline_kws=dict(linewidth=0.05, color="grey", linestyle="--"),
        cbar_kws=dict(shrink=0.6),
    )
plt.tight_layout()
stashfig("big-bhat-fig")
