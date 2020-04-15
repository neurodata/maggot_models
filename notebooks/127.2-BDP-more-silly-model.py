# %% [markdown]
# # THE MIND OF A MAGGOT

# %% [markdown]
# ## Imports
import os
import time
import warnings

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, NodeMixin
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.testing import ignore_warnings
from tqdm import tqdm

import pymaid
from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, selectSVD
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import rdpg
from graspy.utils import augment_diagonal, binarize, pass_to_ranks
from src.cluster import (
    MaggotCluster,
    add_connections,
    compute_pairedness_bipartite,
    crossval_cluster,
    fit_and_score,
    get_paired_inds,
    make_ellipses,
    plot_cluster_pairs,
    plot_metrics,
    predict,
)
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
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

PLOT_MODELS = True

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name)


# %% [markdown]
# ## Load data
# In this case we are working with `G`, the directed graph formed by summing the edge
# weights of the 4 different graph types. Preprocessing here includes removing
# partially differentiated cells, and cutting out the lowest 5th percentile of nodes in
# terms of their number of incident synapses. 5th percentile ~= 12 synapses. After this,
# the largest connected component is used.

mg = load_metagraph("G", version="2020-04-01")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

# plot where we are cutting out nodes based on degree
degrees = mg.calculate_degrees()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sns.distplot(np.log10(degrees["Total edgesum"]), ax=ax)
q = np.quantile(degrees["Total edgesum"], 0.05)
ax.axvline(np.log10(q), linestyle="--", color="r")
ax.set_xlabel("log10(total synapses)")

# remove low degree neurons
idx = meta[degrees["Total edgesum"] > q].index
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)
meta = mg.meta

adj = mg.adj
meta["inds"] = range(len(meta))

# %% [markdown]
# ##
# param_grid = {
#     "embed": ["ase", "unscaled_ase", "lse"],
#     "realign": [True, False],
#     "reembed": [True, False],
#     "metric": ["ARI", "bic", "lik"],
# }
param_grid = {
    "embed": ["ase"],
    "realign": [False],
    "reembed": [False],
    "metric": ["bic"],
}

from sklearn.model_selection import ParameterGrid

params = list(ParameterGrid(param_grid))
n_levels = 7

mcs = []
for p in params:
    metric = p["metric"]
    embed = p["embed"]
    realign = p["realign"]
    reembed = p["reembed"]
    basename = f"-{p}".replace(" ", "")
    basename = basename.replace(":", "=")
    basename = basename.replace(",", "-")
    basename = basename.replace("'", "")
    print(basename)

    np.random.seed(8888)

    mc = MaggotCluster(
        "0",
        adj=adj,
        meta=meta,
        n_init=25,
        stashfig=stashfig,
        min_clusters=2,
        max_clusters=3,
        n_components=4,
        embed=embed,
        realign=realign,
        reembed=reembed,
    )

    for i in range(n_levels):
        for j, node in enumerate(mc.get_lowest_level()):
            node.fit_candidates(plot_metrics=False)
        for j, node in enumerate(mc.get_lowest_level()):
            node.select_model(2, metric=metric)
        mc.collect_labels()

    fig, axs = plt.subplots(1, n_levels, figsize=(10 * n_levels, 30))
    for i in range(n_levels):
        ax = axs[i]
        stacked_barplot(
            mc.meta[f"lvl{i}_labels_side"],
            mc.meta["merge_class"],
            category_order=np.unique(mc.meta[f"lvl{i}_labels_side"].values),
            color_dict=CLASS_COLOR_DICT,
            norm_bar_width=False,
            ax=ax,
        )
        ax.set_yticks([])
        ax.get_legend().remove()

    stashfig(f"count-barplot-lvl{i}" + basename)
    plt.close()

    fig, axs = plt.subplots(1, n_levels, figsize=(10 * n_levels, 30))
    for i in range(n_levels):
        ax = axs[i]
        stacked_barplot(
            mc.meta[f"lvl{i}_labels_side"],
            mc.meta["merge_class"],
            category_order=np.unique(mc.meta[f"lvl{i}_labels_side"].values),
            color_dict=CLASS_COLOR_DICT,
            norm_bar_width=True,
            ax=ax,
        )
        ax.set_yticks([])
        ax.get_legend().remove()

    stashfig(f"prop-barplot-lvl{i}" + basename)
    plt.close()

    for i in range(n_levels):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        adjplot(
            adj,
            meta=mc.meta,
            sort_class=f"lvl{i}_labels_side",
            item_order="merge_class",
            plot_type="scattermap",
            sizes=(0.5, 1),
            ticks=False,
            colors="merge_class",
            ax=ax,
            palette=CLASS_COLOR_DICT,
            gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
        )
        stashfig(f"adj-lvl{i}" + basename)

    mcs.append(mc)


# %%
nodes = mc.get_lowest_level()
counts = []
for n in nodes:
    print(len(n.meta))
    counts.append(len(n.meta))
counts = np.array(counts)
big = np.max(counts)
big_ind = np.where(counts == big)[0][0]

# %% [markdown]
# ##
node = nodes[big_ind]

# get number that are paired
node.meta[node.meta["Pair"].isin(node.meta.index)]
# 52 / 215 have a pair here

# get degrees
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.distplot(node.meta["Total edgesum"], ax=ax)
sns.distplot(meta["Total edgesum"], ax=ax)
stashfig("big-guy-edgesum-joint")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.distplot(node.meta["Total degree"], ax=ax)
sns.distplot(meta["Total degree"], ax=ax)
stashfig("big-guy-degree-joint")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.distplot(node.meta["Total edgesum"], ax=ax)
stashfig("big-guy-edgesum")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.distplot(node.meta["Total degree"], ax=ax)
stashfig("big-guy-degree")

# %% [markdown]
# ##
from src.visualization import plot_neurons


start_instance()
key = "lvl6_labels"
for tp in np.unique(mc.meta[key]):
    plot_neurons(mc.meta, key, tp)
    stashfig(f"neurons-{key}-{tp}")
    plt.close()

