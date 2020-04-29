# %% [markdown]
# ## The goal of this notebook:
# investigate regularization approaches, for now, just on the full graph
# these include
#     - truncate high degree
#     - truncate low degree
#     - plus c
#     - levina paper on row normalization
#     - others?

# %% [markdown]
# ##
import os
import time
import warnings
from itertools import chain

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, NodeMixin
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    select_dimension,
    selectSVD,
)
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import rdpg
from graspy.utils import augment_diagonal, binarize, pass_to_ranks
from src.align import Procrustes
from src.cluster import get_paired_inds
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.traverse import (
    Cascade,
    RandomWalk,
    TraverseDispatcher,
    to_markov_matrix,
    to_path_graph,
    to_transmission_matrix,
)
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    screeplot,
    set_axes_equal,
    stacked_barplot,
)

from graspy.embed import OmnibusEmbed


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
    savecsv(df, name)


graph_type = "G"


def plot_pairs(
    X, labels, model=None, left_pair_inds=None, right_pair_inds=None, equal=False
):

    n_dims = X.shape[1]

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=CLASS_COLOR_DICT,
                )
                if left_pair_inds is not None and right_pair_inds is not None:
                    add_connections(
                        data.iloc[left_pair_inds, j],
                        data.iloc[right_pair_inds, j],
                        data.iloc[left_pair_inds, i],
                        data.iloc[right_pair_inds, i],
                        ax=ax,
                    )

    plt.tight_layout()
    return fig, axs


def lateral_omni(adj, lp_inds, rp_inds):
    left_left_adj = pass_to_ranks(adj[np.ix_(lp_inds, lp_inds)])
    right_right_adj = pass_to_ranks(adj[np.ix_(rp_inds, rp_inds)])
    omni = OmnibusEmbed(n_components=3, n_elbows=2, check_lcc=False, n_iter=10)
    ipsi_embed = omni.fit_transform([left_left_adj, right_right_adj])
    ipsi_embed = np.concatenate(ipsi_embed, axis=-1)
    ipsi_embed = np.concatenate(ipsi_embed, axis=0)

    left_right_adj = pass_to_ranks(adj[np.ix_(lp_inds, rp_inds)])
    right_left_adj = pass_to_ranks(adj[np.ix_(rp_inds, lp_inds)])
    omni = OmnibusEmbed(n_components=3, n_elbows=2, check_lcc=False, n_iter=10)
    contra_embed = omni.fit_transform([left_right_adj, right_left_adj])
    contra_embed = np.concatenate(contra_embed, axis=-1)
    contra_embed = np.concatenate(contra_embed, axis=0)

    embed = np.concatenate((ipsi_embed, contra_embed), axis=1)
    return embed


# %% [markdown]
# ## look at removing low degree nodes
quantiles = np.linspace(0, 0.5, 6)
master_mg = load_metagraph(graph_type, version="2020-04-01")
for quantile in quantiles:
    print(quantile)
    mg = preprocess(
        master_mg,
        threshold=0,
        sym_threshold=False,
        remove_pdiff=True,
        binarize=False,
        weight="weight",
    )
    meta = mg.meta

    degrees = mg.calculate_degrees()
    quant_val = np.quantile(degrees["Total edgesum"], quantile)

    # remove low degree neurons
    idx = meta[degrees["Total edgesum"] > quant_val].index
    print(quant_val)
    mg = mg.reindex(idx, use_ids=True)

    # remove center neurons # FIXME
    idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
    mg = mg.reindex(idx, use_ids=True)

    mg = mg.make_lcc()
    print(len(mg))
    mg.calculate_degrees(inplace=True)
    meta = mg.meta
    meta["inds"] = range(len(meta))
    adj = mg.adj.copy()
    lp_inds, rp_inds = get_paired_inds(meta)
    left_inds = meta[meta["left"]]["inds"]

    embed = lateral_omni(adj, lp_inds, rp_inds)

    labels = np.concatenate(
        (meta["merge_class"].values[lp_inds], meta["merge_class"].values[rp_inds])
    )

    U, S, V = selectSVD(embed, n_components=6)

    plot_pairs(
        U,
        labels,
        left_pair_inds=np.arange(len(lp_inds)),
        right_pair_inds=np.arange(len(lp_inds)) + len(lp_inds),
    )
    stashfig(f"pairs-low-threshold-quantile={quantile}")
    print()


# %% [markdown]
# ## look at removing high degree nodes
quantiles = np.linspace(0, 0.5, 6)
master_mg = load_metagraph(graph_type, version="2020-04-01")
for quantile in quantiles:
    quantile = 1 - quantile
    print(quantile)
    mg = preprocess(
        master_mg,
        threshold=0,
        sym_threshold=False,
        remove_pdiff=True,
        binarize=False,
        weight="weight",
    )
    meta = mg.meta

    degrees = mg.calculate_degrees()
    quant_val = np.quantile(degrees["Total edgesum"], quantile)

    # remove high degree neurons
    idx = meta[degrees["Total edgesum"] < quant_val].index
    print(quant_val)
    mg = mg.reindex(idx, use_ids=True)

    # remove center neurons # FIXME
    idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
    mg = mg.reindex(idx, use_ids=True)

    mg = mg.make_lcc()
    print(len(mg))
    mg.calculate_degrees(inplace=True)
    meta = mg.meta
    meta["inds"] = range(len(meta))
    adj = mg.adj.copy()
    lp_inds, rp_inds = get_paired_inds(meta)
    left_inds = meta[meta["left"]]["inds"]

    embed = lateral_omni(adj, lp_inds, rp_inds)

    labels = np.concatenate(
        (meta["merge_class"].values[lp_inds], meta["merge_class"].values[rp_inds])
    )

    U, S, V = selectSVD(embed, n_components=6)

    plot_pairs(
        U,
        labels,
        left_pair_inds=np.arange(len(lp_inds)),
        right_pair_inds=np.arange(len(lp_inds)) + len(lp_inds),
    )
    stashfig(f"pairs-high-threshold-quantile={quantile}")
    plt.close()
    print()


# %% [markdown]
# ## repeat the above but with omni and induced subgraphs
# just to make sure it's not the alignment step only


quantiles = np.linspace(0, 0.5, 6)
master_mg = load_metagraph(graph_type, version="2020-04-01")
for quantile in quantiles:
    quantile = 1 - quantile
    print(quantile)
    mg = preprocess(
        master_mg,
        threshold=0,
        sym_threshold=False,
        remove_pdiff=True,
        binarize=False,
        weight="weight",
    )
    meta = mg.meta

    degrees = mg.calculate_degrees()
    quant_val = np.quantile(degrees["Total edgesum"], quantile)

    # remove high degree neurons
    idx = meta[degrees["Total edgesum"] < quant_val].index
    print(quant_val)
    mg = mg.reindex(idx, use_ids=True)

    # remove center neurons # FIXME
    idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
    mg = mg.reindex(idx, use_ids=True)

    mg = mg.make_lcc()
    print(len(mg))
    mg.calculate_degrees(inplace=True)
    meta = mg.meta
    meta["inds"] = range(len(meta))
    adj = mg.adj.copy()
    lp_inds, rp_inds = get_paired_inds(meta)
    left_inds = meta[meta["left"]]["inds"]

    # adj = pass_to_ranks(adj)
    left_left_adj = pass_to_ranks(adj[np.ix_(lp_inds, lp_inds)])
    right_right_adj = pass_to_ranks(adj[np.ix_(rp_inds, rp_inds)])
    omni = OmnibusEmbed(n_components=3, n_elbows=2, check_lcc=False, n_iter=10)
    embed = omni.fit_transform([left_left_adj, right_right_adj])
    embed = np.concatenate(embed, axis=-1)
    embed = np.concatenate(embed, axis=0)

    labels = np.concatenate(
        (meta["merge_class"].values[lp_inds], meta["merge_class"].values[rp_inds])
    )

    plot_pairs(
        embed,
        labels,
        left_pair_inds=np.arange(len(lp_inds)),
        right_pair_inds=np.arange(len(lp_inds)) + len(lp_inds),
    )
    stashfig(f"omni-pairs-high-threshold-quantile={quantile}")
    print()

# conclusions: it actually does seem like the procrustes stuff was messing things up!
# omni from now on? but this requires matched... could impute with graph matching, which
# would be janky...

# %% [markdown]
# ## should try just looking at omni on the whole thing
mg = preprocess(
    master_mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

degrees = mg.calculate_degrees()
quant_val = np.quantile(degrees["Total edgesum"], 1)

# remove high degree neurons
idx = meta[degrees["Total edgesum"] < quant_val].index
print(quant_val)
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
print(len(mg))
mg.calculate_degrees(inplace=True)
meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj.copy()
lp_inds, rp_inds = get_paired_inds(meta)
left_inds = meta[meta["left"]]["inds"]

# adj = pass_to_ranks(adj)


plot_pairs(
    embed,
    labels,
    left_pair_inds=np.arange(len(lp_inds)),
    right_pair_inds=np.arange(len(lp_inds)) + len(lp_inds),
)
stashfig("omni-all")

plot_pairs(
    U,
    labels,
    left_pair_inds=np.arange(len(lp_inds)),
    right_pair_inds=np.arange(len(lp_inds)) + len(lp_inds),
)
stashfig("omni-all-reduced")

# %% [markdown]
# ##

inds = np.concatenate((lp_inds.values, rp_inds.values))
pair_meta = meta.iloc[inds]
pair_adj = pass_to_ranks(adj[np.ix_(inds, inds)])


from src.cluster import MaggotCluster

n_levels = 8
metric = "bic"
mc = MaggotCluster(
    "0",
    meta=pair_meta,
    adj=pair_adj,
    n_init=25,
    stashfig=stashfig,
    min_clusters=1,
    max_clusters=3,
    X=U,
)
basename = "bilateral-omni"

for i in range(n_levels):
    for j, node in enumerate(mc.get_lowest_level()):
        node.fit_candidates(show_plot=False)
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
        mc.adj,
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

# %% [markdown]
# ##
uni, counts = np.unique(mc.meta["lvl6_labels"], return_counts=True)
max_ind = np.argmax(counts)
uni[max_ind]

# %% [markdown]
# ##
big_guy_meta = mc.meta[mc.meta["lvl6_labels"] == uni[max_ind]]

# %% [markdown]
# ##
sns.distplot(big_guy_meta["Total edgesum"])

# %% [markdown]
# ##
big_inds = big_guy_meta["inds"]
adjplot(
    pass_to_ranks(adj[np.ix_(big_inds, big_inds)]),
    plot_type="heatmap",
    meta=big_guy_meta,
    sort_class="merge_class",
    item_order="Total edgesum",
)

# %% [markdown]
# ##
plot_pairs(U[big_inds, :] * 1000, labels=big_guy_meta["merge_class"].values)

# %% [markdown]
# ## conclusions
# looked like the low degree nodes were getting "trapped" in a small cluster, numerically
# adjusted maggot cluster code to rescale when things get too small

# %% [markdown]
# ## redo the regularization investigations, but with omni

