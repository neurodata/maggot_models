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
mg = load_metagraph(graph_type, version="2020-04-01")
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
meta["inds"] = range(len(meta))
adj = mg.adj.copy()

# %% [markdown]
# ##
adj = pass_to_ranks(adj)

left_inds = meta[meta["left"]]["inds"].values
right_inds = meta[meta["right"]]["inds"].values
lp_inds, rp_inds = get_paired_inds(meta)

left_left_adj = adj[np.ix_(left_inds, left_inds)]
right_right_adj = adj[np.ix_(right_inds, right_inds)]
right_left_adj = adj[np.ix_(right_inds, left_inds)]
left_right_adj = adj[np.ix_(left_inds, right_inds)]

# %% [markdown]
# ##


def add_connections(x1, x2, y1, y2, color="black", alpha=0.2, linewidth=0.2, ax=None):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    if ax is None:
        ax = plt.gca()
    for i in range(len(x1)):
        ax.plot(
            [x1[i], x2[i]],
            [y1[i], y2[i]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def plot_pairs(
    X,
    # left_inds,
    # right_inds,
    labels,
    model=None,
    left_pair_inds=None,
    right_pair_inds=None,
    equal=False,
):

    n_dims = X.shape[1]

    # if colors is None:
    #     colors = sns.color_palette("tab10", n_colors=k, desat=0.7)

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
                        data.iloc[left_pair_inds.values, j],
                        data.iloc[right_pair_inds.values, j],
                        data.iloc[left_pair_inds.values, i],
                        data.iloc[right_pair_inds.values, i],
                        ax=ax,
                    )

    plt.tight_layout()
    return fig, axs


# %% [markdown]
# ##

ase = AdjacencySpectralEmbed(n_components=None, n_elbows=True)

left_left_embed = ase.fit_transform(left_left_adj)
right_right_embed = ase.fit_transform(right_right_adj)

# left_right_embed = ase.fit_transform(left_right_adj)
# right_left_embed = ase.fit_transform(right_left_adj)


#%%


class Procrustes:
    def __init__(self, method="ortho"):
        self.method = method

    def fit(self, X, Y=None, x_seeds=None, y_seeds=None):
        if Y is None and (x_seeds is not None and y_seeds is not None):
            Y = X[y_seeds]
            X = X[x_seeds]
        elif Y is not None and (x_seeds is not None or y_seeds is not None):
            ValueError("May only use one of \{Y, \{x_seeds, y_seeds\}\}")

        X = X.copy()
        Y = Y.copy()

        if self.method == "ortho":
            R = orthogonal_procrustes(X, Y)[0]
        elif self.method == "diag-ortho":
            norm_X = np.linalg.norm(X, axis=1)
            norm_Y = np.linalg.norm(Y, axis=1)
            norm_X[norm_X <= 1e-15] = 1
            norm_Y[norm_Y <= 1e-15] = 1
            X = X / norm_X[:, None]
            Y = Y / norm_Y[:, None]
            R = orthogonal_procrustes(X, Y)[0]
        else:
            raise ValueError("Invalid `method` parameter")

        self.R_ = R
        return self

    def transform(self, X, map_inds=None):
        if map_inds is not None:
            X_transform = X.copy()
            X_transform[map_inds] = X_transform[map_inds] @ self.R_
        else:
            X_transform = X @ self.R_
        return X_transform


# %% [markdown]
# ##

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g, version="2020-04-01")
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

embed_adjs = [pass_to_ranks(a) for a in adjs]
# embed_adjs = [a + 1 / a.size for a in embed_adjs]
# embed_adjs = [augment_diagonal(a) for a in embed_adjs]

# %% [markdown]
# ##


def bilateral_ase(adj):
    ase = AdjacencySpectralEmbed(n_components=None, n_elbows=2, check_lcc=False)
    ipsi_adj = adj.copy()
    ipsi_adj[np.ix_(left_inds, right_inds)] = 0
    ipsi_adj[np.ix_(right_inds, left_inds)] = 0
    ipsi_embed = ase.fit_transform(ipsi_adj)

    procrust = Procrustes()
    align_ipsi_embed = []
    for e in ipsi_embed:
        procrust.fit(e, x_seeds=lp_inds, y_seeds=rp_inds)
        align_e = procrust.transform(e, map_inds=left_inds)
        align_ipsi_embed.append(align_e)
    align_ipsi_embed = np.concatenate(align_ipsi_embed, axis=1)

    contra_adj = adj.copy()
    contra_adj[np.ix_(left_inds, left_inds)] = 0
    contra_adj[np.ix_(right_inds, right_inds)] = 0
    contra_embed = ase.fit_transform(contra_adj)

    procrust = Procrustes()
    align_contra_embed = []
    for e in contra_embed:
        procrust.fit(e, x_seeds=lp_inds, y_seeds=rp_inds)
        align_e = procrust.transform(e, map_inds=left_inds)
        align_contra_embed.append(align_e)
    align_contra_embed = np.concatenate(align_contra_embed, axis=1)
    return align_ipsi_embed, align_contra_embed


all_embeds = []
for a in embed_adjs:
    embed = bilateral_ase(a)
    all_embeds.append(embed[0])
    all_embeds.append(embed[1])
    # U, _, _ = selectSVD(embed, n_elbows=2)
    # plot_pairs(
    #     U,
    #     labels=meta["merge_class"].values,
    #     left_pair_inds=lp_inds,
    #     right_pair_inds=rp_inds,
    # )
cat_all_embeds = np.concatenate(all_embeds, axis=1)
# %% [markdown]
# ##
# align_joint_embed = np.concatenate((align_ipsi_embed, align_contra_embed), axis=1)
# U, S, V = selectSVD(align_joint_embed)
U, S, V = selectSVD(cat_all_embeds, n_elbows=4)
print(U.shape)
plt.plot(S)
# %% [markdown]
# ##
plot_pairs(
    U,
    labels=meta["merge_class"].values,
    left_pair_inds=lp_inds,
    right_pair_inds=rp_inds,
)

# %% [markdown]
# ##
from graspy.utils import symmetrize

# manifold = TSNE(metric="cosine")
# tsne_embed = tsne.fit_transform(U)
manifold = ClassicalMDS(n_components=U.shape[1] - 1, dissimilarity="precomputed")
# manifold = MDS(n_components=2, dissimilarity="precomputed")
# manifold = Isomap(n_components=2, metric="precomputed")
pdist = symmetrize(pairwise_distances(U, metric="cosine"))
manifold_embed = manifold.fit_transform(pdist)

plot_pairs(
    manifold_embed,
    labels=meta["merge_class"].values,
    left_pair_inds=lp_inds,
    right_pair_inds=rp_inds,
)

# %% [markdown]
# ##

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_df = pd.DataFrame(data=manifold_embed)
plot_df["merge_class"] = meta["merge_class"].values
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
# remove_axis(ax)
ax.axis("off")
add_connections(
    plot_df.iloc[lp_inds, 0],
    plot_df.iloc[rp_inds, 0],
    plot_df.iloc[lp_inds, 1],
    plot_df.iloc[rp_inds, 1],
    ax=ax,
)

# %% [markdown]
# ## Notes
# is it worth trying mvmds here
# is it worth doing an ASE/LSE combo
# way I am doing the procrustes now is also weird.
# maybe try making a similarity matrix for classes

metric = "euclidean"
pdists = []
for embed in all_embeds:
    pdist = pairwise_distances(embed, metric=metric)
    pdists.append(pdist)


# %% [markdown]
# ##
from mvlearn.embed import MVMDS

# %% [markdown]
# ##

mvmds = MVMDS(n_components=6)

mvmds_embed = mvmds.fit_transform(all_embeds)

# %% [markdown]
# ##
plot_pairs(
    mvmds_embed,
    labels=meta["merge_class"].values,
    left_pair_inds=lp_inds,
    right_pair_inds=rp_inds,
)
# dont like mvmds on cosine distances


# %%
from src.cluster import crossval_cluster, plot_metrics, predict, plot_cluster_pairs

results = crossval_cluster(
    mvmds_embed,
    left_inds,
    right_inds,
    min_clusters=2,
    max_clusters=20,
    n_init=25,
    left_pair_inds=lp_inds,
    right_pair_inds=rp_inds,
)

plot_metrics(results)

# %% [markdown]
# ##
metric = "bic"
k = 2
ind = results[results["k"] == k][metric].idxmax()
model = results.loc[ind, "model"]
pred = predict(mvmds_embed, left_inds, right_inds, model, relabel=False)
plot_cluster_pairs(
    mvmds_embed,
    left_inds,
    right_inds,
    model,
    meta["merge_class"].values,
    lp_inds,
    rp_inds,
)
pred_side = predict(mvmds_embed, left_inds, right_inds, model, relabel=True)

stacked_barplot(pred_side, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT)

# %% [markdown]
# ##

from src.cluster import MaggotCluster

from sklearn.model_selection import ParameterGrid

basename = "mvmds"
# params = list(ParameterGrid(param_grid))
# n_levels = 7

# mcs = []
# for p in params:
#     metric = p["metric"]
#     embed = p["embed"]
#     realign = p["realign"]
#     reembed = p["reembed"]
#     basename = f"-{p}".replace(" ", "")
#     basename = basename.replace(":", "=")
#     basename = basename.replace(",", "-")
#     basename = basename.replace("'", "")
#     print(basename)

# np.random.seed(8888)
n_levels = 8
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=25,
    stashfig=stashfig,
    min_clusters=1,
    max_clusters=3,
    X=mvmds_embed,
)

for i in range(n_levels):
    for j, node in enumerate(mc.get_lowest_level()):
        node.fit_candidates()
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

