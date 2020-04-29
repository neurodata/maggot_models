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
from graspy.embed import (
    AdjacencySpectralEmbed,
    LaplacianSpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
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
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
    set_axes_equal,
    stacked_barplot,
)
from graspy.embed import MultipleASE


warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


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

key = "Total degree"
# plot where we are cutting out nodes based on degree
degrees = mg.calculate_degrees()
q = np.quantile(degrees[key], 0.05)

# fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
# sns.distplot(np.log10(degrees["Total edgesum"]), ax=ax)
# ax.axvline(np.log10(q), linestyle="--", color="r")
# ax.set_xlabel("log10(total synapses)")

# remove low degree neurons
idx = meta[degrees[key] > q].index
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)
meta = mg.meta

adj = mg.adj
meta["inds"] = range(len(meta))

lp_inds, rp_inds = get_paired_inds(meta)
left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]

# %% Load and preprocess all graphs
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g, version="2020-04-01")
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

embed_adjs = [pass_to_ranks(a) for a in adjs]
embed_adjs = [a + 1 / a.size for a in embed_adjs]
embed_adjs = [augment_diagonal(a) for a in embed_adjs]

# %% [markdown]
# ##


def omni_procrust_svd(embed_adjs):
    omni = OmnibusEmbed(n_components=None, check_lcc=False)
    joint_embed = omni.fit_transform(embed_adjs)
    cat_embed = np.concatenate(joint_embed, axis=-1)
    # print(f"Omni concatenated embedding shape: {cat_embed.shape}")
    for e in cat_embed:
        e[left_inds] = e[left_inds] @ orthogonal_procrustes(e[lp_inds], e[rp_inds])[0]
    cat_embed = np.concatenate(cat_embed, axis=-1)
    U, S, Vt = selectSVD(cat_embed, n_elbows=3)
    return U


def ase_procrust_svd(embed_adjs):
    ase = AdjacencySpectralEmbed(n_components=None)
    all_embeds = []
    for a in embed_adjs:
        embed = ase.fit_transform(a)
        embed = np.concatenate(embed, axis=1)
        embed[left_inds] = (
            embed[left_inds] @ orthogonal_procrustes(embed[lp_inds], embed[rp_inds])[0]
        )
        print(embed.shape)
        all_embeds.append(embed)
    cat_embed = np.concatenate(all_embeds, axis=1)
    print(cat_embed.shape)
    U, S, Vt = selectSVD(cat_embed, n_elbows=3)
    return U


def mase_unscaled(embed_adjs):
    mase = MultipleASE(n_components=None, n_elbows=2, scaled=False)
    embed = mase.fit_transform(embed_adjs)
    embed = np.concatenate(embed, axis=-1)
    embed[left_inds] = (
        embed[left_inds] @ orthogonal_procrustes(embed[lp_inds], embed[rp_inds])[0]
    )
    return embed


def pairs(embed):
    pg = pairplot(
        embed,
        labels=meta["merge_class"].values,
        palette=CLASS_COLOR_DICT,
        size=20,
        alpha=0.5,
    )
    pg._legend.remove()


for method in (omni_procrust_svd, ase_procrust_svd, mase_unscaled):
    name = method.__name__
    print(name)
    joint_embed = method(embed_adjs)
    pairs(joint_embed)
    stashfig(f"joint-embed-{name}")


# %% [markdown]
# ##
U = omni_procrust_svd(embed_adjs)

# %% [markdown]
# ##
from sklearn.metrics import pairwise_distances

ase = AdjacencySpectralEmbed(n_components=None)
all_embeds = []
all_pdists = []
for a in embed_adjs:
    both_embed = ase.fit_transform(a)
    # embed = np.concatenate(embed, axis=1)
    for embed in both_embed:
        embed[left_inds] = (
            embed[left_inds] @ orthogonal_procrustes(embed[lp_inds], embed[rp_inds])[0]
        )
        print(embed.shape)
        all_embeds.append(embed)
        pdist = pairwise_distances(embed, metric="cosine")
        all_pdists.append(pdist)


from mvlearn.embed import MVMDS

mvmds = MVMDS(n_components=6)
mv_embed = mvmds.fit_transform(all_pdists)

pairs(mv_embed)

# %% [markdown]
# ##

cat_embeds = np.concatenate(all_embeds, axis=-1)
U, S, Vt = selectSVD(cat_embeds, n_elbows=4)
pairs(U)


# %% [markdown]
# ##


results = crossval_cluster(
    U, left_inds, right_inds, left_pair_inds=lp_inds, right_pair_inds=rp_inds
)


plot_metrics(results)


metric = "bic"
k = 6
ind = results[results["k"] == k][metric].idxmax()
model = results.loc[ind, "model"]
pred = predict(U, left_inds, right_inds, model, relabel=False)
plot_cluster_pairs(
    U, left_inds, right_inds, model, meta["merge_class"].values, lp_inds, rp_inds
)
stashfig("omni-svd-reduced-pairs-cluster")
# pred_side = predict(self.X_, self.left_inds, self.right_inds, model, relabel=True)


pred_side = predict(U, left_inds, right_inds, model, relabel=True)

stacked_barplot(pred_side, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT)
stashfig("omni-svd-reduced-barplot")


# %% [markdown]
# ##

# # %% [markdown]
# # ##
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    # stashfig=stashfig,
    min_clusters=2,
    max_clusters=8,
    X=U,
)

mc.fit_candidates()
mc.plot_model(6)

# %% [markdown]
# ##
mc.select_model(6)
for node in mc.get_lowest_level():
    node.fit_candidates()

# %% [markdown]
# ##
for node in mc.get_lowest_level():
    node.stashfig = None
    node.plot_model(2)
    node.plot_model(3)
    node.plot_model(4)

# %% [markdown]
# ##
ks = [3, 3, 0, 4]
for i, node in enumerate(mc.get_lowest_level()):
    node.select_model(ks[i])

for node in mc.get_lowest_level():
    node.fit_candidates()

# %% [markdown]
# ##

for node in mc.get_lowest_level():
    node.plot_model(2)
    node.plot_model(3)
    node.plot_model(4)
    node.plot_model(5)

# %% [markdown]
# ##
ks = [3, 2, 2, 2, 0, 2, 2, 2, 3]
for i, node in enumerate(mc.get_lowest_level()):
    node.select_model(ks[i])

for node in mc.get_lowest_level():
    node.fit_candidates()

# %% [markdown]
# ##

for node in mc.get_lowest_level():
    try:
        node.plot_model(2)
    except:
        pass
    try:
        node.plot_model(3)
    except:
        pass
    try:
        node.plot_model(4)
    except:
        pass
    try:
        node.plot_model(5)
    except:
        pass

# mc.select_model(3)
# for node in mc.get_lowest_level():
#     node.fit_candidates()
# # %%
# for node in mc.get_lowest_level():
#     for k in [2, 3, 4, 5, 6]:
#         node.plot_model(k)

# # %% [markdown]
# # ##
# ks = [2, 6, 2]
# for i, node in enumerate(mc.get_lowest_level()):
#     node.select_model(ks[i])

# #%%
# for i, node in enumerate(mc.get_lowest_level()):
#     print(node.name)
#     node.fit_candidates()
# #%%
# sub_ks = [
#     (2, 3, 4),
#     (2, 3, 4, 6, 7),
#     (2,),
#     (2, 8),
#     (2,),
#     (2, 3),
#     (4,),
#     (2, 3),
#     (2, 3, 4),
#     (2, 3, 4, 5),
# ]
# for node in mc.get_lowest_level():
#     for k in sub_ks[i]:
#         node.plot_model(k)

# # %% [markdown]
# # ##

# for node in mc.get_lowest_level():
#     if node.name == "0-1-2":
#         node.plot_model(4)


# %% [markdown]
# ##

# the below was all for the omni one
# # %% [markdown]
# # ##
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    # stashfig=stashfig,
    min_clusters=2,
    max_clusters=8,
    X=U,
)

mc.fit_candidates()
mc.plot_model(6)

# %% [markdown]
# ##
mc.select_model(4)
for node in mc.get_lowest_level():
    node.fit_candidates()

# %% [markdown]
# ##
for node in mc.get_lowest_level():
    node.stashfig = None
    node.plot_model(2)
    node.plot_model(3)
    node.plot_model(4)

# %% [markdown]
# ##
ks = [3, 3, 0, 4]
for i, node in enumerate(mc.get_lowest_level()):
    node.select_model(ks[i])

for node in mc.get_lowest_level():
    node.fit_candidates()

# %% [markdown]
# ##

for node in mc.get_lowest_level():
    node.plot_model(2)
    node.plot_model(3)
    node.plot_model(4)
    node.plot_model(5)

# %% [markdown]
# ##
ks = [3, 2, 2, 2, 0, 2, 2, 2, 3]
for i, node in enumerate(mc.get_lowest_level()):
    node.select_model(ks[i])

for node in mc.get_lowest_level():
    node.fit_candidates()

# %% [markdown]
# ##

for node in mc.get_lowest_level():
    try:
        node.plot_model(2)
    except:
        pass
    try:
        node.plot_model(3)
    except:
        pass
    try:
        node.plot_model(4)
    except:
        pass
    try:
        node.plot_model(5)
    except:
        pass

# mc.select_model(3)
# for node in mc.get_lowest_level():
#     node.fit_candidates()
# # %%
# for node in mc.get_lowest_level():
#     for k in [2, 3, 4, 5, 6]:
#         node.plot_model(k)

# # %% [markdown]
# # ##
# ks = [2, 6, 2]
# for i, node in enumerate(mc.get_lowest_level()):
#     node.select_model(ks[i])

# #%%
# for i, node in enumerate(mc.get_lowest_level()):
#     print(node.name)
#     node.fit_candidates()
# #%%
# sub_ks = [
#     (2, 3, 4),
#     (2, 3, 4, 6, 7),
#     (2,),
#     (2, 8),
#     (2,),
#     (2, 3),
#     (4,),
#     (2, 3),
#     (2, 3, 4),
#     (2, 3, 4, 5),
# ]
# for node in mc.get_lowest_level():
#     for k in sub_ks[i]:
#         node.plot_model(k)

# # %% [markdown]
# # ##

# for node in mc.get_lowest_level():
#     if node.name == "0-1-2":
#         node.plot_model(4)
