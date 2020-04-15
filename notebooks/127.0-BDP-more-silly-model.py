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
# ## Normalize
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=4,
    n_components=None,
    embed="unscaled_ase",
    # reembed=True,
    realign=True,
    normalize=True,
)
mc.fit_candidates()

# %% [markdown]
# ##
mc.plot_model(4)
# %% [markdown]
# ##

mc.plot_model(6)

# %% [markdown]
# ##
import n_sphere

norm_embed = mc.X_
labels = meta["merge_class"].values
norm_embed_spherical = n_sphere.convert_spherical(norm_embed)
norm_embed_spherical = norm_embed_spherical[:, 1:]  # chop off R dimension
pg = pairplot(norm_embed_spherical, labels=labels, palette=CLASS_COLOR_DICT)
pg._legend.remove()


# %% [markdown]
# ##

from sklearn.cluster import AgglomerativeClustering

for k in range(2, 10):
    ag = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    pred_labels = ag.fit_predict(norm_embed)

    ax = stacked_barplot(pred_labels, labels, color_dict=CLASS_COLOR_DICT)
    ax.set_title(f"k={k}")
# %% [markdown]
# ## new
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=None,
    embed="unscaled_ase",
    reembed=True,
    realign=True,
)
mc.fit_candidates()
mc.plot_model(6)
mc.plot_model(7)
mc.select_model(6)

np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)

# %% [markdown]
# ## new

np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=None,
    embed="unscaled_ase",
    reembed=False,
    realign=True,
)
mc.fit_candidates()
mc.plot_model(6)
mc.plot_model(7)
mc.select_model(6)

np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)

# %% [markdown]
# ## new

np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=None,
    embed="unscaled_ase",
    reembed="masked",
    realign=True,
)
mc.fit_candidates()
mc.plot_model(6)
mc.plot_model(7)
mc.select_model(6)

np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)


# %% [markdown]
# ##
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=4,
    embed="ase",
    realign=True,
)
mc.fit_candidates()
mc.plot_model(6)
mc.plot_model(7)
mc.select_model(6)

#%%
np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)

# %% [markdown]
# ##
np.random.seed(8888)

mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=4,
    embed="ase",
    realign=False,
)
mc.fit_candidates()
mc.plot_model(6)
mc.select_model(6)

np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)

# %% [markdown]
# ##
np.random.seed(8888)

mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=4,
    embed="unscaled_ase",
    realign=False,
)
mc.fit_candidates()
mc.plot_model(6)
mc.select_model(6)

np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)
# %% [markdown]
# ##
np.random.seed(8888)

mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=4,
    embed="ase",
    reembed=True,
)
mc.fit_candidates()
mc.plot_model(6)
mc.select_model(6)

np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)

# # %% [markdown]
# # ##
# np.random.seed(8888)

# mc = MaggotCluster(
#     "0",
#     adj=adj,
#     meta=meta,
#     n_init=50,
#     stashfig=stashfig,
#     max_clusters=8,
#     n_components=4,
#     embed="ase",
#     reembed=True,
# )
# mc.fit_candidates()
# mc.select_model(6)

# np.random.seed(9999)
# for i, node in enumerate(mc.get_lowest_level()):
#     print(node.name)
#     print()
#     node.fit_candidates()

# sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
# for i, node in enumerate(mc.get_lowest_level()):
#     print(node.name)
#     print()
#     for k in sub_ks[i]:
#         node.plot_model(k)


# %%

# %% [markdown]
# ##

np.random.seed(8888)

mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=None,
    embed="unscaled_ase",
    realign=False,
    reembed=True,
)
mc.fit_candidates()
mc.plot_model(6)
mc.select_model(6)

# %% [markdown]
# ##
np.random.seed(9999)
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

sub_ks = [(2, 3, 4, 5), (2, 3, 4), (2, 4, 5, 6), (2, 3, 4), (2, 3, 4), (2, 3, 4, 5)]
for i, node in enumerate(mc.get_lowest_level()):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)


# %% focus on the antenal lobey cluster
at_node = mc.get_lowest_level()[2]
at_node.select_model(2)  # or 6

for i, node in enumerate(at_node.get_lowest_level()):
    print(node.name)
    print()
    node.fit_candidates()

