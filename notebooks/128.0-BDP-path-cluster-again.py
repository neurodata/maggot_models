# %% [markdown]
# # THE MIND OF A MAGGOT

# %% [markdown]
# ## Imports
import os
import time
import warnings
from itertools import chain

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
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.traverse import Cascade, TraverseDispatcher, to_transmission_matrix
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

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name)


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
meta["inds"] = range(len(meta))
adj = mg.adj

# %% [markdown]
# ##


out_groups = [
    ("dVNC", "dVNC;CN", "dVNC;RG", "dSEZ;dVNC"),
    ("dSEZ", "dSEZ;CN", "dSEZ;LHN", "dSEZ;dVNC"),
    ("motor-PaN", "motor-MN", "motor-VAN", "motor-AN"),
    ("RG", "RG-IPC", "RG-ITP", "RG-CA-LP", "dVNC;RG"),
    ("dUnk",),
]
out_group_names = ["VNC", "SEZ" "motor", "RG", "dUnk"]
source_groups = [
    ("sens-ORN",),
    ("sens-MN",),
    ("sens-photoRh5", "sens-photoRh6"),
    ("sens-thermo",),
    ("sens-vtd",),
    ("sens-AN",),
]
source_group_names = ["Odor", "MN", "Photo", "Temp", "VTD", "AN"]
class_key = "merge_class"

sg = list(chain.from_iterable(source_groups))
og = list(chain.from_iterable(out_groups))
sg_name = "All"
og_name = "All"

print(f"Running cascades for {sg_name} and {og_name}")

from src.traverse import to_markov_matrix

np.random.seed(888)
max_hops = 10
n_init = 100
p = 0.05
traverse = Cascade
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)
transition_probs = to_markov_matrix(adj)

source_inds = meta[meta[class_key].isin(sg)]["inds"].values
out_inds = meta[meta[class_key].isin(og)]["inds"].values


# %% [markdown]
# ##

from src.traverse import RandomWalk

n_init = 1000
paths = []
path_lens = []
for s in source_inds:
    rw = RandomWalk(
        transition_probs, stop_nodes=out_inds, max_hops=10, allow_loops=False
    )
    for n in range(n_init):
        rw.start(s)
        paths.append(rw.traversal_)
        path_lens.append(len(rw.traversal_))

# %% [markdown]
# ##
for p in paths:
    path_lens.append(len(p))
sns.distplot(path_lens)

# %% [markdown]
# ##

paths_len_6 = []
for p in paths:
    if len(p) == 6:
        paths_len_6.append(p)


# %% [markdown]
# ##
from src.cluster import get_paired_inds

embedder = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
embed = embedder.fit_transform(pass_to_ranks(adj))
embed = np.concatenate(embed, axis=-1)

lp_inds, rp_inds = get_paired_inds(meta)
R, _, = orthogonal_procrustes(embed[lp_inds], embed[rp_inds])

left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]
embed[left_inds] = embed[left_inds] @ R

from sklearn.metrics import pairwise_distances

pdist = pairwise_distances(embed, metric="cosine")
# %% [markdown]
# ##
paths = paths_len_6
path_len = len(paths[0])
path_dist_mat = np.zeros((len(paths), len(paths)))
for i in range(len(paths)):
    for j in range(len(paths)):
        p1 = paths[i]
        p2 = paths[j]
        dist_sum = 0
        for t in range(path_len):
            dist = pdist[p1[t], p2[t]]
            dist_sum += dist
        path_dist_mat[i, j] = dist_sum
# %% [markdown]
# ##
sns.heatmap(path_dist_mat, square=True)

# %% [markdown]
# ##

inds = np.random.choice(len(path_dist_mat), replace=False, size=2000)
sns.clustermap(path_dist_mat[np.ix_(inds, inds)], figsize=(20, 20))

# %% [markdown]
# ##
from graspy.embed import ClassicalMDS

cmds = ClassicalMDS(dissimilarity="precomputed", n_components=10)

path_embed = cmds.fit_transform(path_dist_mat)

# %% [markdown]
# ##
plt.plot(cmds.singular_values_)

# %% [markdown]
# ##
from graspy.plot import pairplot

pairplot(path_embed[:, :5], alpha=0.02)

# %% [markdown]
# ##
from graspy.cluster import AutoGMMCluster

agmm = AutoGMMCluster(max_components=20, n_jobs=-2)
pred = agmm.fit_predict(path_embed[:, :5])
pairplot(path_embed[:, :5], alpha=0.02, labels=pred, palette=cc.glasbey_light)


# %% [markdown]
# ##

from sklearn.cluster import AgglomerativeClustering


# %% [markdown]
# ##

# %% [markdown]
# ##


# %% [markdown]
# ##
meta["signal_flow"] = -signal_flow(adj)
# %% [markdown]
# ##

sub_path_indicator_mat = path_indicator_mat[inds]

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
matrixplot(
    sub_path_indicator_mat,
    ax=ax,
    plot_type="scattermap",
    col_sort_class="merge_class",
    col_class_order="signal_flow",
    col_meta=meta,
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    row_sort_class=pred,
    row_ticks=False,
    sizes=(2, 2),
)


# %%
name = "122.1-BDP-silly-model-testing"
load = True
loc = f"maggot_models/notebooks/outs/{name}/csvs/stash-label-meta.csv"
if load:
    meta = pd.read_csv(loc, index_col=0)

for col in ["0_pred", "1_pred", "2_pred", "hemisphere"]:
    # meta[col] = meta[col].fillna("")
    meta[col] = meta[col].astype(str)
    meta[col] = meta[col].replace("nan", "")
    meta[col] = meta[col].str.replace(".0", "")
    # meta[col] = meta[col].astype(int).astype(str)
    # meta[col] = meta[col].fillna("")
    # vals =
    # meta[col] = meta[col].astype(int).astype(str)
    # meta[col].fillna("")

meta["lvl0_labels"] = meta["0_pred"]
meta["lvl1_labels"] = meta["0_pred"] + "-" + meta["1_pred"]
meta["lvl2_labels"] = meta["0_pred"] + "-" + meta["1_pred"] + "-" + meta["2_pred"]
meta["lvl0_labels_side"] = meta["lvl0_labels"] + meta["hemisphere"]
meta["lvl1_labels_side"] = meta["lvl1_labels"] + meta["hemisphere"]
meta["lvl2_labels_side"] = meta["lvl2_labels"] + meta["hemisphere"]


# %%
# %% [markdown]
# ##

path_indicator_mat = np.zeros((len(paths), len(adj)), dtype=int)
for i, p in enumerate(paths):
    for j, visit in enumerate(p):
        path_indicator_mat[i, visit] = j + 1

# %% [markdown]
# ##
inds = np.random.choice(len(path_dist_mat), replace=False, size=16000)

sub_path_dist = path_dist_mat[np.ix_(inds, inds)]
ag = AgglomerativeClustering(n_clusters=100, affinity="precomputed", linkage="complete")
pred = ag.fit_predict(sub_path_dist)
sub_path_indicator_mat = path_indicator_mat[inds]

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(30, 20))
matrixplot(
    path_indicator_mat,
    ax=ax,
    plot_type="scattermap",
    col_sort_class=["lvl2_labels"],
    col_class_order="signal_flow",
    col_meta=meta,
    col_colors="merge_class",
    col_item_order=["merge_class", "signal_flow"],
    col_palette=CLASS_COLOR_DICT,
    col_ticks=False,
    row_sort_class=pred,
    # row_class_order="size",
    row_ticks=False,
    sizes=(1, 1),
    hue="weight",
    palette="Set1",
    gridline_kws=dict(linewidth=0.3, color="grey", linestyle="--"),
)
stashfig("path-indicator-map")
# %% [markdown]
# ## compute orders
mean_orders = []
for n in range(path_indicator_mat.shape[1]):
    nz = np.nonzero(path_indicator_mat[:, n])
    mean_order = np.mean(nz)
    mean_orders.append(mean_order)

meta["mean_order"] = mean_orders
# %% [markdown]
# ##
from src.visualization import palplot

fig, axs = plt.subplots(
    1, 2, figsize=(30, 20), gridspec_kw=dict(width_ratios=[0.95, 0.02], wspace=0.02)
)
pal = sns.color_palette("Set1", n_colors=7)
pal = pal[:5] + pal[6:]
ax = axs[0]
matrixplot(
    path_indicator_mat,
    ax=ax,
    plot_type="scattermap",
    col_sort_class=["lvl2_labels"],
    col_class_order="signal_flow",
    col_meta=meta,
    col_colors="merge_class",
    col_item_order=["merge_class", "mean_order"],
    col_palette=CLASS_COLOR_DICT,
    col_ticks=True,
    tick_rot=90,
    row_sort_class=pred,
    # row_class_order="size",
    row_ticks=True,
    sizes=(1, 1),
    hue="weight",
    palette=pal,
    gridline_kws=dict(linewidth=0.3, color="grey", linestyle="--"),
)
ax = axs[1]
palplot(pal, cmap="Set1", ax=ax)
ax.set_title("Visit order")
stashfig("path-indicator-map")

# %% [markdown] 
# ## 

uni_pred = np.unique(pred)

for up in 