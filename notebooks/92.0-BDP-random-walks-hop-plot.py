# %% [markdown]
# #
import itertools
import os
import time
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import textdistance
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, ClassicalMDS, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import get_lcc, symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels
from src.visualization import (
    CLASS_COLOR_DICT,
    barplot_text,
    draw_networkx_nice,
    remove_spines,
    screeplot,
    stacked_barplot,
)
from src.traverse import generate_random_walks, to_markov_matrix

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


#%% Load and preprocess the data

VERSION = "2020-01-29"
print(f"Using version {VERSION}")

graph_type = "G"
threshold = 0
weight = "weight"
mg = load_metagraph(graph_type, VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=True,
    remove_pdiff=False,
    binarize=False,
    weight=weight,
)
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")


out_classes = ["O_dVNC"]
#     "O_dSEZ",
#     "O_IPC",
#     "O_ITP",
#     "O_dSEZ;FFN",
#     "O_CA-LP",
#     "O_dSEZ;FB2N",
# ]
sens_classes = ["sens-ORN"]

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
prob_mat = to_markov_matrix(adj)
n_verts = len(prob_mat)
meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))

#%%
class_key = "Merge Class"
from_inds = meta[meta[class_key].isin(sens_classes)]["idx"].values
out_inds = meta[meta[class_key].isin(out_classes)]["idx"].values

# rename nodes of graph for more convenient lookup
ind_map = dict(zip(meta.index, meta["idx"]))
g = nx.relabel_nodes(g, ind_map, copy=True)

out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# # generate random walks

n_walks = 1000
max_walk = 25

t = time.time()

sm_paths, visit_orders = generate_random_walks(
    prob_mat, from_inds, out_inds, n_walks=n_walks, max_walk=25
)

print(f"{time.time() - t} elapsed seconds")
# %% [markdown]
# #
out_orders = {i: [] for i in range(n_verts)}

for path in sm_paths:
    for i, n in enumerate(path):
        out_orders[n].append(len(path) - i)


# %% [markdown]
# # Figure - median visit order
meta["median_visit"] = -1
meta["n_visits"] = 0

for node_ind, visits in visit_orders.items():
    median_order = np.median(visits)
    meta.iloc[node_ind, meta.columns.get_loc("median_visit")] = median_order
    meta.iloc[node_ind, meta.columns.get_loc("n_visits")] = len(visits)

meta["median_out"] = -1
for node_ind, visits in out_orders.items():
    median_order = np.median(visits)
    meta.iloc[node_ind, meta.columns.get_loc("median_out")] = median_order

sort_class = "Merge Class"
class_rank = meta.groupby(sort_class)["median_visit"].median()
print(class_rank)
class_rank_mapped = meta[sort_class].map(class_rank)
class_rank_mapped.name = "class_rank"
if "class_rank" in meta.columns:
    meta = meta.drop("class_rank", axis=1)
meta = pd.concat((meta, class_rank_mapped), ignore_index=False, axis=1)

# %% [markdown]
# # Figures - plot marginals
sns.set_context("talk")
fc = sns.FacetGrid(
    data=meta,
    col=sort_class,
    col_wrap=10,
    col_order=class_rank.sort_values().index,
    sharey=False,
    height=6,
)
fc.map(sns.distplot, "median_visit", kde=False, norm_hist=True)


def draw_bar(data, color=None):
    ax = plt.gca()
    ax.axvline(np.median(data), color="red", linestyle="--")
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    med = np.median(data)
    ax.text(med + 3, yrange * 0.8, med, color="red")


fc.map(draw_bar, "median_visit")
fc.set(yticks=[])
fc.despine(left=True)

stashfig("rw-order-marginals")


sort_class = "Merge Class"
class_out = meta.groupby(sort_class)["median_out"].median()
class_out_mapped = meta[sort_class].map(class_out)
class_out_mapped.name = "class_out"
if "class_out" in meta.columns:
    meta = meta.drop("class_out", axis=1)
meta = pd.concat((meta, class_out_mapped), ignore_index=False, axis=1)
sns.set_context("talk")
fc = sns.FacetGrid(
    data=meta,
    col=sort_class,
    col_wrap=10,
    col_order=class_out.sort_values().index,
    sharey=False,
    height=6,
)
fc.map(sns.distplot, "median_out", kde=False, norm_hist=True)


def draw_bar(data, color=None):
    ax = plt.gca()
    ax.axvline(np.median(data), color="red", linestyle="--")
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    med = np.median(data)
    ax.text(med + 3, yrange * 0.8, med, color="red")


fc.map(draw_bar, "median_out")
fc.set(yticks=[])
fc.despine(left=True)

stashfig("rw-out-marginals")


# %% [markdown]
# # Figure - num to motor vs num to sensory
std = 0.1

visited_meta = meta[meta["n_visits"] > 0].copy()

visited_meta["median_visit"] = visited_meta["median_visit"] + np.random.normal(
    0, std, size=len(visited_meta)
)
visited_meta["median_out"] = visited_meta["median_out"] + np.random.normal(
    0, std, size=len(visited_meta)
)
visited_meta.sort_values(["class_rank", "Merge Class"], inplace=True)

plt.style.use("seaborn-white")
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.scatterplot(
    data=visited_meta,
    x="median_visit",
    y="median_out",
    s=10,
    alpha=1,
    hue="Merge Class",
    palette=CLASS_COLOR_DICT,
    ax=ax,
    linewidth=0.5,
)
ax.set_xlabel("Median hops from sensory")
ax.set_ylabel("Median hops to motor")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_position(("outward", 40))
ax.spines["bottom"].set_position(("outward", 40))
ax.set_xticks(np.arange(1, 21, 2))
ax.set_yticks(np.arange(1, 21, 2))
ax.set_xlim(0, 21)
ax.set_ylim(0, 21)
ax.set_title(
    f"Random walk hops from {sens_classes} to {out_classes},\n graph={graph_type}, threshold={threshold}"
    + f", sym_threshold={True}, weight={weight}"
)
stashfig("hops")
# %% [markdown]
# # Plot the adjacency sorted like this
sort_class = "Merge Class"
class_size = meta.groupby(sort_class).size()
class_size_mapped = meta[sort_class].map(class_size)
class_size_mapped.name = "class_size"
if "class_size" in meta.columns:
    meta = meta.drop("class_size", axis=1)
meta = pd.concat((meta, class_size_mapped), ignore_index=False, axis=1)


meta["idx"] = range(len(meta))
sort_meta = meta.sort_values(
    ["class_rank", "class_size", sort_class, "median_visit", "dendrite_input"],
    inplace=False,
)
perm_inds = sort_meta.idx.values

from graspy.utils import pass_to_ranks

data = mg.adj.copy()
data = data[np.ix_(perm_inds, perm_inds)]
data = pass_to_ranks(data)
sort_meta["idx"] = range(len(sort_meta))
first_df = sort_meta.groupby([sort_class], sort=False).first()
first_inds = list(first_df["idx"].values)
first_inds.append(len(meta) + 1)
# for the tic locs
middle_df = sort_meta.groupby([sort_class], sort=False).mean()
middle_inds = list(middle_df["idx"].values)
middle_labels = list(middle_df.index)


# %% [markdown]
# #

fig = plt.figure(figsize=(20, 20))
gs = plt.GridSpec(
    2,
    2,
    width_ratios=[0.01, 0.99],
    height_ratios=[0.01, 0.99],
    figure=fig,
    hspace=0,
    wspace=0,
)
ax = fig.add_subplot(gs[1, 1])  # this is the main
ax.axis("equal")
ax.set_aspect(1)
top_cax = fig.add_subplot(gs[0, 1])  # sharex=ax)
left_cax = fig.add_subplot(gs[1, 0])  # sharey=ax)
f = fig.add_subplot(gs[0, 0], sharey=ax)
# f.axis("square")

ax.set_xticks([])
ax.set_yticks([])
# ax.axis("off")

# %% [markdown]
# #

from src.visualization import gridmap, CLASS_COLOR_DICT

# draw the plot, could be heatmap or gridmap here
# fig, ax = plt.subplots(1, 1, figsize=(20, 20))
fig = plt.figure(figsize=(20, 20), constrained_layout=False)
gs = plt.GridSpec(
    2,
    2,
    width_ratios=[0.01, 0.99],
    height_ratios=[0.01, 0.99],
    figure=fig,
    hspace=0,
    wspace=0,
)

ax = fig.add_subplot(gs[1, 1], adjustable="box")  # this is the main
# ax.set_aspect(1)
# ax.axis("equal")
# ax.set(adjustable="box", aspect="equal")

top_cax = fig.add_subplot(gs[0, 1], adjustable="box", sharex=ax)
top_cax.set_aspect("auto")
left_cax = fig.add_subplot(gs[1, 0], adjustable="box", sharey=ax)
left_cax.set_aspect("auto")

classes = sort_meta[sort_class].values
class_colors = np.vectorize(CLASS_COLOR_DICT.get)(classes)
gridmap(data, ax=ax, sizes=(0.5, 1))


from matplotlib.colors import ListedColormap

# make colormap
uni_classes = np.unique(classes)
class_map = dict(zip(uni_classes, range(len(uni_classes))))
color_list = []
for u in uni_classes:
    color_list.append(CLASS_COLOR_DICT[u])
lc = ListedColormap(color_list)
classes = np.vectorize(class_map.get)(classes)
classes = classes.reshape(len(classes), 1)
sns.heatmap(
    classes,
    cmap=lc,
    cbar=False,
    yticklabels=False,
    xticklabels=False,
    ax=left_cax,
    square=False,
)
classes = classes.reshape(1, len(classes))
sns.heatmap(
    classes,
    cmap=lc,
    cbar=False,
    yticklabels=False,
    xticklabels=False,
    ax=top_cax,
    square=False,
)
# sns.heatmap(data, cmap="RdBu_r", ax=ax, vmin=0, center=0)

ax.axis("off")

# left_cax.set_ylim(ax.get_ylim())
# top_cax.set_xlim(ax.get_xlim())

# add tick labels
# ax.set_xticks(middle_inds)
# ax.set_xticklabels(middle_labels)
# ax.set_yticks(middle_inds)
# ax.set_yticklabels(middle_labels)

# add grid lines separating classes
for t in first_inds:
    ax.axvline(t - 0.5, color="grey", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axhline(t - 0.5, color="grey", linestyle="--", alpha=0.5, linewidth=0.5)

# modify the padding / offset every other tick
# axis = ax.xaxis
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_locator(plt.FixedLocator(middle_inds[0::2]))
#     axis.set_minor_locator(plt.FixedLocator(middle_inds[1::2]))
#     axis.set_minor_formatter(plt.FormatStrFormatter("%s"))
#     ax.tick_params(which="minor", pad=60, length=10)
#     ax.tick_params(which="major", length=5)
# ax.set_xticklabels(middle_labels[0::2])
# ax.set_xticklabels(middle_labels[1::2], minor=True)
# ax.set_yticklabels(middle_labels[0::2])
# ax.set_yticklabels(middle_labels[1::2], minor=True)
# ax.xaxis.tick_top()

# set tick size and rotation
# tick_fontsize = 8
# for tick in ax.get_xticklabels():
#     tick.set_rotation(90)
#     tick.set_fontsize(tick_fontsize)
# for tick in ax.get_xticklabels(minor=True):
#     tick.set_rotation(90)
#     tick.set_fontsize(tick_fontsize)
# for tick in ax.get_yticklabels():
#     tick.set_fontsize(tick_fontsize)
# for tick in ax.get_yticklabels(minor=True):
#     tick.set_fontsize(tick_fontsize)

# ax.set_ylabel("Cluster index")
# ax.set_xlabel("Node index")
# ax.set_title("AGMM o CMDS o Jaccard o Shortest Sensorimotor Paths")
# ax.set_aspect(1)
# plt.subplots_adjust(hspace=0, wspace=0)

stashfig("sorted-adj", dpi=300)

