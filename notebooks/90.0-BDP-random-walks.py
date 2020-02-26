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

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def pairwise_sparse_jaccard_distance(X, Y=None):
    """
    Computes the Jaccard distance between two sparse matrices or between all pairs in
    one sparse matrix.

    Args:
        X (scipy.sparse.csr_matrix): A sparse matrix.
        Y (scipy.sparse.csr_matrix, optional): A sparse matrix.

    Returns:
        numpy.ndarray: A similarity matrix.

    REF https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix
    """

    if Y is None:
        Y = X

    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)

    intersect = X.dot(Y.T)

    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = (xx + yy).T - intersect

    return (1 - intersect / union).A


def generate_random_walks(prob_mat, from_inds, out_inds, n_walks=100, max_walk=25):
    n_verts = len(prob_mat)
    dead_inds = np.where(prob_mat.sum(axis=1) == 0)[0]
    stop_reasons = np.zeros(3)
    sm_paths = []
    visit_orders = {i: [] for i in range(n_verts)}
    for s in from_inds:
        for n in range(n_walks):
            curr_ind = s
            n_steps = 0
            path = [s]
            visit_orders[s].append(len(path))
            while (
                (curr_ind not in out_inds)
                and (n_steps <= max_walk)
                and (curr_ind not in dead_inds)
            ):
                next_ind = np.random.choice(n_verts, p=prob_mat[curr_ind])
                n_steps += 1
                curr_ind = next_ind
                path.append(curr_ind)
                visit_orders[curr_ind].append(len(path))
            if curr_ind in out_inds:
                stop_reasons[0] += 1
                sm_paths.append(path)
            if curr_ind in dead_inds:
                stop_reasons[1] += 1
            if n_steps > max_walk:
                stop_reasons[2] += 1

    print(stop_reasons / stop_reasons.sum())
    print(len(sm_paths))
    return sm_paths, visit_orders


#%% Load and preprocess the data

VERSION = "2020-01-29"
print(f"Using version {VERSION}")

graph_type = "G"
threshold = 1
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

out_classes = [
    "O_dVNC",
    "O_dSEZ",
    "O_IPC",
    "O_ITP",
    "O_dSEZ;FFN",
    "O_CA-LP",
    "O_dSEZ;FB2N",
]
sens_classes = ["sens"]

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
prob_mat = adj.copy()
row_sums = prob_mat.sum(axis=1)
dead_inds = np.where(row_sums == 0)[0]
row_sums[row_sums == 0] = 1
prob_mat = prob_mat / row_sums[:, np.newaxis]

n_verts = len(prob_mat)
meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))
from_inds = meta[meta["Class 1"].isin(sens_classes)]["idx"].values
out_inds = meta[meta["Class 1"].isin(out_classes)]["idx"].values

ind_map = dict(zip(meta.index, meta["idx"]))
g = nx.relabel_nodes(g, ind_map, copy=True)

out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# # generate random walks

n_walks = 100
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
# # Plot num from sensory vs num to motor
visited_meta = meta[meta["n_visits"] > 0].copy()

# fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.jointplot(
    data=visited_meta, x="median_visit", y="median_out", kind="hex", height=10
)  # s=10, alpha=0.5)

# %% [markdown]
# #
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

# %% [markdown]
# # Try with Jaccard or something
path_mat = lil_matrix((len(sm_paths), n_verts))
for i, p in enumerate(sm_paths):
    path_mat[i, p] = True
path_csr = path_mat.tocsr()

# %% [markdown]
# #

t = time.time()
jaccard_dists = pairwise_sparse_jaccard_distance(path_csr)
print(f"{time.time() - t} elapsed seconds")


# %% [markdown]
# #
t = time.time()
y = squareform(jaccard_dists)
Z = linkage(y, method="average")
print(f"{time.time() - t} elapsed seconds")

# %% [markdown]
# #


t = time.time()
agg = AgglomerativeClustering(
    n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=0.995
)
pred_labels = agg.fit_predict(jaccard_dists)
print(np.unique(pred_labels, return_counts=True))
print(f"{time.time() - t} elapsed seconds")

# %% [markdown]
# #
paths = sm_paths
path_start_labels = []
path_start_labels = [p[0] for p in paths]
path_start_labels = np.array(path_start_labels)

#%%
class_start_labels = meta.iloc[
    path_start_labels, meta.columns.get_loc("Merge Class")
].values

# %% [markdown]
# #


fig, ax = plt.subplots(1, 1, figsize=(10, 20))
stacked_barplot(pred_labels, class_start_labels, ax=ax)

# %% [markdown]
# # Start on the big visualization


# choose a cluster
# get union path graph

path_graph = nx.MultiDiGraph()
chosen_cluster = 3
path_cluster = []
for i, path in enumerate(paths):
    if pred_labels[i] == chosen_cluster:
        path_cluster.append(path)

all_nodes = list(itertools.chain.from_iterable(path_cluster))
all_nodes = np.unique(all_nodes)
path_graph.add_nodes_from(all_nodes)

for path in paths:
    path_graph.add_edges_from(nx.utils.pairwise(path))


def collapse_multigraph(multigraph):
    """REF : https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-...
        into-simple-graph-with-weighted-edges
    
    Parameters
    ----------
    multigraph : [type]
        [description]
    """
    G = nx.DiGraph()
    for u, v, data in multigraph.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


path_graph = collapse_multigraph(path_graph)

meta["median_visit"]

visit_map = dict(zip(meta["idx"].values, meta["median_visit"].values))
nx.set_node_attributes(path_graph, visit_map, name="median_visit")

class_map = dict(zip(meta["idx"].values, meta["Merge Class"].values))
nx.set_node_attributes(path_graph, class_map, name="cell_class")


def add_attributes(
    g,
    drop_neg=True,
    remove_diag=True,
    size_scaler=1,
    use_counts=False,
    use_weights=True,
    color_map=None,
):
    nodelist = list(g.nodes())

    # add spectral properties
    sym_adj = symmetrize(nx.to_numpy_array(g, nodelist=nodelist))
    n_components = 10
    latent = AdjacencySpectralEmbed(n_components=n_components).fit_transform(sym_adj)
    for i in range(n_components):
        latent_dim = latent[:, i]
        lap_map = dict(zip(nodelist, latent_dim))
        nx.set_node_attributes(g, lap_map, name=f"AdjEvec-{i}")

    # add spring layout properties
    pos = nx.spring_layout(g)
    spring_x = {}
    spring_y = {}
    for key, val in pos.items():
        spring_x[key] = val[0]
        spring_y[key] = val[1]
    nx.set_node_attributes(g, spring_x, name="Spring-x")
    nx.set_node_attributes(g, spring_y, name="Spring-y")

    # add colors
    # nx.set_node_attributes(g, color_map, name="Color")
    for node, data in g.nodes(data=True):
        c = data["cell_class"]
        color = CLASS_COLOR_DICT[c]
        data["color"] = color

    # add size attribute base on number of edges
    size_map = dict(path_graph.degree(weight="weight"))
    nx.set_node_attributes(g, size_map, name="Size")

    return g


path_graph = add_attributes(path_graph)


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
draw_networkx_nice(
    path_graph,
    "Spring-x",
    "median_visit",
    sizes="Size",
    colors="color",
    weight_scale=0.01,
    draw_labels=False,
    ax=ax,
    size_scale=0.25,
)
ax.invert_yaxis()
stashfig(f"cluster-graph-map-c{chosen_cluster}")
plt.close()

# %% [markdown]
# #

flat_labels = fcluster(z, 100, "maxclust")
print(np.unique(flat_labels))


# %% [markdown]
# #

R = dendrogram(z, p=4, truncate_mode="level", orientation="left")

fig, ax = plt.subplots(1, 1, figsize=(5, 10))
R = dendrogram(z, p=30, truncate_mode="lastp", orientation="left")


# %% [markdown]
# #

fig, ax = plt.subplots(1, 1, figsize=(5, 20))
R = dendrogram(z, p=100, truncate_mode="lastp", orientation="left")

# %% [markdown]
# # trying the levenshtein thingy (slow)


string_sm_paths = []
for sm in sm_paths:
    new_str = ""
    for i in sm[1:]:
        new_str += f" {i}"
    string_sm_paths.append(new_str)


def levenshtein(str_paths):
    dist_mat = np.zeros((len(str_paths), len(str_paths)))
    lev = textdistance.Levenshtein(qval=None)
    for i, sp1 in enumerate(str_paths):
        for j, sp2 in enumerate(str_paths[i + 1 :]):
            dist = lev.distance(sp1, sp2) / max(sp1.count(" "), sp2.count(" "))
            dist_mat[i, j] = dist
    dist_mat = symmetrize(dist_mat, method="triu")
    return dist_mat


dists = levenshtein(string_sm_paths[:100])

# %% [markdown]
# #
plt.plot([10000, 20000, 30000], [5.4, 43.9, 188.9])
