# %% [markdown]
# # Imports
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.embed import file_html
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, FactorRange, Legend, Span, PreText, Circle
from bokeh.palettes import Spectral4, all_palettes
from bokeh.plotting import curdoc, figure, output_file, show
from bokeh.resources import CDN
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph_shortest_path import graph_shortest_path
from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.utils import pass_to_ranks, get_lcc
from graspy.plot import degreeplot, edgeplot, gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph, load_networkx
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob, invert_permutation, meta_to_array
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

from bokeh.models import Select
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure
from scipy.linalg import orthogonal_procrustes


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True
BRAIN_VERSION = "2020-01-29"

sns.set_context("talk")

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


def compute_neighbors_at_k(X, left_inds, right_inds, k_max=10, metric="euclidean"):
    nn = NearestNeighbors(radius=0, n_neighbors=k_max + 1, metric=metric)
    nn.fit(X)
    neigh_dist, neigh_inds = nn.kneighbors(X)
    is_neighbor_mat = np.zeros((X.shape[0], k_max), dtype=bool)
    for left_ind, right_ind in zip(left_inds, right_inds):
        left_neigh_inds = neigh_inds[left_ind]
        right_neigh_inds = neigh_inds[right_ind]
        for k in range(k_max):
            if right_ind in left_neigh_inds[: k + 2]:
                is_neighbor_mat[left_ind, k] = True
            if left_ind in right_neigh_inds[: k + 2]:
                is_neighbor_mat[right_ind, k] = True

    neighbors_at_k = np.sum(is_neighbor_mat, axis=0) / is_neighbor_mat.shape[0]
    return neighbors_at_k


def get_paired_inds(meta):
    meta = meta.copy()
    meta["Original index"] = range(len(meta))
    left_paired_df = meta[(meta["Pair"] != -1) & (meta["Hemisphere"] == "L")]
    left_paired_inds = left_paired_df["Original index"].values
    pairs = left_paired_df["Pair"]
    right_paired_inds = meta.loc[pairs, "Original index"].values

    return left_paired_inds, right_paired_inds


def procrustes_match(latent, meta, randomize=False):
    left_inds = np.where(meta["Hemisphere"] == "L")
    left_paired_inds, right_paired_inds = get_paired_inds(meta)
    right_paired_inds = right_paired_inds.copy()
    if randomize:
        np.random.shuffle(right_paired_inds)

    left_paired_latent = latent[left_paired_inds]
    right_paired_latent = latent[right_paired_inds]

    R, scalar = orthogonal_procrustes(left_paired_latent, right_paired_latent)
    diff = np.linalg.norm(left_paired_latent @ R - right_paired_latent, ord="fro")

    rot_latent = latent.copy()
    rot_latent[left_inds] = latent[left_inds] @ R
    if randomize:
        return rot_latent, diff, right_paired_inds
    else:
        return rot_latent, diff


def stable_rank(A):
    f_norm = np.linalg.norm(A, ord="fro")
    spec_norm = np.linalg.norm(A, ord=2)
    return (f_norm ** 2) / (spec_norm ** 2)


def remove_missing_pairs(g):
    n_missing = 0
    for n, data in g.nodes(data=True):
        pair = data["Pair"]
        # pair_id = data["Pair ID"]
        if pair != -1:
            if pair not in g:
                g.node[n]["Pair"] = -1
                g.node[n]["Pair ID"] = -1
                n_missing += 1
    # print(f"Removed {n_missing} missing pairs")
    return g


# def get_edges(adj):
#     edgemat = np.empty((len(adj), 2 * len(adj) - 1))
#     inds = np.arange(len(adj))
#     for i in range(len(adj)):
#         mask = np.logical_or((inds == i)[np.newaxis, :], (inds == i)[:, np.newaxis])
#         edgemat[i, :] = adj[mask]
#     return edgemat


def get_edges(adj):
    edgemat = np.empty((len(adj), 2 * len(adj)))
    for i in range(len(adj)):
        edgemat[i, : len(adj)] = adj[i, :]
        edgemat[i, len(adj) :] = adj[:, i]
    return edgemat


# # %% [markdown]
# # #
# arr = np.arange(4 ** 2).reshape((4, 4))
# get_edges(arr)
# %% [markdown]
# #


graph_type = "Gad"
remove_pdiff = True
threshold_raw = False
graph_weight_key = None
plus_c = False
metric = "jaccard"

mg = load_metagraph(graph_type, BRAIN_VERSION)
print(f"{len(mg.to_edgelist())} original edges")


def preprocess(mg, remove_pdiff=True):
    n_original_verts = mg.n_verts

    if remove_pdiff:
        keep_inds = np.where(~mg["is_pdiff"])[0]
        mg = mg.reindex(keep_inds)
        print(f"Removed {n_original_verts - len(mg.meta)} partially differentiated")

    mg = mg.make_lcc()

    mg.verify(n_checks=100000, version=BRAIN_VERSION, graph_type=graph_type)

    edgelist_df = mg.to_edgelist()
    edgelist_df.rename(columns={"weight": "syn_weight"}, inplace=True)
    edgelist_df["norm_weight"] = (
        edgelist_df["syn_weight"] / edgelist_df["target dendrite_input"]
    )

    max_pair_edges = edgelist_df.groupby("edge pair ID", sort=False)["syn_weight"].max()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    edgelist_df["max_syn_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
        edge_max_weight_map
    )
    temp_df = edgelist_df[edgelist_df["edge pair ID"] == 0]
    edgelist_df.loc[temp_df.index, "max_syn_weight"] = temp_df["syn_weight"]

    max_pair_edges = edgelist_df.groupby("edge pair ID", sort=False)[
        "norm_weight"
    ].max()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    edgelist_df["max_norm_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
        edge_max_weight_map
    )
    temp_df = edgelist_df[edgelist_df["edge pair ID"] == 0]
    edgelist_df.loc[temp_df.index, "max_norm_weight"] = temp_df["norm_weight"]
    return edgelist_df


edgelist_df = preprocess(mg, remove_pdiff=True)
print(f"{len(edgelist_df)} edges after preprocessing")


def get_hemisphere_indices(mg):
    meta = mg.meta.copy()
    meta["Original index"] = range(len(meta))
    left_meta = meta[(meta["Hemisphere"] == "L") & (meta["Pair"] != -1)].copy()
    right_meta = meta[(meta["Hemisphere"] == "R") & (meta["Pair"] != -1)].copy()
    left_meta.sort_values("Pair ID", inplace=True)
    right_meta.sort_values("Pair ID", inplace=True)
    assert np.array_equal(left_meta["Pair ID"].values, right_meta["Pair ID"].values)

    left_inds = left_meta["Original index"].values
    right_inds = right_meta["Original index"].values
    return left_inds, right_inds


if threshold_raw:
    thresholds = np.linspace(0, 6, 7)
    thresh_weight_key = "max_syn_weight"
else:
    thresholds = np.linspace(0, 0.05, 10)
    thresh_weight_key = "max_norm_weight"

rows = []
fake_rows = []
for threshold in thresholds:
    print(threshold)
    # do the thresholding, process the output of that
    thresh_df = edgelist_df[edgelist_df[thresh_weight_key] > threshold].copy()
    thresh_g = nx.from_pandas_edgelist(
        thresh_df, edge_attr=True, create_using=nx.DiGraph
    )
    nx.set_node_attributes(thresh_g, mg.meta.to_dict(orient="index"))
    thresh_g = get_lcc(thresh_g)
    n_verts = len(thresh_g)
    thresh_g = remove_missing_pairs(thresh_g)
    if graph_weight_key is not None:
        thresh_mg = MetaGraph(thresh_g, weight=graph_weight_key)
        adj = thresh_mg.adj.copy()
    else:
        thresh_mg = MetaGraph(thresh_g, weight="syn_weight")  # either would work here
        adj = thresh_mg.adj.copy().astype(bool)
    meta = thresh_mg.meta

    print(f"Adjacency matrix is {adj.shape}")

    # get edges
    left_inds, right_inds = get_hemisphere_indices(thresh_mg)
    left_left_adj = adj[np.ix_(left_inds, left_inds)]
    right_right_adj = adj[np.ix_(right_inds, right_inds)]
    left_edges = get_edges(left_left_adj)
    right_edges = get_edges(right_right_adj)

    edge_mat = np.concatenate((left_edges, right_edges), axis=0)
    left_paired_inds = np.arange(0, len(left_edges))
    right_paired_inds = np.arange(len(left_edges), 2 * len(left_edges))
    neigh_probs = compute_neighbors_at_k(
        edge_mat, left_paired_inds, right_paired_inds, k_max=10, metric=metric
    )
    row = {"threshold": threshold, "n_verts": n_verts}
    for i, p in enumerate(neigh_probs):
        row[i + 1] = p
    rows.append(row)

    # do KNN experiment for random pairs
    np.random.shuffle(right_paired_inds)
    neigh_probs = compute_neighbors_at_k(
        edge_mat, left_paired_inds, right_paired_inds, k_max=10, metric=metric
    )
    fake_row = row.copy()
    for i, p in enumerate(neigh_probs):
        fake_row[i + 1] = p
    fake_rows.append(fake_row)

res_df = pd.DataFrame(rows)
fake_res_df = pd.DataFrame(fake_rows)

title = f"{graph_type}, metric={metric}, graph_weight={graph_weight_key}"
base_save = f"-{graph_type}-m{metric}-w{graph_weight_key}-t{thresh_weight_key}"

knn_df = pd.melt(
    res_df.drop(["n_verts"], axis=1),
    id_vars=["threshold"],
    var_name="K",
    value_name="P(Pair w/in KNN)",
)
fake_knn_df = pd.melt(
    fake_res_df.drop(["n_verts"], axis=1),
    id_vars=["threshold"],
    var_name="K",
    value_name="P(Pair w/in KNN)",
)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(
    x="threshold",
    y="P(Pair w/in KNN)",
    data=knn_df,
    hue="K",
    palette=sns.color_palette("Reds", knn_df["K"].nunique()),
)
sns.lineplot(
    x="threshold",
    y="P(Pair w/in KNN)",
    data=fake_knn_df,
    hue="K",
    palette=sns.color_palette("Blues", knn_df["K"].nunique()),
)
plt.legend(bbox_to_anchor=(1.08, 1), loc=2, borderaxespad=0.0)
ax.set_title(title)
remove_spines(ax, keep_corner=True)
stashfig(f"threshold-vs-knn" + base_save)


# # %%
# thresh_mg_norm = MetaGraph(thresh_g, weight="norm_weight")
# norm_adj = thresh_mg_norm.adj
# thresh_mg_syn = MetaGraph(thresh_g, weight="syn_weight")
# syn_adj = thresh_mg_syn.adj

# # # %% [markdown]
# # # #
# from scipy.spatial.distance import hamming, jaccard

# print(hamming(syn_adj.ravel(), norm_adj.ravel()))
# print
# syn_mask = syn_adj.ravel() > 0
# norm_mask = norm_adj.ravel() > 0
# diff = syn_mask != norm_mask
# print(diff.sum())
# print(hamming(syn_mask, norm_mask))
# print(jaccard(syn_mask, norm_mask))
# # # %%
# elist = list(thresh_g.edges(data=True))
# for e in elist:
#     source, target, data = e
#     syn_weight = data["syn_weight"]
#     norm_weight = data["norm_weight"]
#     if (syn_weight > 0) != (norm_weight > 0):
#         print(source)
#         print(target)
#         print()
