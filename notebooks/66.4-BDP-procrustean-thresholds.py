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


# %% [markdown]
# #

graph_type = "Gad"
remove_pdiff = True
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

#%%

plot_embed = False
plot_n_verts = False
plot_fnorms = False

n_components = 100
threshold_raw = False
graph_weight_key = "norm_weight"
embed = "ase"
plus_c = False
metric = "cosine"

if threshold_raw:
    thresholds = np.linspace(0, 6, 7)
    thresh_weight_key = "max_syn_weight"
else:
    thresholds = np.linspace(0, 0.05, 10)
    thresh_weight_key = "max_norm_weight"

rows = []
fake_rows = []
for threshold in thresholds:
    # do the thresholding, process the output of that
    thresh_df = edgelist_df[edgelist_df[thresh_weight_key] > threshold].copy()
    thresh_g = nx.from_pandas_edgelist(
        thresh_df, edge_attr=True, create_using=nx.DiGraph
    )
    nx.set_node_attributes(thresh_g, mg.meta.to_dict(orient="index"))
    thresh_g = get_lcc(thresh_g)
    n_verts = len(thresh_g)
    thresh_g = remove_missing_pairs(thresh_g)
    thresh_mg = MetaGraph(thresh_g, weight=graph_weight_key)
    meta = thresh_mg.meta
    adj = thresh_mg.adj.copy()
    print(f"Adjacency matrix is {adj.shape}")

    # preprocess the graph and do the embedding
    adj = pass_to_ranks(adj)
    if plus_c:
        adj += np.min(adj[adj != 0])

    sr = stable_rank(adj)
    screeplot(adj, title=f"{graph_type}, threshold = {threshold}, stable rank = {sr}")

    if embed == "lse":
        latent = lse(adj, n_components, ptr=False)
    elif embed == "ase":
        latent = ase(adj, n_components, ptr=False)
    print(f"Latent matrix is {latent.shape}")

    rot_latent, diff = procrustes_match(latent, meta)

    # plot the embedding, first 2 dims only
    if plot_embed:
        plot_df = pd.DataFrame(data=rot_latent)
        plot_df["Class"] = thresh_mg["Class 1"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.scatterplot(x=0, y=1, data=plot_df, hue="Class", legend=False, ax=ax)
        ax.set_title(f"Residual F. norm = {diff}, threshold = {threshold}")

    # calculate KNN
    left_paired_inds, right_paired_inds = get_paired_inds(meta)
    neigh_probs = compute_neighbors_at_k(
        rot_latent, left_paired_inds, right_paired_inds, k_max=10, metric=metric
    )
    row = {
        "threshold": threshold,
        "Residual F-norm": diff,
        "n_verts": n_verts,
        "Norm. Resid. F-norm": diff / n_verts,
        "Stable rank": sr,
    }
    for i, p in enumerate(neigh_probs):
        row[i + 1] = p
    rows.append(row)

    # do KNN experiment for random pairs
    rand_rot_latent, rand_diff, rand_right_inds = procrustes_match(
        latent, meta, randomize=True
    )
    neigh_probs = compute_neighbors_at_k(
        rand_rot_latent, left_paired_inds, rand_right_inds, k_max=10, metric=metric
    )
    fake_row = row.copy()
    for i, p in enumerate(neigh_probs):
        fake_row[i + 1] = p
    fake_rows.append(fake_row)

res_df = pd.DataFrame(rows)
fake_res_df = pd.DataFrame(fake_rows)

title = (
    f"{graph_type}, n_comp.={n_components}, embed={embed},"
    + f" +C={plus_c}, metric={metric}, graph_weight={graph_weight_key}"
)
base_save = (
    f"-{graph_type}-ncomp{n_components}-e{embed}-pc{plus_c}-m{metric}"
    + f"-w{graph_weight_key}"
)

if plot_fnorms:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.scatterplot(
        x="threshold", y="Residual F-norm", data=res_df, legend=False, ax=ax
    )
    ax.set_title(title)
    stashfig("threshold-vs-f-norm" + base_save)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.scatterplot(
        x="threshold", y="Norm. Resid. F-norm", data=res_df, legend=False, ax=ax
    )
    ax.set_ylim((0, res_df["Norm. Resid. F-norm"].max() * 1.05))
    ax.set_title(title)
    stashfig(f"threshold-vs-norm-f-norm" + base_save)

if plot_n_verts:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.scatterplot(x="threshold", y="n_verts", data=res_df, legend=False, ax=ax)
    ax.set_title(title)
    stashfig(f"threshold-vs-n-verts" + base_save)

knn_df = pd.melt(
    res_df.drop(
        ["Residual F-norm", "n_verts", "Norm. Resid. F-norm", "Stable rank"], axis=1
    ),
    id_vars=["threshold"],
    var_name="K",
    value_name="P(Pair w/in KNN)",
)
fake_knn_df = pd.melt(
    fake_res_df.drop(
        ["Residual F-norm", "n_verts", "Norm. Resid. F-norm", "Stable rank"], axis=1
    ),
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

