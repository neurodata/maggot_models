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
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob, invert_permutation
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    sankey,
    screeplot,
    remove_spines,
)


warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2020-01-16"

SAVEFIGS = True
SAVESKELS = False
SAVEOBJS = True

PTR = True
if PTR:
    ptr_type = "PTR"
else:
    ptr_type = "Raw"

brain_type = "Full Brain"
brain_type_short = "fullbrain"

GRAPH_TYPE = "Gad"
if GRAPH_TYPE == "Gad":
    graph_type = r"A $\to$ D"

N_INIT = 200

CLUSTER_METHOD = "graspy-gmm"
if CLUSTER_METHOD == "graspy-gmm":
    cluster_type = "GraspyGMM"
elif CLUSTER_METHOD == "auto-gmm":
    cluster_type = "AutoGMM"

EMBED = "LSE"
if EMBED == "LSE":
    embed_type = "LSE"

N_COMPONENTS = None


np.random.seed(23409857)

# Set up plotting constants
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=0.8)


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


def stashobj(obj, name, **kws):
    saveobj(obj, name, foldername=FNAME, save_on=SAVEOBJS, **kws)


def add_connections(x1, x2, y1, y2, color="black", alpha=0.2, ax=None):
    if ax is None:
        ax = plt.gca()
    for i in range(len(x1)):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color=color, alpha=alpha)


def quick_gridmap(adj, labels):
    gridplot(
        [adj],
        height=20,
        sizes=(10, 20),
        inner_hier_labels=labels,
        sort_nodes=False,
        palette="deep",
        legend=False,
    )


def pair_augment(mg):
    pair_df = pd.read_csv(
        "maggot_models/data/raw/Maggot-Brain-Connectome/pairs/knownpairsatround5.csv"
    )

    skeleton_labels = mg.meta.index.values

    # extract valid node pairings
    left_nodes = pair_df["leftid"].values
    right_nodes = pair_df["rightid"].values

    left_right_pairs = list(zip(left_nodes, right_nodes))

    left_nodes_unique, left_nodes_counts = np.unique(left_nodes, return_counts=True)
    left_duplicate_inds = np.where(left_nodes_counts >= 2)[0]
    left_duplicate_nodes = left_nodes_unique[left_duplicate_inds]

    right_nodes_unique, right_nodes_counts = np.unique(right_nodes, return_counts=True)
    right_duplicate_inds = np.where(right_nodes_counts >= 2)[0]
    right_duplicate_nodes = right_nodes_unique[right_duplicate_inds]

    left_nodes = []
    right_nodes = []
    for left, right in left_right_pairs:
        if left not in left_duplicate_nodes and right not in right_duplicate_nodes:
            if left in skeleton_labels and right in skeleton_labels:
                left_nodes.append(left)
                right_nodes.append(right)

    pair_nodelist = np.concatenate((left_nodes, right_nodes))
    not_paired = np.setdiff1d(skeleton_labels, pair_nodelist)
    sorted_nodelist = np.concatenate((pair_nodelist, not_paired))

    # sort the graph and metadata according to this
    sort_map = dict(zip(sorted_nodelist, range(len(sorted_nodelist))))
    inv_perm_inds = np.array(itemgetter(*skeleton_labels)(sort_map))
    perm_inds = invert_permutation(inv_perm_inds)

    mg.reindex(perm_inds)

    side_labels = mg["Hemisphere"]
    side_labels = side_labels.astype("<U2")
    for i, l in enumerate(side_labels):
        if mg.meta.index.values[i] in not_paired:
            side_labels[i] = "U" + l
    mg["Hemisphere"] = side_labels
    n_pairs = len(left_nodes)
    return mg, n_pairs


def compute_neighbors_at_k(X, n_pairs, k_max=10):
    nn = NearestNeighbors(radius=0, n_neighbors=k_max + 1)
    nn.fit(X)

    neigh_dist, neigh_inds = nn.kneighbors(X)

    is_neighbor_mat = np.zeros((X.shape[0], k_max), dtype=bool)

    modifier = n_pairs
    for i in range(2 * n_pairs):
        if i >= n_pairs:
            modifier = -n_pairs
        inds = neigh_inds[i]
        for k in range(k_max):
            if i + modifier in inds[: k + 2]:  # first neighbor is self, also need +1
                is_neighbor_mat[i, k] = True

    neighbors_at_k = np.sum(is_neighbor_mat, axis=0) / is_neighbor_mat.shape[0]
    return neighbors_at_k


def max_symmetrize(mg, n_pairs):
    """ assumes that mg is sorted
    
    Parameters
    ----------
    mg : [type]
        [description]
    n_pairs : [type]
        [description]
    """
    adj = mg.adj
    left_left_adj = adj[:n_pairs, :n_pairs]
    left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
    right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
    right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

    # max, average gives similar results
    sym_ipsi_adj = np.maximum(left_left_adj, right_right_adj)
    sym_contra_adj = np.maximum(left_right_adj, right_left_adj)

    sym_adj = adj.copy()
    sym_adj[:n_pairs, :n_pairs] = sym_ipsi_adj
    sym_adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs] = sym_ipsi_adj
    sym_adj[:n_pairs, n_pairs : 2 * n_pairs] = sym_contra_adj
    sym_adj[n_pairs : 2 * n_pairs, :n_pairs] = sym_contra_adj

    sym_mg = MetaGraph(sym_adj, mg.meta)  # did not change indices order so this ok

    return sym_mg


# %% look at several different embedding types and parameters. Compute degree to which
# pairs are close together

# Embeddings:
# ASE
#    d
# LSE
#    d
#    r
# OMNI ASE
# OMNI LSE

# Graphs:
# Sum vs A to D
# Raw vs Normalized
# PTR or no

#%%


def plot_latent_sweep(latent, n_pairs):
    for d in range(latent.shape[1]):
        dim = latent[:, d]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        data_df = pd.DataFrame()
        data_df["Label"] = n_pairs * ["Left"] + n_pairs * ["Right"]
        data_df["Latent"] = dim[: 2 * n_pairs]
        data_df["Index"] = list(range(n_pairs)) + list(range(n_pairs))
        ax = sns.scatterplot(
            data=data_df, x="Index", y="Latent", hue="Label", ax=ax, s=15
        )
        add_connections(
            range(n_pairs),
            range(n_pairs),
            dim[:n_pairs],
            dim[n_pairs : 2 * n_pairs],
            ax=ax,
            color="grey",
        )
        remove_spines(ax)
        ax.xaxis.set_major_locator(plt.FixedLocator([0]))
        ax.yaxis.set_major_locator(plt.FixedLocator([0]))
        ax.set_title(f"Dimension {d}")


def remove_cols(mat, remove_inds):
    kept_inds = list(range(mat.shape[1]))
    [kept_inds.remove(i) for i in remove_inds]
    return mat[:, kept_inds]


# %% [markdown]
# #
ad_norm_mg = load_metagraph("Gadn", BRAIN_VERSION)
ad_raw_mg = load_metagraph("Gad", BRAIN_VERSION)

# ad_norm_mg.sort_values(["Hemisphere", "Pair ID"])

# %% [markdown]
# #

norm_edges = ad_norm_mg.adj.ravel()
norm_edges = norm_edges[norm_edges != 0]
raw_edges = ad_raw_mg.adj.ravel()
raw_edges = raw_edges[raw_edges != 0]
num_removed = np.sum(raw_edges[norm_edges < 0.02])
num_kept = np.sum(raw_edges[norm_edges >= 0.02])
num_total = raw_edges.sum()
num_kept / num_total
# %% [markdown]
# #
sns.distplot(edges)
plt.figure()
x = np.sort(edges)
y = np.arange(len(x)) / float(len(x))
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(x, y)
ax.set_xlim(0, 0.2)


# %% [markdown]
# #


# heatmap(
#     ad_norm_mg.adj, sort_nodes=False, transform="binarize", figsize=(20, 20), cbar=False
# )
# heatmap(
#     ad_norm_mg.adj,
#     sort_nodes=False,
#     transform="binarize",
#     figsize=(20, 20),
#     inner_hier_labels=ad_norm_mg["Merge Class"],
#     outer_hier_labels=ad_norm_mg["Hemisphere"],
#     hier_label_fontsize=5,
# )

threshold = 0.0
adj = ad_norm_mg.adj
adj[adj < threshold] = 0
mod_mg = MetaGraph(adj, ad_norm_mg.meta)

# check pairs

uni_pairs, counts = np.unique(mod_mg["Pair ID"], return_counts=True)
assert (counts[1:] == 2).all()
n_pairs = len(counts) - 1
mod_mg = max_symmetrize(mod_mg, n_pairs)

heatmap(
    mod_mg.adj,
    sort_nodes=False,
    transform="binarize",
    figsize=(20, 20),
    inner_hier_labels=mod_mg["Class 1"],
    outer_hier_labels=mod_mg["Hemisphere"],
    hier_label_fontsize=5,
)


# %% [markdown]
# #
# ad_norm_mg, n_pairs = pair_augment(ad_norm_mg)
ad_norm_mg = max_symmetrize(ad_norm_mg, n_pairs)
ad_norm_mg.make_lcc()
ad_norm_lse_latent = lse(ad_norm_mg.adj, n_components=None)
# plot_latent_sweep(ad_norm_lse_latent, n_pairs)
remove_inds = [2, 7, 10, 15]
ad_norm_lse_latent = remove_cols(ad_norm_lse_latent, remove_inds)
# %% [markdown]
# #
ad_raw_mg = load_metagraph("Gad")
ad_raw_mg, n_pairs = pair_augment(ad_raw_mg)
ad_raw_mg = max_symmetrize(ad_raw_mg, n_pairs)
ad_raw_mg.make_lcc()
ad_raw_lse_latent = lse(ad_raw_mg.adj, n_components=None)
# plot_latent_sweep(ad_raw_lse_latent, n_pairs)
remove_inds = [3, 5, 10, 12]
ad_raw_lse_latent = remove_cols(ad_raw_lse_latent, remove_inds)

# %% [markdown]
# #
ad_norm_ase_latent = ase(ad_norm_mg.adj, n_components=None)
# plot_latent_sweep(ad_norm_ase_latent, n_pairs)
remove_inds = [2, 9]
ad_norm_ase_latent = remove_cols(ad_norm_ase_latent, remove_inds)

# %% [markdown]
#
ad_raw_ase_latent = ase(ad_raw_mg.adj, n_components=None)
# plot_latent_sweep(ad_raw_ase_latent, n_pairs)
remove_inds = [2, 6, 8, 11, 15, 17]
ad_raw_ase_latent = remove_cols(ad_raw_ase_latent, remove_inds)

# %% [markdown]
# #
embeddings = [
    (ad_norm_ase_latent, "Norm-ASE"),
    (ad_norm_lse_latent, "Norm-LSE"),
    (ad_raw_ase_latent, "Raw-ASE"),
    (ad_raw_lse_latent, "Raw-LSE"),
]

for e in embeddings:
    latent, names = e
    n_verts = latent.shape[0]
    nn = NearestNeighbors(radius=0, n_neighbors=n_verts)
    nn.fit(latent)
    neighbors = nn.kneighbors(latent, return_distance=False)
    pair_locs = []
    for p in range(n_pairs):
        pair_ind = p + n_pairs
        k_for_pair = np.where(neighbors[p, :] == pair_ind)[0][0]
        pair_locs.append(k_for_pair)
    for p in range(n_pairs, 2 * n_pairs):
        pair_ind = p - n_pairs
        k_for_pair = np.where(neighbors[p, :] == pair_ind)[0][0]
        pair_locs.append(k_for_pair)
    pair_locs = np.array(pair_locs)
    pair_locs[pair_locs == 0] = 1
    plot_df = pd.DataFrame()
    plot_df["K to find pair"] = pair_locs
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bins = np.linspace(0, 4, 50)
    sns.distplot(np.log(pair_locs), ax=ax, kde=False, norm_hist=False, bins=bins)
    ax.set_xlabel("log(K) to find pair")
    plt.title(names)

# %% [markdown]
# # Observations
# For Raw, ASE seems better
# For Norm, ASE seems better
# Overall, best seems to be Norm-ASE.

# %% [markdown]
# # Look at this embedding more closely

latent = ad_norm_lse_latent
n_components = latent.shape[1]
class_labels = ad_norm_mg["Merge Class"]
n_unique = len(np.unique(class_labels))

for dim1 in range(n_components):
    for dim2 in range(dim1 + 1, n_components):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        data_df = pd.DataFrame()
        dim1_label = f"Dim {dim1}"
        dim2_label = f"Dim {dim2}"
        data_df[dim1_label] = latent[:, dim1]
        data_df[dim2_label] = latent[:, dim2]
        data_df["Label"] = class_labels
        ax = sns.scatterplot(
            data=data_df,
            x=dim1_label,
            y=dim2_label,
            palette=cc.glasbey_light[:n_unique],
            hue="Label",
            ax=ax,
            s=20,
        )
        add_connections(
            latent[:n_pairs, dim1],
            latent[n_pairs : 2 * n_pairs, dim1],
            latent[:n_pairs, dim2],
            latent[n_pairs : 2 * n_pairs, dim2],
            ax=ax,
            color="grey",
        )
        remove_spines(ax)
        ax.xaxis.set_major_locator(plt.FixedLocator([0]))
        ax.yaxis.set_major_locator(plt.FixedLocator([0]))
        stashfig(f"trunc-max-sym-dim{dim1}-vs-dim{dim2}")
        plt.close()


# %% [markdown]
# # Fuck all this, do Omni
from graspy.embed import AdjacencySpectralEmbed
from graspy.utils import augment_diagonal, pass_to_ranks

ad_norm_mg = load_metagraph("Gadn")
ad_norm_mg, n_pairs = pair_augment(ad_norm_mg)
side_labels = ad_norm_mg["Hemisphere"]
n_pairs = np.count_nonzero(side_labels == "L")
assert (side_labels[:n_pairs] == "L").all()
assert (side_labels[n_pairs : 2 * n_pairs] == "R").all()

# %% [markdown]
# #

adj = ad_norm_mg.adj
left_left_adj = adj[:n_pairs, :n_pairs]
left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]
sym_ipsi_adj = np.maximum(left_left_adj, right_right_adj)
sym_contra_adj = np.maximum(left_right_adj, right_left_adj)

n_components = 4
ipsi_latent = ase(sym_ipsi_adj, n_components)
contra_latent = ase(sym_contra_adj, n_components)
merge_latent = np.concatenate((ipsi_latent, contra_latent), axis=-1)


# omni_mat = np.empty((2 * n_pairs, 2 * n_pairs))
# avg = (sym_ipsi_adj + sym_contra_adj) / 2
# omni_mat[:n_pairs, :n_pairs] = sym_ipsi_adj
# omni_mat[n_pairs:, n_pairs:] = sym_contra_adj
# omni_mat[:n_pairs, n_pairs:] = avg
# omni_mat[n_pairs:, :n_pairs] = avg
# side_labels = np.array(n_pairs * ["Ipsi"] + n_pairs * ["Contra"])
# quick_gridmap(omni_mat, side_labels)


# %% [markdown]
# #
n_components = merge_latent.shape[1]
latent = merge_latent
class_labels = ad_norm_mg["Merge Class"][:n_pairs]  # assume L = R
n_unique = len(np.unique(class_labels))

for dim1 in range(n_components):
    for dim2 in range(dim1 + 1, n_components):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        data_df = pd.DataFrame()
        dim1_label = f"Dim {dim1}"
        dim2_label = f"Dim {dim2}"
        data_df[dim1_label] = latent[:, dim1]
        data_df[dim2_label] = latent[:, dim2]
        data_df["Label"] = class_labels
        ax = sns.scatterplot(
            data=data_df,
            x=dim1_label,
            y=dim2_label,
            palette=cc.glasbey_light[:n_unique],
            hue="Label",
            ax=ax,
            s=20,
        )
        remove_spines(ax)
        ax.xaxis.set_major_locator(plt.FixedLocator([0]))
        ax.yaxis.set_major_locator(plt.FixedLocator([0]))
        stashfig(f"trunc-max-sym-dim{dim1}-vs-dim{dim2}")
        plt.close()

