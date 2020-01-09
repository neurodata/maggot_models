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
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph
from src.embed import lse, preprocess_graph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob, invert_permutation
from src.visualization import bartreeplot, get_color_dict, get_colors, sankey, screeplot
from src.graph import MetaGraph

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2019-12-18"

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

N_COMPONENTS = 5


np.random.seed(23409857)


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


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


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


# Set up plotting constants
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=0.8)

# %% [markdown]
# # Load in data and pairs


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


mg = load_metagraph("Gadn", version=BRAIN_VERSION)
mg, n_pairs = pair_augment(mg)

# %% [markdown]
# # Look at what happens on just the raw data
preface = "heterogenous-"
n_verts = mg.n_verts
mg.make_lcc()
print(f"Removed {n_verts - mg.n_verts} when finding the LCC")
adj = mg.adj
side_labels = mg["Hemisphere"]
class_labels = mg["Merge Class"]
latent, laplacian = lse(adj, N_COMPONENTS, regularizer=None, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)

quick_gridmap(adj, side_labels)
stashfig(preface + "adj")

n_components = latent.shape[1]
for dim1 in range(n_components):
    for dim2 in range(dim1 + 1, n_components):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        data_df = pd.DataFrame()
        dim1_label = f"Dim {dim1}"
        dim2_label = f"Dim {dim2}"
        data_df[dim1_label] = latent[:, dim1]
        data_df[dim2_label] = latent[:, dim2]
        data_df["Label"] = side_labels
        ax = sns.scatterplot(
            data=data_df, x=dim1_label, y=dim2_label, hue="Label", ax=ax, s=15
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
        stashfig(preface + f"lines-dim{dim1}-vs-dim{dim2}")

# %% [markdown]
# # Look at what happens when we symmetrize, leaving in unpaired

# make the proper parts of the graph equal to the left/right averages
preface = "sym-max-"
mg = load_metagraph("Gad", version=BRAIN_VERSION)
mg, n_pairs = pair_augment(mg)
adj = mg.adj
left_left_adj = adj[:n_pairs, :n_pairs]
left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

graphs = [left_left_adj, right_right_adj, left_right_adj, right_left_adj]
names = ["LL", "RR", "LR", "RL"]
for g, n in zip(graphs, names):
    print(n)
    print(f"Synapses: {g.sum()}")
    print(f"Edges: {np.count_nonzero(g)}")
    print(f"Sparsity: {np.count_nonzero(g) / g.size}")
    print()
# %% [markdown]
# #
# take the average of left and right. this could also be max
# average
# sym_ipsi_adj = (left_left_adj + right_right_adj) / 2
# sym_contra_adj = (left_right_adj + right_left_adj) / 2

# max
sym_ipsi_adj = np.maximum(left_left_adj, right_right_adj)
sym_contra_adj = np.maximum(left_right_adj, right_left_adj)

sym_adj = adj.copy()
sym_adj[:n_pairs, :n_pairs] = sym_ipsi_adj
sym_adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs] = sym_ipsi_adj
sym_adj[:n_pairs, n_pairs : 2 * n_pairs] = sym_contra_adj
sym_adj[n_pairs : 2 * n_pairs, :n_pairs] = sym_contra_adj

full_sym_mg = MetaGraph(sym_adj, mg.meta)

n_verts = full_sym_mg.n_verts
full_sym_mg.make_lcc()
print(f"Removed {n_verts - full_sym_mg.n_verts} when finding the LCC")
adj = full_sym_mg.adj
side_labels = full_sym_mg["Hemisphere"]
class_labels = full_sym_mg["Merge Class"]
latent, laplacian = lse(adj, N_COMPONENTS, regularizer=None, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)

quick_gridmap(adj, side_labels)
stashfig(preface + "adj")

n_components = latent.shape[1]
for dim1 in range(n_components):
    for dim2 in range(dim1 + 1, n_components):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        data_df = pd.DataFrame()
        dim1_label = f"Dim {dim1}"
        dim2_label = f"Dim {dim2}"
        data_df[dim1_label] = latent[:, dim1]
        data_df[dim2_label] = latent[:, dim2]
        data_df["Label"] = side_labels
        ax = sns.scatterplot(
            data=data_df, x=dim1_label, y=dim2_label, hue="Label", ax=ax, s=15
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
        stashfig(preface + f"lines-dim{dim1}-vs-dim{dim2}")

# %% [markdown]
# #


# %% [markdown]
# # Try doing only the symmetrized parts
preface = "only-sym-avg-"
sym_adj = sym_adj[: 2 * n_pairs, : 2 * n_pairs]
only_sym_mg = MetaGraph(sym_adj, mg.meta.iloc[: 2 * n_pairs, :])
n_verts = only_sym_mg.n_verts
only_sym_mg.make_lcc()
print(f"Removed {n_verts - only_sym_mg.n_verts} when finding the LCC")
adj = only_sym_mg.adj
side_labels = only_sym_mg["Hemisphere"]
class_labels = only_sym_mg["Merge Class"]
latent, laplacian = lse(adj, N_COMPONENTS, regularizer=None, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)

quick_gridmap(adj, side_labels)
stashfig(preface + "adj")

n_components = latent.shape[1]
for dim1 in range(n_components):
    for dim2 in range(dim1 + 1, n_components):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        data_df = pd.DataFrame()
        dim1_label = f"Dim {dim1}"
        dim2_label = f"Dim {dim2}"
        data_df[dim1_label] = latent[:, dim1]
        data_df[dim2_label] = latent[:, dim2]
        data_df["Label"] = side_labels
        ax = sns.scatterplot(
            data=data_df, x=dim1_label, y=dim2_label, hue="Label", ax=ax, s=15
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
        stashfig(preface + f"lines-dim{dim1}-vs-dim{dim2}")


# %% [markdown]
# # Look at actual ARI for clustering

from src.cluster import DivisiveCluster
from graspy.cluster import GaussianCluster, AutoGMMCluster

brain_type_short = "leftlatent"
name_base = f"-{cluster_type}-{embed_type}-{ptr_type}-{brain_type_short}-{GRAPH_TYPE}"

mg = load_metagraph("Gadn", version=BRAIN_VERSION)
mg, _ = pair_augment(mg)
n_verts = mg.n_verts
mg.make_lcc()
print(f"Removed {n_verts - mg.n_verts} when finding the LCC")
adj = mg.adj
side_labels = mg["Hemisphere"]
class_labels = mg["Merge Class"]
latent, laplacian = lse(adj, N_COMPONENTS, regularizer=None, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)

left_inds = np.where(np.logical_or(side_labels == "L", side_labels == "UL"))[0]
right_inds = np.where(np.logical_or(side_labels == "R", side_labels == "UR"))[0]

left_latent = latent[left_inds, :]
right_latent = latent[right_inds, :]

base = f"maggot_models/notebooks/outs/{FNAME}/objs/"
filename = base + "dc" + name_base + ".pickle"
# clean_start = True
# if os.path.isfile(filename) and not clean_start:
#     print("Attempting to load file")
#     with open(filename, "rb") as f:
#         dc = pickle.load(f)
#     print(f"Loaded file from {filename}")
# else:
print("Fitting DivisiveCluster model")
start = timer()
left_dc = DivisiveCluster(n_init=N_INIT, cluster_method=CLUSTER_METHOD)
left_dc.fit(left_latent)
end = end = timer()
print()
print(f"DivisiveCluster took {(end - start)/60.0} minutes to fit")
print()
left_dc.print_tree(print_val="bic_ratio")

print("Fitting DivisiveCluster model")
start = timer()
right_dc = DivisiveCluster(n_init=N_INIT, cluster_method=CLUSTER_METHOD)
right_dc.fit(right_latent)
end = end = timer()
print()
print(f"DivisiveCluster took {(end - start)/60.0} minutes to fit")
print()
right_dc.print_tree(print_val="bic_ratio")


# %% [markdown]
# #

left_pred_labels = left_dc.predict(left_latent)
right_pred_labels = right_dc.predict(right_latent)

class_labels = mg["Merge Class"]
class_color_dict = get_color_dict(class_labels, pal=cc.glasbey_cool)
left_color_dict = get_color_dict(left_pred_labels, pal=cc.glasbey_warm)
right_color_dict = get_color_dict(right_pred_labels, pal=cc.glasbey_warm)

bartreeplot(
    left_dc,
    class_labels[left_inds],
    left_pred_labels,
    show_props=True,
    print_props=True,
    inverse_memberships=False,
    title="Left",
    color_dict=class_color_dict,
)
stashfig("left-barplot")

bartreeplot(
    right_dc,
    class_labels[right_inds],
    right_pred_labels,
    show_props=True,
    print_props=True,
    inverse_memberships=False,
    title="Right",
    color_dict=class_color_dict,
)
stashfig("right-barplot")

# %% [markdown]
# #
n_pairs = np.count_nonzero(side_labels == "L")
side_labels = mg["Hemisphere"]
# np.unique(side_labels[left_inds][:n_pairs])

from sklearn.metrics import adjusted_rand_score

print(adjusted_rand_score(left_pred_labels[:n_pairs], right_pred_labels[:n_pairs]))

stashobj(left_dc, "left_dc")
stashobj(right_dc, "right_dc")

# %% [markdown]
# # Now try to just fix a K, do GMM

cluster = GaussianCluster
N_INIT = 200
aris = []
for k in range(2, 12):
    gc = cluster(
        min_components=k, max_components=k, covariance_type="all", n_init=N_INIT
    )
    left_pred_labels = gc.fit_predict(left_latent)
    right_pred_labels = gc.fit_predict(right_latent)
    ari = adjusted_rand_score(left_pred_labels[:n_pairs], right_pred_labels[:n_pairs])
    print(ari)
    aris.append(ari)
    print(gc.model_.covariance_type)

# %% [markdown]
# #
plt.plot(aris)

