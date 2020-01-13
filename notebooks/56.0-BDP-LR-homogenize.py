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
# %% [markdown]
# # What is the probabiltiy that a neuron has it's nearest neighbor as opposite pair
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
from src.visualization import (bartreeplot, get_color_dict, get_colors, sankey,
                               screeplot)

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


# %% [markdown]
# # Load and add pair info

mg = load_metagraph("Gadn", version=BRAIN_VERSION)
mg, n_pairs = pair_augment(mg)

# %% [markdown]
# # Symmetrize by max, leaving in unpaired
def max_symmetrize:
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

# max, average gives similar results
sym_ipsi_adj = np.maximum(left_left_adj, right_right_adj)
sym_contra_adj = np.maximum(left_right_adj, right_left_adj)

sym_adj = adj.copy()
sym_adj[:n_pairs, :n_pairs] = sym_ipsi_adj
sym_adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs] = sym_ipsi_adj
sym_adj[:n_pairs, n_pairs : 2 * n_pairs] = sym_contra_adj
sym_adj[n_pairs : 2 * n_pairs, :n_pairs] = sym_contra_adj

sym_mg = MetaGraph(sym_adj, mg.meta)  # did not change indices order so this ok

n_verts = sym_mg.n_verts
sym_mg.make_lcc()
print(f"Removed {n_verts - sym_mg.n_verts} when finding the LCC")
adj = sym_mg.adj
side_labels = sym_mg["Hemisphere"]
class_labels = sym_mg["Merge Class"]
latent, laplacian = lse(adj, N_COMPONENTS, regularizer=None, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)

quick_gridmap(adj, side_labels)
stashfig("max-sym-adj")

plot_latent = True
if plot_latent:
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
            stashfig(f"max-sym-dim{dim1}-vs-dim{dim2}")
            plt.close()

# %% [markdown]
# # Remove dim2 and dim7
remove_inds = [2, 7, 10, 15]
kept_inds = list(range(latent.shape[1]))
[kept_inds.remove(i) for i in remove_inds]
sym_latent = latent[:, kept_inds]
print(sym_latent.shape)

replot_latent = False
if replot_latent:
    n_components = sym_latent.shape[1]
    for dim1 in range(n_components):
        for dim2 in range(dim1 + 1, n_components):
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            data_df = pd.DataFrame()
            dim1_label = f"Dim {dim1}"
            dim2_label = f"Dim {dim2}"
            data_df[dim1_label] = sym_latent[:, dim1]
            data_df[dim2_label] = sym_latent[:, dim2]
            data_df["Label"] = side_labels
            ax = sns.scatterplot(
                data=data_df, x=dim1_label, y=dim2_label, hue="Label", ax=ax, s=15
            )
            add_connections(
                sym_latent[:n_pairs, dim1],
                sym_latent[n_pairs : 2 * n_pairs, dim1],
                sym_latent[:n_pairs, dim2],
                sym_latent[n_pairs : 2 * n_pairs, dim2],
                ax=ax,
                color="grey",
            )
            remove_spines(ax)
            ax.xaxis.set_major_locator(plt.FixedLocator([0]))
            ax.yaxis.set_major_locator(plt.FixedLocator([0]))
            stashfig(f"trunc-max-sym-dim{dim1}-vs-dim{dim2}")
            plt.close()


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


sns.set_context("talk", font_scale=1.2)
k_max = 10
neighbors_at_k = compute_neighbors_at_k(latent, n_pairs, k_max=10)
plot_df = pd.DataFrame()
plot_df["Proportion w/ pair w/in KNN"] = neighbors_at_k
plot_df["K"] = range(1, k_max + 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.scatterplot(data=plot_df, x="K", y="Proportion w/ pair w/in KNN", ax=ax[0])
ax[0].set_title("Homogenized embedding")

neighbors_at_k = compute_neighbors_at_k(sym_latent, n_pairs, k_max=10)
plot_df = pd.DataFrame()
plot_df["Proportion w/ pair w/in KNN"] = neighbors_at_k
plot_df["K"] = range(1, k_max + 1)

sns.scatterplot(data=plot_df, x="K", y="Proportion w/ pair w/in KNN", ax=ax[1])
ax[1].set_title("Truncated/homogenized embedding")

plt.tight_layout()
stashfig("knn-comparison")

# %% [markdown]
# # Look at actual ARI for clustering

name_base = f"-{cluster_type}-{embed_type}-{ptr_type}-{brain_type_short}-{GRAPH_TYPE}"

left_inds = np.where(np.logical_or(side_labels == "L", side_labels == "UL"))[0]
right_inds = np.where(np.logical_or(side_labels == "R", side_labels == "UR"))[0]

left_latent = sym_latent[left_inds, :]
right_latent = sym_latent[right_inds, :]

# %% [markdown]
# #
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
side_labels = sym_mg["Hemisphere"]

print(adjusted_rand_score(left_pred_labels[:n_pairs], right_pred_labels[:n_pairs]))

stashobj(left_dc, "left_dc")
stashobj(right_dc, "right_dc")

# %% [markdown]
# # Now try to just fix a K, do GMM

cluster = GaussianCluster
N_INIT = 20
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

# %% [markdown]
# # Now cluster the whole thing and look at how oftern cells fall into the same cluster

print("Fitting DivisiveCluster model")
start = timer()
dc = DivisiveCluster(n_init=N_INIT, cluster_method=CLUSTER_METHOD)
dc.fit(latent)
end = timer()
print()
print(f"DivisiveCluster took {(end - start)/60.0} minutes to fit")
print()
dc.print_tree(print_val="bic_ratio")


# %%
pred_labels = dc.predict(latent)
class_labels = sym_mg["Merge Class"]
class_color_dict = get_color_dict(class_labels, pal=cc.glasbey_cool)
pred_color_dict = get_color_dict(pred_labels, pal=cc.glasbey_warm)

sns.set_context("talk")
fig, ax, leaf_names = bartreeplot(
    dc,
    class_labels,
    pred_labels,
    show_props=True,
    print_props=True,
    inverse_memberships=False,
    title="All",
    color_dict=class_color_dict,
    figsize=(27, 29),
)
stashfig("all-barplot")

n_same_cluster = 0
for l, r in zip(pred_labels[:n_pairs], pred_labels[n_pairs : 2 * n_pairs]):
    if l == r:
        n_same_cluster += 1
print(n_same_cluster / n_pairs)

# %% [markdown]
# #
degree_df = sym_mg.calculate_degrees()
degree_df["Cluster"] = pred_labels
degree_df["Class"] = class_labels


# %% [markdown]
# #
total_degree = degree_df["Total degree"].values
sns.distplot(total_degree[total_degree < 20], norm_hist=False, kde=False)

# %% [markdown]
# #
fg = sns.FacetGrid(
    degree_df,
    col="Cluster",
    col_wrap=6,
    sharex=True,
    sharey=False,
    hue="Cluster",
    hue_order=pred_color_dict.keys(),
    palette=pred_color_dict.values(),
    col_order=leaf_names,
)

fg.map(sns.distplot, "Total degree", norm_hist=True, kde=False)
fg.set(yticks=[], yticklabels=[])
stashfig("total-degree")

# %% [markdown]
# # compute probability of having pair w/in KNN as a function of degree
n_verts = latent.shape[0]
nn = NearestNeighbors(radius=0, n_neighbors=n_verts)
nn.fit(latent)
neighbors = nn.kneighbors(latent, return_distance=False)
pair_locs = []
for p in range(n_pairs):
    pair_ind = p + n_pairs
    print(p)
    print(neighbors[p, :])
    print(pair_ind)
    k_for_pair = np.where(neighbors[p, :] == pair_ind)[0][0]
    print(k_for_pair)
    pair_locs.append(k_for_pair)
    print()
for p in range(n_pairs, 2 * n_pairs):
    pair_ind = p - n_pairs
    k_for_pair = np.where(neighbors[p, :] == pair_ind)[0][0]
    pair_locs.append(k_for_pair)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(x=total_degree[: 2 * n_pairs], y=pair_locs)

# %% [markdown]
# #
sns.kdeplot(total_degree[: 2 * n_pairs], pair_locs)


# %%
plot_df = pd.DataFrame()
plot_df["Total degree"] = total_degree[: 2 * n_pairs]
plot_df["K to find pair"] = pair_locs
sns.jointplot(data=plot_df, x="Total degree", y="K to find pair", kind="hex", height=10)

stashfig("k-prob-degree")


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

    graphs = [left_left_adj, right_right_adj, left_right_adj, right_left_adj]

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


def plot_latent_sweep(latent, n_pairs):
    for d in range(latent.shape[1]):
        dim = latent[:, d]
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        data_df = pd.DataFrame()
        data_df["Label"] = n_pairs*["Left"] + n_pairs*["Right"]
        data_df["Latent"] = dim[:2*n_pairs]
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
        ax.set_title(f'Dimension {d}')

def remove_cols(mat, remove_inds):
    kept_inds = list(range(mat.shape[1]))
    [kept_inds.remove(i) for i in remove_inds]
    return mat[:, kept_inds]


# %% [markdown] 
# # 
ad_norm_mg = load_metagraph("Gadn")
ad_norm_mg, n_pairs = pair_augment(ad_norm_mg)
ad_norm_mg = max_symmetrize(ad_norm_mg, n_pairs)
ad_norm_mg.make_lcc()
ad_norm_lse_latent = lse(ad_norm_mg.adj, n_components=None)
plot_latent_sweep(ad_norm_lse_latent, n_pairs)
remove_inds = [2, 7, 10, 15]
ad_norm_lse_latent = remove_cols(ad_norm_lse_latent, remove_inds)
# %% [markdown] 
# # 
ad_raw_mg = load_metagraph("Gad")
ad_raw_mg, n_pairs = pair_augment(ad_raw_mg)
ad_raw_mg = max_symmetrize(ad_raw_mg, n_pairs)
ad_raw_mg.make_lcc()
ad_raw_lse_latent = lse(ad_raw_mg.adj, n_components=None)
plot_latent_sweep(ad_raw_lse_latent, n_pairs)
remove_inds = [3, 5, 10, 12]
ad_raw_lse_latent = remove_cols(ad_raw_lse_latent, remove_inds)

# %% [markdown] 
# # 
ad_norm_ase_latent = ase(ad_norm_mg.adj, n_components=None)
plot_latent_sweep(ad_norm_ase_latent, n_pairs)
remove_inds = [2, 9]
ad_norm_ase_latent = remove_cols(ad_norm_ase_latent, remove_inds)

# %% [markdown] 
# # 

ad_raw_lse_latent = lse(ad_raw_mg.adj, n_components=None)
plot_latent_sweep(ad_raw_lse_latent, n_pairs)
# remove_inds = [3, 5, 10, 12]
# ad_raw_lse_latent = remove_cols(ad_raw_lse_latent, remove_inds)
 


# %% [markdown] 
# # 
graphs = [(ad_norm_mg, "Norm"), (ad_raw_mg, "Raw")]
embeddings = [(ase, "ASE"), (lse, "LSE")]
n_components = list(range(6, 8))

param_grid = {"graph":graphs,
                "embedding":embeddings,
              "n_components":n_components}

param_list = list(ParameterGrid(param_grid))

# for p in param_list:
# %% [markdown] 
# # 


def do_embed(p):
    print(p)
    embed, embed_name = p["embedding"]
    graph, graph_name = p["graph"]
    n_components = p["n_components"]
    latent = embed(graph.adj, n_components, ptr=ptr)
    names = {"graph name": graph_name,
             "embedding name": embed_name,
             "n_components":n_components}
    return (latent, names)

outs = Parallel(n_jobs=1)(delayed(do_embed)(p) for p in param_list)

# %% [markdown] 
# # 
degree_df = ad_raw_mg.calculate_degrees()
total_degree = degree_df["Total degree"].values

# %% [markdown] 
# # 
for o in outs: 
    latent, names = o
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
    
    plot_df = pd.DataFrame()
    plot_df["Total degree"] = total_degree[: 2 * n_pairs]
    plot_df["K to find pair"] = pair_locs
    # sns.jointplot(data=plot_df, x="Total degree", y="K to find pair", kind="hex", height=10)
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    sns.distplot(pair_locs, ax=ax)
    ax.set_xlabel("K to find pair")
    plt.title(names)

# %% [markdown] 
# # 
sns.distplot(np.log(pair_locs))
