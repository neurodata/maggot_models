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

from spherecluster import VonMisesFisherMixture
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything
from src.embed import lse, preprocess_graph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob
from src.visualization import bartreeplot, get_color_dict, get_colors, sankey, screeplot

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

ONLY_RIGHT = False
if ONLY_RIGHT:
    brain_type = "Right Hemisphere"
    brain_type_short = "righthemi"
else:
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


# Set up plotting constants
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=0.8)

# %% [markdown]
# # Take 2


def sort_arrays(inds, *args):
    outs = []
    for a in args:
        sort_a = a[inds]
        outs.append(sort_a)
    return tuple(outs)


def sort_graph_and_meta(inds, adj, *args):
    outs = []
    meta_outs = sort_arrays(inds, *args)
    adj = adj[np.ix_(inds, inds)]
    outs = tuple([adj] + list(meta_outs))
    return outs


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


adj, class_labels, side_labels, pair_labels, skeleton_labels, = load_everything(
    "Gadn",
    version=BRAIN_VERSION,
    return_keys=["Merge Class", "Hemisphere", "Pair"],
    return_ids=True,
)

pair_df = pd.read_csv(
    "maggot_models/data/raw/Maggot-Brain-Connectome/pairs/knownpairsatround5.csv"
)

print(pair_df.head())

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
all_nodelist = skeleton_labels
not_paired = np.setdiff1d(all_nodelist, pair_nodelist)
sorted_nodelist = np.concatenate((pair_nodelist, not_paired))

# sort the graph and metadata according to this
sort_map = dict(zip(sorted_nodelist, range(len(sorted_nodelist))))
perm_inds = np.array(itemgetter(*skeleton_labels)(sort_map))
sort_inds = invert_permutation(perm_inds)

adj, class_labels, side_labels, pair_labels, skeleton_labels = sort_graph_and_meta(
    sort_inds, adj, class_labels, side_labels, pair_labels, skeleton_labels
)

side_labels = side_labels.astype("<U2")
for i, l in enumerate(side_labels):
    if skeleton_labels[i] in not_paired:
        side_labels[i] = "U" + l

# make the proper parts of the graph equal to the left/right averages
n_pairs = len(left_nodes)

left_left_adj = adj[:n_pairs, :n_pairs]
left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

sym_ipsi_adj = (left_left_adj + right_right_adj) / 2
sym_contra_adj = (left_right_adj + right_left_adj) / 2

sym_adj = adj.copy()
sym_adj[:n_pairs, :n_pairs] = sym_ipsi_adj
sym_adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs] = sym_ipsi_adj
sym_adj[:n_pairs, n_pairs : 2 * n_pairs] = sym_contra_adj
sym_adj[n_pairs : 2 * n_pairs, :n_pairs] = sym_contra_adj


gridplot([sym_adj], transform="binarize", height=20, sizes=(10, 20))


# %% [markdown]
# # Preprocess graph
old_n_verts = sym_adj.shape[0]
sym_adj, class_labels, side_labels = preprocess_graph(
    sym_adj, class_labels, side_labels
)
n_verts = sym_adj.shape[0]
print(f"Removed {old_n_verts - n_verts} nodes")
# %% [markdown]
# # Embedding
from src.embed import ase

n_verts = sym_adj.shape[0]
latent = ase(sym_adj, N_COMPONENTS, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)
print(f"ZG chose dimension {latent_dim} + {latent_dim}")

plot_latent = np.concatenate(
    (latent[:, :3], latent[:, latent_dim : latent_dim + 3]), axis=-1
)
pairplot(plot_latent, labels=side_labels)

# take the mean for the paired cells, making sure to add back in the unpaired cells
sym_latent = (latent[:n_pairs] + latent[n_pairs : 2 * n_pairs]) / 2
sym_latent = np.concatenate((sym_latent, latent[2 * n_pairs :]))
latent = sym_latent

# make new labels
side_labels = np.concatenate((n_pairs * ["P"], side_labels[2 * n_pairs :]))
# this is assuming that the class labels are perfectly matches left right, probs not
class_labels = np.concatenate((class_labels[:n_pairs], class_labels[2 * n_pairs :]))
# skeleton labels are weird for now


plot_latent = np.concatenate(
    (latent[:, :3], latent[:, latent_dim : latent_dim + 3]), axis=-1
)
pairplot(plot_latent, labels=side_labels)

# %% [markdown]
# #
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

latent = latent / np.linalg.norm(latent, axis=1)[:, np.newaxis]
cos_dists = pairwise_distances(latent, metric="cosine")
mds = MDS(n_components=2, metric=True, dissimilarity="precomputed")
mds_latent = mds.fit_transform(cos_dists)
scatter_df = pd.DataFrame()
scatter_df["x"] = mds_latent[:, 0]
scatter_df["y"] = mds_latent[:, 1]
scatter_df["Label"] = class_labels


true_color_dict = get_color_dict(class_labels, pal=cc.glasbey_cool)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.scatterplot(
    data=scatter_df,
    x="x",
    y="y",
    hue="Label",
    ax=ax,
    legend="full",
    hue_order=true_color_dict.keys(),
    palette=true_color_dict.values(),
)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.0)


# %% [markdown]
# #

vmm = VonMisesFisherMixture(n_clusters=2, init="spherical-k-means")
pred_labels = vmm.fit_predict(latent)
pred_color_dict = get_color_dict(pred_labels, pal=cc.glasbey_warm)
# %% [markdown]
# #
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
scatter_df["Clusters"] = pred_labels
sns.scatterplot(
    data=scatter_df,
    x="x",
    y="y",
    hue="Clusters",
    ax=ax,
    legend="full",
    hue_order=pred_color_dict.keys(),
    palette=pred_color_dict.values(),
)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.0)


# %% [markdown]
# # Separate sensory modalities
print(np.unique(class_labels))
sensory_classes = ["mPN-m", "mPN-o", "mPN;FFN-m", "tPN", "uPN", "vPN"]
sensory_map = {
    "mPN-m": "multimodal",
    "mPN-o": "olfaction",
    "mPN;FFN-m": "multimodal",
    "tPN": "thermo",
    "uPN": "olfaction",
    "vPN": "visual",
}
is_sensory = np.vectorize(lambda s: s in sensory_classes)(class_labels)
inds = np.arange(len(class_labels))
sensory_inds = inds[is_sensory]
nonsensory_inds = inds[~is_sensory]
sensory_labels = class_labels[is_sensory]
sensory_labels = np.array(itemgetter(*sensory_labels)(sensory_map))
cluster_latent = latent[~is_sensory, :]
