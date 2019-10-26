#%% Load
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.embed import OmnibusEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import augment_diagonal, pass_to_ranks
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spherecluster import SphericalKMeans

from src.data import load_networkx
from src.utils import meta_to_array, savefig
from src.visualization import incidence_plot

plt.style.use("seaborn-white")
sns.set_palette("deep")


def calc_edge_agreement(edges_base, edges_other):
    no_edge_mask = edges_other == 0
    missing_in_other = np.count_nonzero(edges_base[no_edge_mask])
    return missing_in_other


def get_paired_adj(graph_type, nodelist):
    graph = load_networkx(graph_type)
    matched_graph = graph.subgraph(nodelist)
    adj_df = nx.to_pandas_adjacency(matched_graph, nodelist=nodelist)
    adj = adj_df.values
    return adj


def get_paired_edges(adj):
    n_per_side = adj.shape[0] // 2
    left_left_adj = adj[:n_per_side, :n_per_side]
    left_right_adj = adj[:n_per_side, n_per_side:]
    right_right_adj = adj[n_per_side:, n_per_side:]
    right_left_adj = adj[n_per_side:, :n_per_side]

    left_left_edges = left_left_adj.ravel()
    left_right_edges = left_right_adj.ravel()
    right_right_edges = right_right_adj.ravel()
    right_left_edges = right_left_adj.ravel()

    left_edges = np.concatenate((left_left_edges, left_right_edges))
    right_edges = np.concatenate((right_right_edges, right_left_edges))
    edge_inds = np.where(np.logical_or(left_edges != 0, right_edges != 0))[0]
    left_edges = left_edges[edge_inds]
    right_edges = right_edges[edge_inds]

    return (left_edges, right_edges)


#%% Load some graph info
graph = load_networkx("Gn")

filename = (
    "maggot_models/data/raw/Maggot-Brain-Connectome/4-color-matrices_Brain/"
    + "2019-09-18-v2/brain_meta-data.csv"
)

meta_df = pd.read_csv(filename, index_col=0)
print(meta_df.head())

pair_df = pd.read_csv(
    "maggot_models/data/raw/Maggot-Brain-Connectome/pairs/knownpairsatround5.csv"
)

print(pair_df.head())


#%% Split up left and right nodes, sort adj and labels


left_nodes = pair_df["leftid"].values.astype(str)
right_nodes = pair_df["rightid"].values.astype(str)

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
        if left in graph and right in graph:
            left_nodes.append(left)
            right_nodes.append(right)

side_labels = np.array(len(left_nodes) * ["Left"] + len(right_nodes) * ["Right"])
nodelist = np.concatenate((left_nodes, right_nodes)).astype(str)

n_per_side = len(left_nodes)

class_labels = meta_df.loc[nodelist.astype(int), "Class"].values

name_map = {
    "CN": "CN",
    "DANs": "MBIN",
    "KCs": "MBIN",
    "LHN": "LHN",
    "LHN; CN": "LHN",
    "MBINs": "MBIN",
    "MBON": "MBON",
    "MBON; CN": "MBON",
    "OANs": "MBIN",
    "ORN mPNs": "PN",
    "ORN uPNs": "PN",
    "tPNs": "PN",
    "vPNs": "PN",
    "Unidentified": "Unk",
    "Other": "Unk",
}
class_labels = np.array(itemgetter(*class_labels)(name_map))

#%% Try to fit a DCSBM, essentially

from graspy.cluster import KMeansCluster
from graspy.models import DCSBMEstimator
from graspy.embed import LaplacianSpectralEmbed

adj = get_paired_adj("G", nodelist)
embed_adj = pass_to_ranks(adj)

skmeans_kws = dict(n_init=100, n_jobs=-2)
n_clusters = 12
n_components = None
regularizer = 1

lse = LaplacianSpectralEmbed(n_components=n_components, regularizer=regularizer)
latent = lse.fit_transform(embed_adj)
latent = np.concatenate(latent, axis=-1)

models = []


def relabel(labels):
    """
    Remaps integer labels based on who is most frequent
    """
    uni_labels, uni_inv, uni_counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    sort_inds = np.argsort(uni_counts)[::-1]
    new_labels = range(len(uni_labels))
    uni_labels_sorted = uni_labels[sort_inds]
    relabel_map = dict(zip(uni_labels_sorted, new_labels))

    new_labels = np.array(itemgetter(*labels)(relabel_map))
    return new_labels


def countplot(graphs, graph_types, figsize=(12, 1)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis("off")

    counts = []
    for g in graphs:
        num_edges = g.sum().astype(int)
        counts.append(num_edges)

    counts = np.array(counts)
    count_cumsum = counts.cumsum()

    starts = count_cumsum - counts
    widths = counts
    centers = starts + widths / 2
    ax.set_xlim(0, np.sum(counts))

    for i, e in enumerate(graph_types):
        y = 0.5
        plt.barh(y, width=widths[i], left=starts[i], height=0.5, label=e)
        prop = 100 * widths[i] / widths.sum()
        prop = f"{prop:2.0f}%"
        ax.text(
            centers[i], 0.15, prop, ha="center", va="center", color="k", fontsize=15
        )


def survey(labels, category_names, ax=None):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    sns.set_context("talk")

    uni_labels = np.unique(labels)

    counts_by_label = []
    for label in uni_labels:
        inds = np.where(pred_labels == label)
        classes_in_cluster = class_labels[inds]
        counts_by_class = []
        for c in uni_class:
            num_class_in_cluster = len(np.where(classes_in_cluster == c)[0])
            counts_by_class.append(num_class_in_cluster)
        counts_by_label.append(counts_by_class)
    results = dict(zip(uni_labels, counts_by_label))

    labels = list(results.keys())
    data = np.array(list(results.values()))
    # print(category_names)
    # print(data)
    data = data[:, 1:]
    data_cum = data.cumsum(axis=1)
    category_names = category_names[1:]
    category_colors = sns.color_palette("tab20", n_colors=len(category_names))
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    max_size = np.sum(data, axis=1).max()
    ax.set_xlim(0 - 0.02 * max_size, max_size * 1.02)
    # ax.set_ylim()
    # ax.set_ylim(max(labels) + 1, -1)
    ax.set_yticklabels(labels)
    # ax.set_yticks(labels)
    height = 0.3

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=height, label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b = color
        # text_color = "white" if r * g * b < 0.5 else "darkgrey"
        text_color = "black"
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c != 0:
                ax.text(
                    x,
                    y - height / 2 - i % 2 * height / 4,
                    str(int(c)),
                    ha="center",
                    va="bottom",
                    color=text_color,
                )
    ax.legend(
        ncol=6,  # len(category_names),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )

    return ax


def get_label_counts(labels):
    labels, counts = np.unique(labels, return_counts=True)
    return counts


# Try with my new labels

meta_file = "maggot_models/data/processed/2019-09-18-v2/BP_metadata.csv"

meta_df = pd.read_csv(meta_file, index_col=0)
print(meta_df.head())
class_labels = meta_df.loc[nodelist.astype(int), "BP_Class"].values
class_labels[class_labels == "LH2N"] = "Other"

from graspy.utils import binarize

uni_class, class_counts = np.unique(class_labels, return_counts=True)
inds = np.argsort(class_counts)[::-1]
uni_class = uni_class[inds]
class_counts = class_counts[inds]

n_clusters = 4
for k in range(2, n_clusters):
    skmeans = SphericalKMeans(n_clusters=k, **skmeans_kws)
    pred_labels = skmeans.fit_predict(latent)
    pred_labels = relabel(pred_labels)
    models.append(skmeans)

    # gridplot(
    #     [adj], inner_hier_labels=pred_labels, hier_label_fontsize=18, sizes=(2, 10)
    # )
    fig, ax = plt.subplots(1, 2, figsize=(30, 18))
    heatmap(
        binarize(adj),
        inner_hier_labels=pred_labels,
        # outer_hier_labels=side_labels,
        hier_label_fontsize=18,
        ax=ax[0],
        cbar=False,
        sort_nodes=True,
    )
    uni_labels = np.unique(pred_labels)
    # survey(pred_labels, uni_class, ax=ax[1])
    survey(class_labels, uni_labels, ax=ax[1])

#%%
# heatmap(adj, inner_hier_labels=pred_labels)

