#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances

from graspy.embed import AdjacencySpectralEmbed
from graspy.models import DCSBMEstimator, HSBMEstimator
from graspy.plot import gridplot, heatmap, hierplot, pairplot
from graspy.utils import binarize, get_lcc, import_graph, pass_to_ranks, to_laplace
from src.data import load_june
from src.utils import (
    get_best,
    get_simple,
    get_subgraph,
    meta_to_array,
    savefig,
    to_simple_class,
)


def normalized_ase(graph, n_components=None):
    ase = AdjacencySpectralEmbed(n_components=n_components)
    latent = ase.fit_transform(graph)
    if isinstance(latent, tuple):
        latent = np.concatenate(latent, axis=-1)
    norm_vec = np.linalg.norm(latent, axis=1)
    norm_vec[norm_vec == 0] = 1
    norm_latent = latent / norm_vec[:, np.newaxis]
    return norm_latent


def hierarchical_split(graph, cluster_obj, n_components=3, n_stop=50):

    if graph.shape[0] <= n_stop:
        return [np.arange(graph.shape[0])]

    # define a notion of similarity and subdivide graph
    norm_latent = normalized_ase(graph, n_components=n_components)

    cluster_labels = cluster_obj.fit_predict(norm_latent)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    # for each graph subset, recurse (potentially)
    original_graph_inds = []
    for label in unique_labels:
        subgraph_inds = np.where(cluster_labels == label)[0]
        print(subgraph_inds)
        subgraph = graph[np.ix_(subgraph_inds, subgraph_inds)]
        subgraph_split = hierarchical_split(
            subgraph, cluster_obj, n_components=n_components, n_stop=n_stop
        )
        # subgraph split is a list of arrays

        for i, r_inds in enumerate(subgraph_split):
            print(i)
            original_inds = subgraph_inds[r_inds]

            print()
            print(original_graph_inds)
            original_graph_inds.append(original_inds)
            print("after append")
            print(original_graph_inds)
            print()

    return original_graph_inds


# split func should return a vector of labels
# stop func should return true if you should stop
# otherwise, will stop when label vector returns everything in one class


# def bisplit_tree(graph, split_func, stop_func=None):
#     if stop_func is not None:
#         if stop_func(graph):
#             return [np.arange(graph.shape[0])]

#     split_labels = split_func(graph)
#     unique_labels = np.unique(split_labels)

#     original_graph_inds = []
#     for label in unique_labels:
#         subgraph_inds = np.where(split_labels == label)[0]
#         subgraph = graph[np.ix_(subgraph_inds, subgraph_inds)]

#         recurse_split = bisplit_tree(subgraph, split_func, stop_func)

#         for i, r_inds in enumerate(recurse_split):
#             original_inds = subgraph_inds[r_inds]
#             original_graph_inds.append(original_inds)


plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)
gridplot_kws = dict(
    hier_label_fontsize=10,
    title_pad=120,
    sizes=(1, 10),
    height=15,
    transform="simple-all",
)
#%%
print("Loading graph")

graph_type = "Gadn"
weight_threshold = 0.01

full_graph = load_june(graph_type)
simple_classes = get_simple(full_graph)
hemisphere = meta_to_array(full_graph, "Hemisphere")

n_nodes = len(full_graph)
n_edges = len(full_graph.edges())

print(f"Number of nodes: {n_nodes}")
print(f"Number of edges: {n_edges}")

# heatmap(
#     full_graph,
#     transform="simple-all",
#     figsize=(20, 20),
#     inner_hier_labels=simple_classes,
#     outer_hier_labels=hemisphere,
#     hier_label_fontsize=10,
#     title="Full graph, Gadn, PTR-simple-all",
# )
gridplot(
    [full_graph],
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    title="Full graph, Gadn, PTR-simple-all",
    **gridplot_kws,
)
#%%
print("Performing thresholding")
full_graph_t = full_graph.copy()
full_graph_t = import_graph(full_graph_t)
full_graph_t[full_graph_t < 0.01] = 0
n_edges_t = np.count_nonzero(full_graph_t)

print(f"Number of edges remaining: {n_edges_t}")
print(f"Removed {(n_edges - n_edges_t) / n_edges} of edges")

# heatmap(
#     full_graph_t,
#     transform="simple-all",
#     figsize=(20, 20),
#     inner_hier_labels=simple_classes,
#     outer_hier_labels=hemisphere,
#     hier_label_fontsize=10,
# )
gridplot(
    [full_graph_t],
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    title="Weight thresholded, Gadn, PTR-simple-all",
    **gridplot_kws,
)

#%%
print("Finding largest connected component")
lcc_graph_t, lcc_inds = get_lcc(full_graph_t, return_inds=True)
lcc_simple_classes = simple_classes[lcc_inds]
lcc_hemisphere = hemisphere[lcc_inds]
n_nodes_t = lcc_graph_t.shape[0]
print(f"Number of remaining nodes: {n_nodes_t}")
print(f"Removed {(n_nodes - n_nodes_t) / n_nodes} of nodes")


#%%

print("Embedding binarized graph")

n_components = 4
regularizer = 2
ptr = False
binary = True
embed_graph = lcc_graph_t
if ptr:
    embed_graph = pass_to_ranks(embed_graph)
if binary:
    embed_graph = binarize(embed_graph)

gridplot_kws["sizes"] = (10, 10)
gridplot(
    [embed_graph],
    inner_hier_labels=lcc_simple_classes,
    outer_hier_labels=lcc_hemisphere,
    **gridplot_kws,
)

ase = AdjacencySpectralEmbed(n_components=n_components)
latent = ase.fit_transform(embed_graph)
latent = np.concatenate(latent, axis=-1)
pairplot(latent, title="ASE o binarized o thresholded")


norm_latent = latent / np.linalg.norm(latent, axis=1)[:, np.newaxis]
pairplot(norm_latent, title="normalized ASE o binarized o thresholded")

# this is the same as (norm_latent @ norm_latent.T) - min()
dists = pairwise_distances(norm_latent, metric="cosine")


#%%


mpl.rcParams["lines.linewidth"] = 1
ax = hierplot(dists, figsize=(20, 20))
ax.set_title("cosine distances o normalized ASE", pad=400)
savefig("dros_heatmap_dissim.png", fmt="png")

#%%
sns.set_context("talk")
print("Cluster into 7 groups in this case")
hierclust = AgglomerativeClustering(
    affinity="precomputed", compute_full_tree=True, n_clusters=7, linkage="average"
)
hier1_labels = hierclust.fit_predict(dists)

gridplot(
    [embed_graph],
    inner_hier_labels=hier1_labels,
    title="whole graph sorted by hierclust, n_clust=7",
    **gridplot_kws,
)


#%%
uni_labels = np.unique(hier1_labels)
for i in uni_labels:
    subvert_inds = np.where(hier1_labels == i)[0]
    subgraph = embed_graph[np.ix_(subvert_inds, subvert_inds)]
    print(i)
    print(subgraph.shape)
    subgraph = get_lcc(subgraph)
    print(subgraph.shape)
    subgraph_labels = lcc_simple_classes[subvert_inds]
    uni_sublabels, subcounts = np.unique(subgraph_labels, return_counts=True)
    print(uni_sublabels)
    print(subcounts / np.sum(subcounts))

    print()

    # ase = AdjacencySpectralEmbed()
    # ase.fit_transform(subgraph)

#%% focus on subgraph id 2
subvert_inds = np.where(hier1_labels == 2)[0]
subgraph = embed_graph[np.ix_(subvert_inds, subvert_inds)]
subgraph = get_lcc(subgraph)

# subgraph_labels = lcc_simple_classes[subvert_inds]
# gridplot(
#     [subgraph],
#     height=15,
#     inner_hier_labels=subgraph_labels,
#     hier_label_fontsize=10,
#     sizes=(1, 10),
# )

subgraph_latent = normalized_ase(subgraph, n_components=4)
dists = pairwise_distances(subgraph_latent, metric="cosine")
hierplot(dists, figsize=(20, 20))
savefig("hier2_dissim_heat", fmt="png")
#%%
hierclust = AgglomerativeClustering(
    affinity="precomputed", compute_full_tree=True, n_clusters=7, linkage="average"
)
hier2_labels = hierclust.fit_predict(dists)
gridplot(
    [subgraph],
    height=15,
    inner_hier_labels=hier2_labels,
    hier_label_fontsize=10,
    sizes=(2, 20),
)

#%%

# stop if number of verts in graph is <= n_stop
# embed
# normalize
# cluster
# chose the number of clusters somehow
# take the induced subgraph
# recurse on each subgraph, returns partition of the nodes
# return partition of the nodes


class DotAgglomerativeClustering(AgglomerativeClustering):
    def fit_predict(self, X, y=None):
        dists = X @ X.T
        dists = 1 - dists
        return super().fit_predict(dists)


cluster_obj = DotAgglomerativeClustering(
    n_clusters=6, affinity="precomputed", linkage="average"
)
graph_partitions = hierarchical_split(
    lcc_graph_t, cluster_obj, n_components=6, n_stop=50
)
#%%
label_vec = np.zeros(lcc_graph_t.shape[0], dtype="U10")
count_vec = np.zeros(len(graph_partitions))
for i, inds in enumerate(graph_partitions):
    count_vec[i] = len(inds)
    if len(inds) > 1:
        label_vec[inds] = f"{i}"
        print(i)
    else:
        label_vec[inds] = "U"

gridplot(
    [binarize(lcc_graph_t)],
    inner_hier_labels=label_vec,
    height=25,
    hier_label_fontsize=5,
    sizes=(8, 20),
)
savefig("grouped_gridplot", fmt="png")
#%%
sbm_model = DCSBMEstimator().fit(binarize(lcc_graph_t), y=label_vec)
sample = sbm_model.sample()[0]
gridplot(
    [binarize(lcc_graph_t)],
    height=10,
    hier_label_fontsize=5,
    sizes=(4, 4),
    inner_hier_labels=lcc_simple_classes,
)
gridplot(
    [sample],
    height=10,
    hier_label_fontsize=5,
    sizes=(4, 4),
    inner_hier_labels=lcc_simple_classes,
)

#%%
plt.style.use("seaborn-white")
l = normalized_ase(lcc_graph_t, n_components=6)
dists = pairwise_distances(l, metric="cosine")
c = AgglomerativeClustering(n_clusters=7, affinity="precomputed", linkage="average")
labels = c.fit_predict(dists)
np.unique(labels, return_counts=True)
hierplot(dists)
#%%
print("Fitting HSBM")

fit_graph = lcc_graph_t.copy()
n_subgroups = 3
n_subgraphs = 6
n_components_lvl1 = 8
n_components_lvl2 = 8
hsbm = HSBMEstimator(
    n_levels=2,
    n_subgraphs=n_subgraphs,
    n_subgroups=n_subgroups,
    n_components_lvl1=n_components_lvl1,
    n_components_lvl2=n_components_lvl2,
)
hsbm.fit(fit_graph)


#%%


class GraphPartitionTree:
    def __init__(self,):
        self.children = None
        self.graph = None
        self.split_inds = None
        self.global_inds = None


def split_graph(gpt, split_func, stop_func=None):
    graph = gpt.graph

    if stop_func is not None:
        if stop_func(graph):
            return gpt

    split_labels = split_func(graph)
    unique_labels, counts = np.unique(split_labels, return_counts=True)

    # # for each graph subset, recurse (potentially)
    # original_graph_inds = []
    children = []
    sub_inds = []
    for label in unique_labels:
        subgraph_inds = np.where(split_labels == label)[0]
        subgraph = graph[np.ix_(subgraph_inds, subgraph_inds)]

        sub_gpt = GraphPartitionTree()
        sub_gpt.global_inds = gpt.global_inds[subgraph_inds]
        sub_gpt.graph = subgraph

        out_gpt = split_graph(sub_gpt, split_func, stop_func)

        children.append(out_gpt)
        sub_inds.append(subgraph_inds)

    gpt.split_inds = sub_inds
    gpt.children = children
    return gpt
    #     subgraph_split = split_tree(subgraph, split_func, stop_func)
    #     # subgraph split is a list of arrays

    #     for i, r_inds in enumerate(subgraph_split):
    #         original_inds = subgraph_inds[r_inds]
    #         original_graph_inds.append(original_inds)

    # return original_graph_inds
    # left_subgraph = graph[np.ix_()]


def split_tree(graph, split_func, stop_func=None):

    if stop_func is not None:
        if stop_func(graph):
            return [np.arange(graph.shape[0])]

    split_labels = split_func(graph)
    unique_labels, counts = np.unique(split_labels, return_counts=True)

    # for each graph subset, recurse (potentially)
    original_graph_inds = []
    for label in unique_labels:
        subgraph_inds = np.where(split_labels == label)[0]
        subgraph = graph[np.ix_(subgraph_inds, subgraph_inds)]
        subgraph_split = split_tree(subgraph, split_func, stop_func)
        # subgraph split is a list of arrays

        for i, r_inds in enumerate(subgraph_split):
            original_inds = subgraph_inds[r_inds]
            original_graph_inds.append(original_inds)

    return original_graph_inds


def kmeans_split(graph):
    # ase = AdjacencySpectralEmbed(n_components=10)
    # latent = ase.fit_transform(graph)
    # if isinstance(latent, tuple):
    #     latent = np.concatenate(latent, axis=-1)
    latent = normalized_ase(graph, n_components=10)
    kmeans = KMeans(n_clusters=2)
    cluster_prediction = kmeans.fit_predict(latent)
    return cluster_prediction


def simple_stop(graph):
    if graph.shape[0] < 100:
        return True


# split_out = split_tree(embed_graph, kmeans_split, simple_stop)
root_gpt = GraphPartitionTree()
root_gpt.graph = embed_graph
root_gpt.global_inds = np.arange(embed_graph.shape[0])
split_out = split_graph(root_gpt, kmeans_split, simple_stop)
#%%

gpt = split_out

leaf_holder = []


def get_leaves(gpt, leaf_holder):
    if gpt.children is None:
        print("leaf")
        leaf_holder.append(gpt.global_inds)
        return leaf_holder
    else:
        print("node")
        get_leaves(gpt.children[0], leaf_holder)
        get_leaves(gpt.children[1], leaf_holder)
        return leaf_holder


leaf_holder = get_leaves(split_out, leaf_holder)
leaf_indicator = np.zeros(embed_graph.shape[0])
#%%
for i, leaf in enumerate(leaf_holder):
    leaf_indicator[leaf] = i

#%%
gridplot([embed_graph], inner_hier_labels=leaf_indicator, **gridplot_kws)

