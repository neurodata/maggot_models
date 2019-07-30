#%%
from operator import itemgetter
from pathlib import Path
from warnings import filterwarnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

import src.utils as utils
from graspy.cluster import GaussianCluster
from graspy.embed import (
    AdjacencySpectralEmbed,
    LaplacianSpectralEmbed,
    MultipleASE,
    OmnibusEmbed,
)
from graspy.models import DCEREstimator, DCSBMEstimator, RDPGEstimator, SBMEstimator

from graspy.plot import gridplot, heatmap, pairplot, screeplot
from graspy.utils import (
    binarize,
    get_lcc,
    import_graph,
    is_fully_connected,
    pass_to_ranks,
    remove_loops,
    to_laplace,
)
from src.data import load_june, load_left, load_right
from src.models import (
    GridSweep,
    fit_a_priori,
    gen_scorers,
    select_dcsbm,
    select_rdpg,
    select_sbm,
)
from src.utils import get_best, get_subgraph, to_simple_class


def meta_to_array(graph, key):
    data = [meta[key] for node, meta in graph.nodes(data=True)]
    return np.array(data)


def get_simple(graph):
    classes = meta_to_array(graph, "Class")
    simple_classes = to_simple_class(classes)
    return simple_classes


def save(name, fmt="pdf"):
    path = Path("./maggot_models/notebooks/outs")
    plt.savefig(path / str(name + "." + fmt), fmt=fmt, facecolor="w")


#%%

# def preprocess(graph):
#     graph[graph < ]
#     return graph

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

heatmap(
    full_graph,
    transform="simple-all",
    figsize=(20, 20),
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    hier_label_fontsize=10,
)
gridplot(
    [full_graph],
    transform="simple-all",
    height=15,
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    hier_label_fontsize=10,
    sizes=(1, 10),
)
#%%
print("Performing thresholding")
full_graph_t = full_graph.copy()
full_graph_t = import_graph(full_graph_t)
full_graph_t[full_graph_t < 0.01] = 0
n_edges_t = np.count_nonzero(full_graph_t)

print(f"Number of edges remaining: {n_edges_t}")
print(f"Removed {(n_edges - n_edges_t) / n_edges} of edges")

heatmap(
    full_graph_t,
    transform="simple-all",
    figsize=(20, 20),
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    hier_label_fontsize=10,
)
gridplot(
    [full_graph_t],
    transform="simple-all",
    height=15,
    inner_hier_labels=simple_classes,
    outer_hier_labels=hemisphere,
    hier_label_fontsize=10,
    sizes=(1, 10),
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
print("Embedding graph")

n_components = 4
regularizer = 2
ptr = True
binary = False
embed_graph = lcc_graph_t
if ptr:
    embed_graph = pass_to_ranks(embed_graph)
if binary:
    embed_graph = binarize(embed_graph)

# lse = LaplacianSpectralEmbed(
#     n_components=n_components, form="R-DAD", regularizer=regularizer
# )
embed_graph = to_laplace(embed_graph, form="R-DAD", regularizer=regularizer)

heatmap(
    embed_graph,
    figsize=(20, 20),
    inner_hier_labels=lcc_simple_classes,
    outer_hier_labels=lcc_hemisphere,
    hier_label_fontsize=10,
    sort_nodes=True,
)
gridplot(
    [embed_graph],
    height=15,
    inner_hier_labels=lcc_simple_classes,
    outer_hier_labels=lcc_hemisphere,
    hier_label_fontsize=10,
    sizes=(1, 10),
    sort_nodes=True,
)

#%%
dcer = DCEREstimator()
dcer.fit(embed_graph)
gridplot(
    [binarize(embed_graph)],
    height=15,
    inner_hier_labels=lcc_simple_classes,
    outer_hier_labels=lcc_hemisphere,
    hier_label_fontsize=10,
    sizes=(5, 5),
    sort_nodes=True,
)
gridplot(
    [dcer.sample()[0]],
    height=15,
    inner_hier_labels=lcc_simple_classes,
    outer_hier_labels=lcc_hemisphere,
    hier_label_fontsize=10,
    sizes=(5, 5),
    sort_nodes=True,
)
#%%
screeplot(embed_graph, cumulative=False, show_first=30)


#%%
# pairplot(latent, labels=lcc_simple_classes)

#%%
sns.set_palette("deep")
sns.set_context("talk")
min_components = 2
max_components = 35
comps = list(range(min_components, max_components))
cluster_kws = dict(
    min_components=min_components, max_components=max_components, covariance_type="full"
)
n_sims = 20
n_components_range = list(range(3, 5))

bic_results = []
for i, n_components in tqdm(enumerate(n_components_range)):
    ase = AdjacencySpectralEmbed(n_components=n_components)
    latent = ase.fit_transform(embed_graph)
    latent = np.concatenate(latent, axis=-1)

    for _ in range(n_sims):
        cluster = GaussianCluster(**cluster_kws)
        cluster.fit(latent)
        bics = cluster.bic_
        bics["n_clusters"] = bics.index
        bics["n_components"] = n_components
        bic_results.append(bics)

#%%
result_df = pd.concat(bic_results, axis=0)
result_df.rename(columns={"full": "bic"}, inplace=True)


plt.figure(figsize=(15, 10))
sns.lineplot(data=result_df, x="n_clusters", y="bic", hue="n_components")

save("clustering_june_bics")
# #%%
# test_mat = np.zeros((100, 100))
# test_mat[:25, :][:, :25] = -1
# test_mat[25:50, :][:, 25:50] = -2
# test_mat[50:75, :][:, 50:75] = 1
# test_mat[75:100, :][:, 75:100] = 2
# heatmap(test_mat)
# labels = 25 * [0] + 25 * [1] + 25 * [2] + 25 * [3]
# labels = np.array(labels)
# outer_labels = 50 * ["low"] + 50 * ["high"]
# outer_labels = np.array(outer_labels)
# rand_perm = np.random.permutation(100)
# test_mat = test_mat[np.ix_(rand_perm, rand_perm)]
# labels = labels[rand_perm]
# outer_labels = outer_labels[rand_perm]
# heatmap(test_mat, inner_hier_labels=labels)

# heatmap(test_mat, inner_hier_labels=labels, outer_hier_labels=outer_labels)

# #%%
