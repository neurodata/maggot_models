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

graph_type = "Gadn"
weight_threshold = 0.01
font_scale = 2
gridplot_kws = dict(
    hier_label_fontsize=10,
    title_pad=120,
    sizes=(1, 10),
    height=15,
    transform="simple-all",
    sort_nodes=True,
)

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=font_scale)


def normalized_ase(graph, n_components=None, embed_kws={}):
    ase = AdjacencySpectralEmbed(n_components=n_components, **embed_kws)
    latent = ase.fit_transform(graph)
    if isinstance(latent, tuple):
        latent = np.concatenate(latent, axis=-1)
    norm_vec = np.linalg.norm(latent, axis=1)
    norm_vec[norm_vec == 0] = 1
    norm_latent = latent / norm_vec[:, np.newaxis]
    return norm_latent


#%%
print("Loading graph")


full_graph = load_june(graph_type)
simple_classes = get_simple(full_graph)
hemisphere = meta_to_array(full_graph, "Hemisphere")

n_nodes = len(full_graph)
n_edges = len(full_graph.edges())

print(f"Number of nodes: {n_nodes}")
print(f"Number of edges: {n_edges}")

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
from graspy.plot import screeplot

screeplot(embed_graph, cumulative=False, show_first=20, n_elbows=3)
#%%
n_components = None
n_elbows = 1
embed_graph = lcc_graph_t
embed_graph = binarize(lcc_graph_t)

gridplot_kws["sizes"] = (10, 10)
gridplot(
    [embed_graph],
    inner_hier_labels=lcc_simple_classes,
    outer_hier_labels=lcc_hemisphere,
    **gridplot_kws,
)

ase = AdjacencySpectralEmbed(n_components=n_components, n_elbows=n_elbows)
latent = ase.fit_transform(embed_graph)
latent = np.concatenate(latent, axis=-1)
pairplot(latent, title="ASE o binarized o thresholded")


norm_latent = latent / np.linalg.norm(latent, axis=1)[:, np.newaxis]
pairplot(norm_latent, title="normalized ASE o binarized o thresholded")

# this is the same as (norm_latent @ norm_latent.T) - min()
dists = pairwise_distances(norm_latent, metric="cosine")


#%%
print("Fitting HSBM")

fit_graph = embed_graph
n_subgroups = 3  # this only affects the dissimilarity clustering
n_subgraphs = 15  # maximum number of subgraphs
n_components_lvl1 = None  # this will use ZG1
n_components_lvl2 = None  # this will use ZG2
cluster_kws = dict(n_init=100)
hsbm = HSBMEstimator(
    n_levels=2,
    n_subgraphs=n_subgraphs,
    n_subgroups=n_subgroups,
    n_components_lvl1=n_components_lvl1,
    n_components_lvl2=n_components_lvl2,
    cluster_kws=cluster_kws,
    cluster_method="sphere-kmeans",
    n_elbows=1,
)
hsbm.fit(fit_graph)

#%%
print("Plotting spherical kmeans clustering result on normalized ASE, ZG1")
latent = hsbm.latent_
labels = hsbm.vertex_assignments_
pairplot(
    latent,
    labels=labels,
    title="sphere-kmeans o normalized ASE, ZG1 o binarized o thresholded",
)
#%%
print("Plotting graph sorted by kmeans clustering result")
gridplot(
    [embed_graph],
    inner_hier_labels=labels,
    outer_hier_labels=lcc_hemisphere,
    title="LCC graph sorted by spherical kmeans result",
    **gridplot_kws,
)

#%%
print("Plotting subgraph dissimilarities")
dists = hsbm.subgraph_dissimilarities_
dists -= dists.min()
hierplot(dists)
plt.title("Subgraph dissimilarity matrix (nonpar test stat)", pad=200, loc="left")


#########
#%%

mpl.rcParams["lines.linewidth"] = 1
ax = hierplot(dists, figsize=(20, 20))
ax.set_title("cosine distances o normalized ASE", pad=400)

#%%
sns.set_context("talk", font_scale=2)
print("Cluster into 6 groups in this case")
hierclust = AgglomerativeClustering(
    affinity="precomputed", compute_full_tree=True, n_clusters=6, linkage="average"
)
hier1_labels = hierclust.fit_predict(dists)

gridplot(
    [embed_graph],
    inner_hier_labels=hier1_labels,
    title="whole graph sorted by hierclust, n_clust=6",
    **gridplot_kws,
)

###########
#%% focus on subgraph id 2
subvert_inds = np.where(hier1_labels == 2)[0]
subgraph = embed_graph[np.ix_(subvert_inds, subvert_inds)]
subgraph = get_lcc(subgraph)

subgraph_latent = normalized_ase(subgraph, n_components=4)
dists = pairwise_distances(subgraph_latent, metric="cosine")
hierplot(dists, figsize=(20, 20))
savefig("hier2_dissim_heat", fmt="png")
#%% cluster a subgraph
hierclust = AgglomerativeClustering(
    affinity="precomputed", compute_full_tree=True, n_clusters=7, linkage="average"
)
hier2_labels = hierclust.fit_predict(dists)
gridplot([subgraph], inner_hier_labels=hier2_labels, **gridplot_kws)


#%% embed a subgraph
plt.style.use("seaborn-white")
l = normalized_ase(lcc_graph_t, n_components=6)
dists = pairwise_distances(l, metric="cosine")
c = AgglomerativeClustering(n_clusters=7, affinity="precomputed", linkage="average")
labels = c.fit_predict(dists)
np.unique(labels, return_counts=True)
hierplot(dists)


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

