#%%
from operator import itemgetter
from pathlib import Path
from warnings import filterwarnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed
from graspy.plot import gridplot, heatmap, pairplot

# simple_classes = simple_classes[inds]
from graspy.utils import (
    binarize,
    get_lcc,
    import_graph,
    is_fully_connected,
    pass_to_ranks,
    remove_loops,
    to_laplace,
)

plt.style.use("seaborn-white")
# data_path = Path("maggot_models/data/raw/20190615_mw")
# base_file_name = "mw_20190615_"
graph_types = ["Gaa", "Gad", "Gda", "Gdd"]


def load_graph(graph_type):
    file_path = data_path / (base_file_name + graph_type + ".graphml")
    graph = nx.read_graphml(file_path)
    return graph


def meta_to_array(graph, key):
    data = [meta[key] for node, meta in graph.nodes(data=True)]
    return np.array(data)


# data paths, and get the types of graph we want to look at
# %config InlineBackend.figure_format = 'png'

graph = import_graph(load_graph("G"))
degree_sum = np.count_nonzero(graph, axis=0) + np.count_nonzero(graph, axis=1)
sort_inds = np.argsort(degree_sum)[::-1]

#%%
fig, ax = plt.subplots(2, 2, figsize=(30, 30))
ax = ax.ravel()


graph_dict = {}
graph_list = []
plot_graphs = []
for i, graph_type in enumerate(graph_types):
    file_path = data_path / (base_file_name + graph_type + ".graphml")
    graph = nx.read_graphml(file_path)

    graph_dict[graph_type] = graph
    graph_list.append(graph)

    node_data = list(graph.nodes.data())
    names, data = zip(*node_data)
    meta_df = pd.DataFrame(data)
    classes = meta_df["Class"]
    simple_classes = to_simple_class(classes)
    graph = binarize(graph)
    graph = graph[np.ix_(sort_inds, sort_inds)]
    simple_classes = simple_classes[sort_inds]
    plot_graphs.append(graph)
    heatmap(
        graph,
        inner_hier_labels=simple_classes,
        transform="simple-nonzero",
        hier_label_fontsize=15,
        ax=ax[i],
        title=graph_type,
        cbar=False,
        title_pad=70,
        font_scale=2,
    )
# plt.tight_layout()
plt.savefig("4-color.png", facecolor="w", format="png")

labels = ["A -> A", "A -> D", "D -> D", "D -> A"]
gridplot(
    plot_graphs,
    labels=labels,
    inner_hier_labels=simple_classes,
    height=40,
    sizes=(5, 5),
    transform="binarize",
    title="Right Drosophila larva, 4-color graph",
    title_pad=190,
    font_scale=3.5,
)
plt.savefig("gridplot-4color.pdf", format="pdf", facecolor="w", bbox_inches="tight")

#%%
g = graph_list[0]
right_nodes = [x for x, y in g.nodes(data=True) if y["Hemisphere"] == "right"]
left_subgraph = g.subgraph(right_nodes)


#%%
# %config InlineBackend.figure_format = 'png'

right_graph_list = [get_subgraph(g, "Hemisphere", "right") for g in graph_list]

graphs = right_graph_list
n_graphs = 4
n_verts = len(graphs[0].nodes)
n_components = 2

embed_graphs = [pass_to_ranks(g) for g in graphs]

omni = OmnibusEmbed(n_components=n_components)
latent = omni.fit_transform(embed_graphs)
latent = np.concatenate(latent, axis=-1)
plot_latent = latent.reshape((n_graphs * n_verts, 2 * n_components))
labels = (
    n_verts * ["A -> A"]
    + n_verts * ["A -> D"]
    + n_verts * ["D -> D"]
    + n_verts * ["D -> A"]
)
# latent = np.concatenate(list(latent))
pairplot(plot_latent, labels=labels)

#%% concatenate and look at that
concatenate_latent = np.concatenate(list(latent), axis=-1)
concatenate_latent.shape
pairplot(concatenate_latent, labels=unknown)
#%%
g = graphs[0]
classes = [meta["Class"] for node, meta in g.nodes(data=True)]
classes = np.array(classes)
unknown = classes == "Other"
plot_unknown = np.tile(unknown, n_graphs)
pairplot(plot_latent, labels=plot_unknown, alpha=0.3, legend_name="Unknown")


clust_latent = np.concatenate(list(latent), axis=-1)
clust_latent.shape
#%%
gc = GaussianCluster(min_components=2, max_components=15, covariance_type="all")

filterwarnings("ignore")
n_init = 50
sim_mat = np.zeros((n_verts, n_verts))

for i in tqdm(range(n_init)):
    assignments = gc.fit_predict(clust_latent)
    for c in np.unique(assignments):
        inds = np.where(assignments == c)[0]
        sim_mat[np.ix_(inds, inds)] += 1


sim_mat -= np.diag(np.diag(sim_mat))
sim_mat = sim_mat / n_init
heatmap(sim_mat)


#%%
thresh_sim_mat = sim_mat.copy()
thresh_sim_mat[thresh_sim_mat > 0.5] = 1
thresh_sim_mat[thresh_sim_mat < 0.5] = 0

heatmap(thresh_sim_mat)
ase = AdjacencySpectralEmbed(n_components=5)
sim_latent = ase.fit_transform(sim_mat)
pairplot(sim_latent, labels=~unknown)
c = GaussianCluster(min_components=2, max_components=10)
overall = c.fit_predict(sim_latent)
#%%
heatmap(sim_mat, inner_hier_labels=overall, sort_nodes=True)

#%%
graph_types = ["Gaan", "Gadn", "Gdan", "Gddn"]
n_components = 4

# load the right side
use_graph = "Gn"
hemisphere = "right"
print(f"Using graph {use_graph}")
Gn = load_graph(use_graph)
Gn = get_subgraph(Gn, "Hemisphere", hemisphere)
n_verts_original = len(Gn)
print(f"Selected {hemisphere} side")

print("Checking if graph is fully connected")
print(is_fully_connected(Gn))
Gn, inds = get_lcc(Gn, return_inds=True)
num_removed = n_verts_original - len(Gn)
print(f"Removed {num_removed} node")

# select metadata
classes = meta_to_array(Gn, "Class")
simple_classes = to_simple_class(classes)
names = meta_to_array(Gn, "Name")
ids = meta_to_array(Gn, "ID")

# load adjacency and preprocess
Gn_adj = import_graph(Gn)
Gn_adj = remove_loops(Gn_adj)
# Gn = pass_to_ranks(Gn)
Gn_adj = binarize(Gn_adj)

# plot graph
heatmap(
    Gn_adj,
    inner_hier_labels=simple_classes,
    sort_nodes=True,
    figsize=(15, 15),
    title=use_graph,
)

# compute and plot directed normalized laplacian
L = to_laplace(Gn_adj, form="R-DAD", regularizer=1)
heatmap(
    L,
    inner_hier_labels=simple_classes,
    sort_nodes=True,
    figsize=(15, 15),
    title="Normalized Laplacian",
)

# compute lse, plot with simple class labels
lse = AdjacencySpectralEmbed(n_components=n_components)
latent = lse.fit_transform(L)
latent = np.concatenate(latent, axis=-1)
pairplot(latent, labels=simple_classes)


#%% do clustering
gc = GaussianCluster(min_components=2, max_components=15, covariance_type="all")

n_init = 50
sim_mat = np.zeros((latent.shape[0], latent.shape[0]))

for i in tqdm(range(n_init)):
    assignments = gc.fit_predict(latent)
    for c in np.unique(assignments):
        inds = np.where(assignments == c)[0]
        sim_mat[np.ix_(inds, inds)] += 1
sim_mat = sim_mat / n_init
heatmap(sim_mat)

#%% consensus clusteting
ase = AdjacencySpectralEmbed(n_components=5)
sim_latent = ase.fit_transform(sim_mat)
pairplot(sim_latent, labels=simple_classes)
c = GaussianCluster(min_components=2, max_components=10)
overall = c.fit_predict(sim_latent)

heatmap(sim_mat, inner_hier_labels=overall, sort_nodes=True)
heatmap(Gn_adj, inner_hier_labels=overall, sort_nodes=True)

pairplot(latent[:, [0, 1, 2, 4, 5, 6]], overall)


#%% look at the cells in class
c = 4
pred_inds = np.where(overall == c)[0]
coi_df = pd.DataFrame(columns=("id", "name", "class"))
coi_df["id"] = ids[pred_inds]
coi_df["name"] = names[pred_inds]
coi_df["class"] = classes[pred_inds]
coi_df
coi_df.to_csv("maggot_models/notebooks/unknown_class_BDP_1.csv", index=False)


#%%

