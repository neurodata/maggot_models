#%%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from graspy.plot import gridplot, heatmap
from src.data import load_networkx
from src.utils import meta_to_array, savefig
from src.visualization import incidence_plot

plt.style.use("seaborn-white")
sns.set_palette("deep")

graph = load_networkx("Gn")

meta_df = pd.read_csv(
    "maggot_models/data/raw/Maggot-Brain-Connectome/4-color-matrices_Brain/2019-09-18-v2/brain_meta-data.csv",
    index_col=0,
)
print(meta_df.head())

pair_df = pd.read_csv(
    "maggot_models/data/raw/Maggot-Brain-Connectome/pairs/knownpairsatround5.csv"
)

print(pair_df.head())

left_nodes = pair_df["leftid"].values.astype(str)
right_nodes = pair_df["rightid"].values.astype(str)

left_right_pairs = list(zip(left_nodes, right_nodes))

left_nodes_unique, left_nodes_counts = np.unique(left_nodes, return_counts=True)
left_duplicate_inds = np.where(left_nodes_counts >= 2)[0]
left_duplicate_nodes = left_nodes_unique[left_duplicate_inds]
# left_nodes = np.setdiff1d(left_nodes, left_duplicate_nodes)

right_nodes_unique, right_nodes_counts = np.unique(right_nodes, return_counts=True)
right_duplicate_inds = np.where(right_nodes_counts >= 2)[0]
right_duplicate_nodes = right_nodes_unique[right_duplicate_inds]
# right_nodes = np.setdiff1d(right_nodes, right_duplicate_nodes)

left_nodes = []
right_nodes = []
for left, right in left_right_pairs:
    if left not in left_duplicate_nodes and right not in right_duplicate_nodes:
        if left in graph and right in graph:
            left_nodes.append(left)
            right_nodes.append(right)

side_labels = np.array(len(left_nodes) * ["left"] + len(right_nodes) * ["right"])
nodelist = np.concatenate((left_nodes, right_nodes)).astype(str)

matched_graph = graph.subgraph(nodelist)

# for node in nodelist:
#     if node not in graph:
#         print(node)


adj_df = nx.to_pandas_adjacency(matched_graph, nodelist=nodelist)

# class_labels = meta_to_array(matched_graph, "Class")
class_labels = meta_df.loc[nodelist.astype(int), "Class"].values
adj = adj_df.values

#%%
heatmap(
    adj,
    inner_hier_labels=class_labels,
    outer_hier_labels=side_labels,
    transform="simple-all",
    figsize=(30, 30),
    hier_label_fontsize=10,
    sort_nodes=False,
)


#%%
n_per_side = len(left_nodes)

left_left_adj = adj[:n_per_side, :n_per_side]
left_right_adj = adj[:n_per_side, n_per_side:]
right_right_adj = adj[n_per_side:, n_per_side:]
right_left_adj = adj[n_per_side:, :n_per_side]

#%%
left_left_edges = left_left_adj.ravel()
left_right_edges = left_right_adj.ravel()
right_right_edges = right_right_adj.ravel()
right_left_edges = right_left_adj.ravel()

left_edges = np.concatenate((left_left_edges, left_right_edges))
right_edges = np.concatenate((right_right_edges, right_left_edges))

plt.figure(figsize=(15, 15))
sns.scatterplot(left_edges, right_edges)

#%%
mean_within_adj = (left_left_adj + right_right_adj) / 2
heatmap(
    mean_within_adj,
    inner_hier_labels=class_labels[:n_per_side],
    transform="simple-all",
    figsize=(30, 30),
    hier_label_fontsize=10,
    sort_nodes=False,
)

#%%
from graspy.embed import LaplacianSpectralEmbed
from graspy.plot import pairplot

lse = LaplacianSpectralEmbed(form="R-DAD")
latent = lse.fit_transform(mean_within_adj)
latent = np.concatenate(latent, axis=-1)
pairplot(latent)

#%%
graph_types = ["Gaan", "Gadn", "Gdan", "Gddn"]
adjs = []

for g in graph_types:
    g = load_networkx("Gn")
    matched_graph = g.subgraph(nodelist)
    adj_df = nx.to_pandas_adjacency(matched_graph, nodelist=nodelist)
    adj = adj_df.values
    adjs.append(adj)
# class_labels = meta_df.loc[nodelist.astype(int), "Class"]

from graspy.embed import OmnibusEmbed
from graspy.utils import pass_to_ranks

omni = OmnibusEmbed(n_components=2)
adjs = [pass_to_ranks(a) for a in adjs]
omni_latent = omni.fit_transform(adjs)
omni_latent = np.concatenate(omni_latent, axis=-1)
omni_latent.shape

cat_omni_latent = np.concatenate(omni_latent, axis=-1)
cat_omni_latent.shape
#%%
pairplot(cat_omni_latent, labels=side_labels)

#%%
mean_omni_latent = (cat_omni_latent[:n_per_side] + cat_omni_latent[n_per_side:]) / 2

pairplot(mean_omni_latent, labels=class_labels[:n_per_side])


#%%
