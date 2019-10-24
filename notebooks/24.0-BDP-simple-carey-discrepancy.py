#%% Load
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.plot import gridplot, heatmap, pairplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graspy.utils import augment_diagonal, pass_to_ranks
from graspy.embed import OmnibusEmbed
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


#%%
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

#%% Omni the left and right, using the sum matrix, RAW
ptr = False


adj = get_paired_adj("G", nodelist)
if ptr:
    adj = pass_to_ranks(adj)
left_left_adj = adj[:n_per_side, :n_per_side]
right_right_adj = adj[n_per_side:, n_per_side:]

adjs = [left_left_adj, right_right_adj]

omni = OmnibusEmbed(n_components=None)
latents = omni.fit_transform(adjs)
latents = np.concatenate(latents, axis=-1)
diff = latents[0] - latents[1]
norm_diff_summed = np.linalg.norm(diff, axis=1)

sns.distplot(norm_diff_summed)

#%% Now do the same thing, but incorporate the 4-colors

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
left_color_adjs = []
right_color_adjs = []
for t in graph_types:
    adj = get_paired_adj(t, nodelist)
    if ptr:
        adj = pass_to_ranks(adj)
    left_left_adj = adj[:n_per_side, :n_per_side]
    right_right_adj = adj[n_per_side:, n_per_side:]
    left_color_adjs.append(left_left_adj)
    right_color_adjs.append(right_right_adj)

color_adjs = left_color_adjs + right_color_adjs
omni = OmnibusEmbed(n_components=None)
latents = omni.fit_transform(color_adjs)
latents = np.concatenate(latents, axis=-1)
left_latents = latents[:4, :, :]
right_latents = latents[4:, :, :]
left_latent_mean = left_latents.mean(axis=0)
right_latent_mean = right_latents.mean(axis=0)

diff = left_latent_mean - right_latent_mean
norm_diff_colors = np.linalg.norm(diff, axis=1)
sns.distplot(norm_diff_colors)

#%% now compute ranks
from scipy.stats import rankdata
from src.utils import savefig

rank_summed = rankdata(norm_diff_summed)[::-1]
rank_colors = rankdata(norm_diff_colors)[::-1]

rank_diff = np.abs(rank_summed - rank_colors)

reciprocal_diff = np.abs(1 / rank_summed - 1 / rank_colors)


result_df = pd.DataFrame()
result_df["Left ID"] = left_nodes
result_df["Right ID"] = right_nodes
result_df["Flat Distance"] = norm_diff_summed
result_df["4color Distance"] = norm_diff_colors
result_df["Flat Rank"] = rank_summed
result_df["4color Rank"] = rank_colors
result_df["Reciprocal Rank Diff"] = reciprocal_diff
result_df["Rank Diff"] = rank_diff

plt.figure(figsize=(10, 10))
sns.set_context("talk", font_scale=1)
sns.scatterplot(data=result_df, x="Flat Distance", y="4color Distance", alpha=0.8)
savefig("color_distance_comparison", fmt="png", dpi=200)
plt.figure(figsize=(10, 10))

sns.scatterplot(
    data=result_df,
    x="Flat Distance",
    y="4color Distance",
    # hue="Reciprocal Rank Diff",
    alpha=0.8,
    palette="gist_heat_r",
    size="Reciprocal Rank Diff",
    sizes=(10, 1000),
)
savefig("color_distance_comparison_heat", fmt="png", dpi=200)

plt.figure(figsize=(10, 10))
sns.distplot(reciprocal_diff)

#%%
print("Sorted by reciprocal rank diff")
result_df.sort_values("Reciprocal Rank Diff", ascending=False, inplace=True)
print(result_df.head())

print("Sorted by rank diff")
result_df.sort_values("Rank Diff", ascending=False, inplace=True)
print(result_df.head())

plt.figure(figsize=(20, 10))
sns.distplot(rank_diff)

