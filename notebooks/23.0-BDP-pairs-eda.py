#%% Load
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.plot import gridplot, heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


#%% plot normalized heatmap, sorted
adj = get_paired_adj("Gn", nodelist)

heatmap(
    adj,
    inner_hier_labels=class_labels,
    outer_hier_labels=side_labels,
    transform="simple-all",
    figsize=(30, 30),
    hier_label_fontsize=10,
    sort_nodes=False,
)

#%%  plot edge weight agreement for the normalized graph

left_edges_norm, right_edges_norm = get_paired_edges(adj)
plot_df = pd.DataFrame(columns=("Left edge weight", "Right edge weight"))
plot_df["Left edge weight"] = left_edges_norm
plot_df["Right edge weight"] = right_edges_norm

plt.figure(figsize=(15, 15))
sns.set_context("talk", font_scale=1.5)
sns.scatterplot(x="Left edge weight", y="Right edge weight", data=plot_df, alpha=0.5)
plt.title("Normalized graph")

#%% plot edge weight agreement for the raw graph
adj = get_paired_adj("G", nodelist)
left_edges_raw, right_edges_raw = get_paired_edges(adj)
plot_df = pd.DataFrame(columns=("Left edge weight", "Right edge weight"))
plot_df["Left edge weight"] = left_edges_raw
plot_df["Right edge weight"] = right_edges_raw

plt.figure(figsize=(15, 15))
sns.set_context("talk", font_scale=1.5)
sns.scatterplot(x="Left edge weight", y="Right edge weight", data=plot_df, alpha=0.2)
plt.title("Raw graph")


#%% look at different thresholds for the raw and normalized graphs

thresh_range = np.linspace(0, 0.5, 30)
eps = 0.002

left_gone = []
right_gone = []

for t in thresh_range:
    temp_left_edges = left_edges_norm.copy()
    temp_left_edges[temp_left_edges < t] = 0
    temp_right_edges = right_edges_norm.copy()
    temp_right_edges[temp_right_edges < t] = 0
    right_not_in_left = calc_edge_agreement(temp_right_edges, temp_left_edges)
    left_not_in_right = calc_edge_agreement(temp_left_edges, temp_right_edges)
    right_gone.append(right_not_in_left)
    left_gone.append(left_not_in_right)

plot_df = pd.DataFrame(
    columns=("Edge weight threshold", "Contralateral edges missing", "Side")
)
missing_edges = np.concatenate((left_gone, right_gone))
plot_df["Contralateral edges missing"] = missing_edges
plot_df["Edge weight threshold"] = np.concatenate((thresh_range, thresh_range + eps))
plot_df["Side"] = len(thresh_range) * ["Left"] + len(thresh_range) * ["Right"]
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Edge weight threshold", y="Contralateral edges missing", data=plot_df, hue="Side"
)
plt.ylabel("Contralateral edges missing")
plt.xlabel("Edge weight threshold")
plt.title("Normalized graph")

thresh_range = np.linspace(0, 100, 30)
eps = 0.5
left_gone = []
right_gone = []

for t in thresh_range:
    temp_left_edges = left_edges_raw.copy()
    temp_left_edges[temp_left_edges < t] = 0
    temp_right_edges = right_edges_raw.copy()
    temp_right_edges[temp_right_edges < t] = 0
    right_not_in_left = calc_edge_agreement(temp_right_edges, temp_left_edges)
    left_not_in_right = calc_edge_agreement(temp_left_edges, temp_right_edges)
    right_gone.append(right_not_in_left)
    left_gone.append(left_not_in_right)

plot_df = pd.DataFrame(
    columns=("Edge weight threshold", "Contralateral edges missing", "Side")
)
missing_edges = np.concatenate((left_gone, right_gone))
plot_df["Contralateral edges missing"] = missing_edges
plot_df["Edge weight threshold"] = np.concatenate((thresh_range, thresh_range + eps))
plot_df["Side"] = len(thresh_range) * ["Left"] + len(thresh_range) * ["Right"]
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Edge weight threshold", y="Contralateral edges missing", data=plot_df, hue="Side"
)
plt.ylabel("Contralateral edges missing")
plt.xlabel("Edge weight threshold")
plt.title("Raw graph")

thresh_range = np.linspace(0, 100, 30)
eps = 0.5
left_gone = []
right_gone = []

for t in thresh_range:
    temp_left_edges = left_edges_raw.copy()
    temp_left_edges[temp_left_edges < t] = 0
    temp_right_edges = right_edges_raw.copy()
    temp_right_edges[temp_right_edges < t] = 0
    right_not_in_left = calc_edge_agreement(temp_right_edges, temp_left_edges)
    left_not_in_right = calc_edge_agreement(temp_left_edges, temp_right_edges)
    right_not_in_left /= np.count_nonzero(temp_right_edges) + 1
    left_not_in_right /= np.count_nonzero(temp_left_edges) + 1
    right_gone.append(right_not_in_left)
    left_gone.append(left_not_in_right)

plot_df = pd.DataFrame(
    columns=("Edge weight threshold", "Contralateral edges missing", "Side")
)
missing_edges = np.concatenate((left_gone, right_gone))
plot_df["Contralateral edges missing"] = missing_edges
plot_df["Edge weight threshold"] = np.concatenate((thresh_range, thresh_range + eps))
plot_df["Side"] = len(thresh_range) * ["Left"] + len(thresh_range) * ["Right"]
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Edge weight threshold", y="Contralateral edges missing", data=plot_df, hue="Side"
)
plt.ylabel("Contralateral edges missing")
plt.xlabel("Edge weight threshold")
plt.title("Raw graph")


#%% plot the mean hemisphere induced subgraph for the raw matrix
name_map = {
    "CN": "CN",
    "DANs": "MBIN",
    "KCs": "KC",
    "LHN": "LHN",
    "LHN; CN": "CN/LHN",
    "MBINs": "MBIN",
    "MBON": "MBON",
    "MBON; CN": "MBON",
    "OANs": "MBIN",
    "ORN mPNs": "PN",
    "ORN uPNs": "PN",
    "tPNs": "PN",
    "vPNs": "PN",
    "Unidentified": "Unknown",
    "Other": "Unknown",
}
simple_class_labels = np.array(itemgetter(*class_labels)(name_map))
# side_map = {"left": "Left", "right": "Right"}
# side_labels = np.array(itemgetter(*side_labels)(side_map))

adj = get_paired_adj("G", nodelist)
n_per_side = adj.shape[0] // 2
left_left_adj = adj[:n_per_side, :n_per_side]
left_right_adj = adj[:n_per_side, n_per_side:]
right_right_adj = adj[n_per_side:, n_per_side:]
right_left_adj = adj[n_per_side:, :n_per_side]
mean_within_adj = (left_left_adj + right_right_adj) / 2
heatmap(
    mean_within_adj,
    inner_hier_labels=simple_class_labels[:n_per_side],
    transform="simple-all",
    figsize=(30, 30),
    hier_label_fontsize=18,
    sort_nodes=False,
)

#%%

mean_across_adj = (left_right_adj + right_left_adj) / 2
ax = heatmap(
    mean_across_adj,
    inner_hier_labels=simple_class_labels[:n_per_side],
    transform="simple-all",
    figsize=(30, 30),
    hier_label_fontsize=18,
    sort_nodes=False,
)

bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
ax.annotate(
    "figure fraction : 0, 0",
    xy=(0, 0.9),
    xycoords="figure fraction",
    xytext=(20, 20),
    textcoords="offset points",
    ha="left",
    va="top",
    bbox=bbox_args,
    arrowprops=arrow_args,
)

#%% find the most-disagreeing cells
diffs = np.zeros(n_per_side)

for i in range(n_per_side):
    left_out_edges = adj[i, :]
    left_in_edges = adj[:, i]
    right_out_edges = adj[i + n_per_side, :]
    right_in_edges = adj[:, i + n_per_side]
    left_edges = np.concatenate((left_out_edges, left_in_edges))
    right_edges = np.concatenate((right_out_edges, right_in_edges))
    diff = np.abs(left_edges - right_edges).sum() / (
        left_edges.sum() + right_edges.sum()
    )
    diffs[i] = diff

plt.figure(figsize=(15, 10))
sns.distplot(diffs)
plt.xlabel("# differing synapses")

diff_df = pd.DataFrame(columns=("Difference", "Left", "Right"))
diff_df["Difference"] = diffs
diff_df["Left"] = left_nodes
diff_df["Right"] = right_nodes
diff_df.sort_values("Difference", ascending=False, inplace=True)
print(diff_df.head())
sort_inds = diff_df.index.values

#%%
mean_within_adj_sorted = mean_within_adj[np.ix_(sort_inds, sort_inds)]
heatmap(
    mean_within_adj_sorted, transform="simple-all", figsize=(30, 30), sort_nodes=False
)
#%%
degree = adj.sum(axis=0) + adj.sum(axis=1) - np.diag(adj)
diff_within = np.abs(left_left_adj - right_right_adj)
diff_across = np.abs(left_right_adj - right_left_adj)
diff_total = diff_within + diff_across
diff_degree_total = diff_total.sum(axis=0) + diff_total.sum(axis=1)
degree_mean = degree[:n_per_side] + degree[n_per_side:]
diff_proportion = diff_degree_total / degree_mean
sns.distplot(diff_proportion)
#%% Junk
# data_df = pd.DataFrame(columns=("Left edge", "Right edge"))
# data_df["Left edge"] = left_edges
# data_df["Right edge"] = right_edges
# plt.figure(figsize=(15, 15))
# sns.lmplot("Left edge", "Right edge", data=data_df)


# sns.jointplot(
#     x="Left edge weight", y="Right edge weight", data=plot_df, kind="hex", height=15
# )
