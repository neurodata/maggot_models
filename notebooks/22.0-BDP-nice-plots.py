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
from src.visualization import incidence_plot, countplot, freqplot


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


palette = "Set1"
sns.set_context("talk", font_scale=1.2)
plt.style.use("seaborn-white")
sns.set_palette(palette)
graph_type_labels = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]


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
#%%
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
n_per_side = len(left_nodes)


adj_df = nx.to_pandas_adjacency(matched_graph, nodelist=nodelist)

# class_labels = meta_to_array(matched_graph, "Class")
class_labels = meta_df.loc[nodelist.astype(int), "Class"].values

from operator import itemgetter

name_map = {
    "CN": "CN",
    "DANs": "MB",
    "KCs": "MB",
    "LHN": "LHN",
    "LHN; CN": "CN",
    "MBINs": "MB",
    "MBON": "MB",
    "MBON; CN": "MB",
    "OANs": "MB",
    "ORN mPNs": "PN",
    "ORN uPNs": "PN",
    "tPNs": "PN",
    "vPNs": "PN",
    "Unidentified": "Unknown",
    "Other": "Unknown",
}
simple_class_labels = np.array(itemgetter(*class_labels)(name_map))
side_map = {"left": "Left", "right": "Right"}
side_labels = np.array(itemgetter(*side_labels)(side_map))

adj = adj_df.values

sns.set_palette("deep")
heatmap(
    adj,
    inner_hier_labels=simple_class_labels,
    outer_hier_labels=side_labels,
    transform="simple-all",
    figsize=(30, 30),
    hier_label_fontsize=10,
    sort_nodes=False,
)

gridplot(
    [adj],
    inner_hier_labels=simple_class_labels,
    outer_hier_labels=side_labels,
    transform="simple-all",
    height=20,
    hier_label_fontsize=10,
    sort_nodes=False,
    sizes=(2, 10),
)

#%% Gridplot for the colors
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
color_graphs = []
for t in graph_types:
    g = load_networkx(t)
    matched_g = g.subgraph(nodelist)
    g_df = nx.to_pandas_adjacency(matched_g, nodelist=nodelist)
    color_graphs.append(g_df.values)

gp = gridplot(
    color_graphs,
    labels=graph_type_labels,
    outer_hier_labels=side_labels,
    inner_hier_labels=simple_class_labels,
    transform="simple-all",
    height=20,
    hier_label_fontsize=22,
    sort_nodes=False,
    sizes=(4, 20),
    palette=palette,
    title=r"All paired neurons",
    font_scale=2.2,
    title_pad=260,
    legend_name="Edge type",
)
savefig("color_graphs_pretty", pad_inches=1, bbox_inches="tight", fmt="png", dpi=200)

countplot(color_graphs, graph_type_labels)
savefig(
    "color_graphs_pretty_prop", bbox_inches="tight", pad_inches=0.5, fmt="png", dpi=200
)
#%% try just one induced subgraph
# Plot left to left
color_graphs_small = [c[:n_per_side, :n_per_side] for c in color_graphs]
sns.set_palette(palette, desat=0.7)
sns.set_context("talk", font_scale=1.2)
gridplot(
    color_graphs_small,
    labels=graph_type_labels,
    inner_hier_labels=simple_class_labels[:n_per_side],
    transform="simple-all",
    height=20,
    hier_label_fontsize=25,
    sort_nodes=False,
    sizes=(4, 20),
    palette=palette,
    title=r"Left $\to$ left hemisphere, paired neurons",
    font_scale=2.2,
    title_pad=160,
    legend_name="Edge type",
)
savefig("color_graphs_left", pad_inches=2, bbox_inches="tight", fmt="png", dpi=200)

countplot(color_graphs_small, graph_type_labels)
savefig(
    "color_graphs_left_prop", bbox_inches="tight", pad_inches=0.5, fmt="png", dpi=200
)
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# fig, ax = plt.subplots(figsize=(4, 4))
# ax.pie(counts, autopct="%1.0f%%", pctdistance=1.2)

# %%

#%%
from matplotlib import lines

# Plot left to right
color_graphs_small = [c[:n_per_side, :][:, n_per_side:] for c in color_graphs]
f = plt.figure()
p = gridplot(
    color_graphs_small,
    labels=graph_type_labels,
    inner_hier_labels=simple_class_labels[:n_per_side],
    transform="simple-all",
    height=20,
    hier_label_fontsize=18,
    sort_nodes=False,
    sizes=(4, 20),
    palette="Set1",
    title=r"Left $\to$ right hemisphere, paired neurons",
    font_scale=2,
    title_pad=150,
)
ax2 = plt.axes([0, 0, 1, 1])

x, y = np.array([[0.05, 0.1, 0.9], [0.05, 0.5, 0.9]])
line = lines.Line2D(x, y, lw=5.0, color="r", alpha=0.4)
ax2.add_line(line)

plt.show()
#%% Try the same but with heatmap
fig, ax = plt.subplots(2, 2, figsize=(40, 40))
ax = ax.ravel()
for i, t in enumerate(graph_type_labels):
    heatmap(
        color_graphs_small[i],
        inner_hier_labels=simple_class_labels[:n_per_side],
        transform="simple-all",
        hier_label_fontsize=18,
        sort_nodes=False,
        title=r"Left $\to$ left hemisphere, paired neurons",
        font_scale=1.75,
        title_pad=140,
        ax=ax[i],
        cbar=False,
    )
plt.tight_layout()


#%% try the MB

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
color_graphs = []

for t in graph_types:
    g = load_networkx(t)
    matched_g = g.subgraph(nodelist)
    g_df = nx.to_pandas_adjacency(matched_g, nodelist=nodelist)
    color_graphs.append(g_df.values)


mb_color_graphs = [g[np.ix_(mb_inds, mb_inds)] for g in color_graphs_small]
# mb_simple_labels = simple_class_labels[:n_per_side]
mb_simple_labels = simple_class_labels[mb_inds]

gridplot(
    mb_color_graphs,
    labels=graph_type_labels,
    inner_hier_labels=mb_simple_labels,
    transform="simple-all",
    height=20,
    hier_label_fontsize=15,
    sort_nodes=False,
    sizes=(16, 60),
    palette="Set1",
)

##########
#%%

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
