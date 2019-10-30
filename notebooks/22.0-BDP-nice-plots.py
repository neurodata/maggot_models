#%%
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.plot import gridplot, heatmap

#%%
from matplotlib import lines
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.data import load_networkx
from src.utils import meta_to_array, savefig
from src.visualization import countplot, freqplot, incidence_plot


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
    transform="binarize",
    figsize=(30, 30),
    hier_label_fontsize=10,
    sort_nodes=False,
    cbar=False,
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
plt.axis("equal")
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


#%% try the MB

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
graph_type_labels = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]

color_graphs = []


# def get_paired_adj(graph_type, nodelist):
#     graph = load_networkx(graph_type)
#     matched_graph = graph.subgraph(nodelist)
#     adj_df = nx.to_pandas_adjacency(matched_graph, nodelist=nodelist)
#     adj = adj_df.values
#     return adj


# def get_mean_induced(matched_adj):
#     n_per_side = matched_adj.shape[0] // 2
#     left_left_adj = matched_adj[:n_per_side, :n_per_side]
#     # left_right_adj = matched_adj[:n_per_side, n_per_side:]
#     right_right_adj = matched_adj[n_per_side:, n_per_side:]
#     # right_left_adj = matched_adj[n_per_side:, :n_per_side]
#     mean_adj = (left_left_adj + right_right_adj) / 2
#     return mean_adj
flat_g = load_networkx("G", version="mb_2019-09-23")
mb_class_labels = meta_to_array(flat_g, "Class")
side_labels = meta_to_array(flat_g, "Hemisphere")
right_inds = np.where(side_labels == "right")[0]
adj = nx.to_pandas_adjacency(flat_g).values
adj = adj[np.ix_(right_inds, right_inds)]
degrees = adj.sum(axis=0) + adj.sum(axis=1)
sort_inds = np.argsort(degrees)[::-1]
mb_class_labels = mb_class_labels[right_inds]
mb_class_labels = mb_class_labels[sort_inds]

mb_color_graphs = []
for t in graph_types:
    color_g = load_networkx(t, version="mb_2019-09-23")
    adj = nx.to_pandas_adjacency(color_g).values
    adj = adj[np.ix_(right_inds, right_inds)]
    adj = adj[np.ix_(sort_inds, sort_inds)]
    mb_color_graphs.append(adj)


name_map = {
    "APL": "APL",
    "Gustatory PN": "PN",
    "KC 1 claw": "KC",
    "KC 2 claw": "KC",
    "KC 3 claw": "KC",
    "KC 4 claw": "KC",
    "KC 5 claw": "KC",
    "KC 6 claw": "KC",
    "KC young": "KC",
    "MBIN": "MBIN",
    "MBON": "MBON",
    "ORN mPN": "PN",
    "ORN uPN": "PN",
    "Unknown PN": "PN",
    "tPN": "PN",
    "vPN": "PN",
}
mb_class_labels = np.array(itemgetter(*mb_class_labels)(name_map))


fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax = ax.ravel()
for i, g in enumerate(mb_color_graphs):
    heatmap(
        g,
        inner_hier_labels=mb_class_labels,
        transform="simple-all",
        hier_label_fontsize=18,
        sort_nodes=False,
        ax=ax[i],
        cbar=False,
        title=graph_type_labels[i],
        title_pad=70,
        font_scale=1.7,
    )
plt.suptitle("Right Mushroom Body", fontsize=45, x=0.525, y=1.02)
plt.tight_layout()
arrow_args = dict(
    arrowstyle="-|>",
    color="k",
    connectionstyle="arc3,rad=-0.4",  # "angle3,angleA=90,angleB=90"
)

t = ax[0].annotate("Target", xy=(0.061, 0.93), xycoords="figure fraction")

ax[0].annotate(
    "Source", xy=(0, 0.5), xycoords=t, xytext=(-1.4, -2.1), arrowprops=arrow_args
)

# # ax.add_patch(el)

# ax[0].annotate(
#     "$->$",
#     xy=(0.1, 0.9),
#     xycoords="figure fraction",
#     # xytext=(-150, -140),
#     textcoords="offset points",
#     bbox=dict(boxstyle="round", fc="0.8"),
#     arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=90,angleB=0,rad=10"),
# )

bbox_args = dict(boxstyle="round", fc="0.8")
# text = ax[0].annotate(
#     "xy=(0, 1)",
#     xy=(0.1, 1),
#     xycoords="figure fraction",
#     xytext=(0, -20),
#     textcoords="offset points",
#     ha="center",
#     va="top",
#     bbox=bbox_args,
#     arrowprops=arrow_args,
# )

# ax[0].annotate(
#     "xy=(0.5, 0)",
#     xy=(0.1, 0.3),
#     xycoords=text,
#     xytext=(0, -20),
#     textcoords="offset points",
#     ha="center",
#     va="top",
#     bbox=bbox_args,
#     arrowprops=arrow_args,
# )

#%%

from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.utils import pass_to_ranks
from graspy.plot import pairplot


sum_adj = np.sum(np.array(mb_color_graphs), axis=0)

n_components = 4

#
ptr_adj = pass_to_ranks(sum_adj)
ase = AdjacencySpectralEmbed(n_components=n_components)
sum_latent = ase.fit_transform(ptr_adj)
sum_latent = np.concatenate(sum_latent, axis=-1)
pairplot(sum_latent, labels=mb_class_labels)

ptr_color_adjs = [pass_to_ranks(a) for a in mb_color_graphs]
# graph_sum = [np.sum(a) for a in mb_color_graphs]
# ptr_color_adjs = [ptr_color_adjs[i] + (1 / graph_sum[i]) for i in range(4)]
omni = OmnibusEmbed(n_components=n_components // 4)
color_latent = omni.fit_transform(ptr_color_adjs)
color_latent = np.concatenate(color_latent, axis=-1)
color_latent = np.concatenate(color_latent, axis=-1)
pairplot(color_latent, labels=mb_class_labels)

from graspy.embed import MultipleASE

mase = MultipleASE(n_components=n_components)
mase_latent = mase.fit_transform(ptr_color_adjs)
mase_latent = np.concatenate(mase_latent, axis=-1)
pairplot(mase_latent, labels=mb_class_labels)


#%%
graph = load_networkx("G")
class_labels = meta_to_array(graph, "Class")

name_map = {
    "CN": "CN",
    "DANs": "DAN",
    "KCs": "KC",
    "LHN": "LHN",
    "LHN; CN": "CN",
    "MBINs": "MBIN",
    "MBON": "MBON",
    "MBON; CN": "MBON",
    "OANs": "OAN",
    "ORN mPNs": "mPN",
    "ORN uPNs": "uPN",
    "tPNs": "tPN",
    "vPNs": "vPN",
    "Unidentified": "Unknown",
    "Other": "Unknown",
}
simple_class_labels = np.array(itemgetter(*class_labels)(name_map))

df_adj = nx.to_pandas_adjacency(graph)
adj = df_adj.values

classes = meta_to_array(graph, "Class")
classes = classes.astype("<U64")
classes[classes == "MBON; CN"] = "MBON"
print(np.unique(classes))

PREDEFINED_CLASSES = [
    "DANs",
    "KCs",
    "MBINs",
    "MBON",
    "OANs",
    "ORN mPNs",
    "ORN uPNs",
    "tPNs",
    "vPNs",
]


nx_ids = np.array(list(graph.nodes()), dtype=int)
df_ids = df_adj.index.values.astype(int)
print("nx indexed same as pd")
print(np.array_equal(nx_ids, df_ids))
cell_ids = df_ids

PREDEFINED_IDS = []
for c, id in zip(classes, cell_ids):
    if c in PREDEFINED_CLASSES:
        PREDEFINED_IDS.append(id)

IDS_TO_INDS_MAP = dict(zip(cell_ids, range(len(cell_ids))))


def ids_to_inds(ids):
    inds = [IDS_TO_INDS_MAP[k] for k in ids]
    return inds


def proportional_search(adj, class_ind_map, or_classes, ids, thresh):
    """finds the cell ids of neurons who receive a certain proportion of their 
    input from one of the cells in or_classes 
    
    Parameters
    ----------
    adj : np.array
        adjacency matrix, assumed to be normalized so that columns sum to 1
    class_map : dict
        keys are class names, values are arrays of indices describing where that class
        can be found in the adjacency matrix
    or_classes : list 
        which classes to consider for the input thresholding. Neurons will be selected 
        which satisfy ANY of the input threshold criteria
    ids : np.array
        names of each cell 
    """
    if not isinstance(thresh, list):
        thresh = [thresh]

    pred_cell_ids = []
    for i, class_name in enumerate(or_classes):
        inds = class_ind_map[class_name]  # indices for neurons of that class
        from_class_adj = adj[inds, :]  # select the rows corresponding to that class
        prop_input = from_class_adj.sum(axis=0)  # sum input from that class
        # prop_input /= adj.sum(axis=0)
        if thresh[i] >= 0:
            flag_inds = np.where(prop_input >= thresh[i])[0]  # inds above threshold
        elif thresh[i] < 0:
            flag_inds = np.where(prop_input <= -thresh[i])[0]  # inds below threshold
        pred_cell_ids += list(ids[flag_inds])  # append to cells which satisfied

    pred_cell_ids = np.unique(pred_cell_ids)

    pred_cell_ids = np.setdiff1d(pred_cell_ids, PREDEFINED_IDS)

    return pred_cell_ids


def update_class_map(cell_ids, classes):
    unique_classes, inverse_classes = np.unique(classes, return_inverse=True)
    class_ind_map = {}
    class_ids_map = {}
    for i, class_name in enumerate(unique_classes):
        inds = np.where(inverse_classes == i)[0]
        ids = cell_ids[inds]
        class_ind_map[class_name] = inds
        class_ids_map[class_name] = ids
    return class_ids_map, class_ind_map


og_class_ids_map, og_class_ind_map = update_class_map(cell_ids, classes)

sums = []
degrees = adj.sum(axis=0)
degrees[degrees == 0] = 1
for c in PREDEFINED_CLASSES:
    inds = og_class_ind_map[c]
    input_sum = adj[inds, :].sum(axis=0) / degrees
    sums.append(input_sum)
sums = np.array(sums).T

from sklearn.decomposition import PCA

pairplot(
    sums, labels=simple_class_labels, col_names=PREDEFINED_CLASSES, palette="tab20"
)
pca = PCA(n_components=6)
projections = pca.fit_transform(sums)
pairplot(projections, labels=simple_class_labels, palette="tab20")
unk_inds = og_class_ind_map["Other"]
pairplot(projections[unk_inds])

#%%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans_labels = kmeans.fit_predict(sums)
pairplot(sums, labels=kmeans_labels, col_names=PREDEFINED_CLASSES, palette="Set1")

class_names = list(PREDEFINED_CLASSES.copy())
class_ind_map = og_class_ind_map
class_ind_map["New 1"] = np.where(kmeans_labels == 0)[0]
class_ind_map["New 2"] = np.where(kmeans_labels == 1)[0]
class_names.append("New 1")
class_names.append("New 2")
# %%
sums = []
for c in class_names:
    inds = class_ind_map[c]
    input_sum = adj[inds, :].sum(axis=0) / degrees
    sums.append(input_sum)
sums = np.array(sums).T
kmeans = KMeans(n_clusters=2)
kmeans_labels = kmeans.fit_predict(sums)
pairplot(sums, labels=kmeans_labels, col_names=class_names, palette="Set1")

# %%
