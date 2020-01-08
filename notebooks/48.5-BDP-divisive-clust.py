# %% [markdown]
# # Imports
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph
from src.embed import lse, preprocess_graph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob
from src.visualization import bartreeplot, get_color_dict, get_colors, sankey, screeplot
from src.graph import MetaGraph

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2019-12-18"

SAVEFIGS = True
SAVESKELS = False
SAVEOBJS = True

PTR = True
if PTR:
    ptr_type = "PTR"
else:
    ptr_type = "Raw"

brain_type = "Full Brain"
brain_type_short = "fullbrain"

GRAPH_TYPE = "Gad"
if GRAPH_TYPE == "Gad":
    graph_type = r"A $\to$ D"

N_INIT = 200

CLUSTER_METHOD = "graspy-gmm"
if CLUSTER_METHOD == "graspy-gmm":
    cluster_type = "GraspyGMM"
elif CLUSTER_METHOD == "auto-gmm":
    cluster_type = "AutoGMM"

EMBED = "LSE"
if EMBED == "LSE":
    embed_type = "LSE"

N_COMPONENTS = None


np.random.seed(23409857)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


def stashobj(obj, name, **kws):
    saveobj(obj, name, foldername=FNAME, save_on=SAVEOBJS, **kws)


# Set up plotting constants
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=0.8)

# %% [markdown]
# # Take 2


def sort_arrays(inds, *args):
    outs = []
    for a in args:
        sort_a = a[inds]
        outs.append(sort_a)
    return tuple(outs)


def sort_graph_and_meta(inds, adj, *args):
    outs = []
    meta_outs = sort_arrays(inds, *args)
    adj = adj[np.ix_(inds, inds)]
    outs = tuple([adj] + list(meta_outs))
    return outs


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


# %% [markdown]
# # Load data

mg = load_metagraph("Gadn", version=BRAIN_VERSION)
adj = mg.adj
skeleton_labels = mg.meta.index.values
# %% [markdown]
# # Deal with pairs
pair_df = pd.read_csv(
    "maggot_models/data/raw/Maggot-Brain-Connectome/pairs/knownpairsatround5.csv"
)

print(pair_df.head())

# extract valid node pairings
left_nodes = pair_df["leftid"].values
right_nodes = pair_df["rightid"].values

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
        if left in skeleton_labels and right in skeleton_labels:
            left_nodes.append(left)
            right_nodes.append(right)

pair_nodelist = np.concatenate((left_nodes, right_nodes))
all_nodelist = skeleton_labels
not_paired = np.setdiff1d(all_nodelist, pair_nodelist)
sorted_nodelist = np.concatenate((pair_nodelist, not_paired))

# sort the graph and metadata according to this
sort_map = dict(zip(sorted_nodelist, range(len(sorted_nodelist))))
perm_inds = np.array(itemgetter(*skeleton_labels)(sort_map))
sort_inds = invert_permutation(perm_inds)

mg.reindex(sort_inds)

side_labels = mg["Hemisphere"]
side_labels = side_labels.astype("<U2")
for i, l in enumerate(side_labels):
    if mg.meta.index.values[i] in not_paired:
        side_labels[i] = "U" + l
mg["Hemisphere"] = side_labels

# make the proper parts of the graph equal to the left/right averages
n_pairs = len(left_nodes)
adj = mg.adj
left_left_adj = adj[:n_pairs, :n_pairs]
left_right_adj = adj[:n_pairs, n_pairs : 2 * n_pairs]
right_right_adj = adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs]
right_left_adj = adj[n_pairs : 2 * n_pairs, :n_pairs]

sym_ipsi_adj = (left_left_adj + right_right_adj) / 2
sym_contra_adj = (left_right_adj + right_left_adj) / 2

sym_adj = adj.copy()
sym_adj[:n_pairs, :n_pairs] = sym_ipsi_adj
sym_adj[n_pairs : 2 * n_pairs, n_pairs : 2 * n_pairs] = sym_ipsi_adj
sym_adj[:n_pairs, n_pairs : 2 * n_pairs] = sym_contra_adj
sym_adj[n_pairs : 2 * n_pairs, :n_pairs] = sym_contra_adj

side_labels = mg["Hemisphere"]
f = gridplot(
    [sym_adj],
    transform="binarize",
    height=20,
    sizes=(10, 20),
    inner_hier_labels=side_labels,
    sort_nodes=False,
    palette="deep",
    legend=False,
)
stashfig("sym-adj")
mg = MetaGraph(sym_adj, mg.meta)

# %% [markdown]
# # Preprocess graph
n_verts = mg.n_verts
mg.make_lcc()
print(f"Removed {n_verts - mg.n_verts} when finding the LCC")
# old_n_verts = sym_adj.shape[0]
# sym_adj, class_labels, side_labels = preprocess_graph(
#     sym_adj, class_labels, side_labels
# )
# n_verts = sym_adj.shape[0]
# print(f"Removed {old_n_verts - n_verts} nodes")
# %% [markdown]
# # Embedding
n_verts = mg.n_verts
sym_adj = mg.adj
side_labels = mg["Hemisphere"]
class_labels = mg["Merge Class"]

latent, laplacian = lse(sym_adj, N_COMPONENTS, regularizer=None, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)
print(f"ZG chose dimension {latent_dim} + {latent_dim}")

plot_latent = np.concatenate(
    (latent[:, :3], latent[:, latent_dim : latent_dim + 3]), axis=-1
)
pairplot(plot_latent, labels=side_labels)

# take the mean for the paired cells, making sure to add back in the unpaired cells
sym_latent = (latent[:n_pairs] + latent[n_pairs : 2 * n_pairs]) / 2
sym_latent = np.concatenate((sym_latent, latent[2 * n_pairs :]))
latent = sym_latent

# make new labels
side_labels = np.concatenate((n_pairs * ["P"], side_labels[2 * n_pairs :]))
# this is assuming that the class labels are perfectly matches left right, probs not
class_labels = np.concatenate((class_labels[:n_pairs], class_labels[2 * n_pairs :]))
# skeleton labels are weird for now


plot_latent = np.concatenate(
    (latent[:, :3], latent[:, latent_dim : latent_dim + 3]), axis=-1
)
pairplot(plot_latent, labels=side_labels)


# %% [markdown]
# # Separate sensory modalities
print(np.unique(class_labels))
sensory_classes = ["mPN-m", "mPN-o", "mPN;FFN-m", "tPN", "uPN", "vPN"]
sensory_map = {
    "mPN-m": "multimodal",
    "mPN-o": "olfaction",
    "mPN;FFN-m": "multimodal",
    "tPN": "thermo",
    "uPN": "olfaction",
    "vPN": "visual",
}
is_sensory = np.vectorize(lambda s: s in sensory_classes)(class_labels)
inds = np.arange(len(class_labels))
sensory_inds = inds[is_sensory]
nonsensory_inds = inds[~is_sensory]
sensory_labels = class_labels[is_sensory]
sensory_labels = np.array(itemgetter(*sensory_labels)(sensory_map))
cluster_latent = latent[~is_sensory, :]

# %% [markdown]
# # Fitting divisive cluster model with GraspyGMM
name_base = f"-{cluster_type}-{embed_type}-{ptr_type}-{brain_type_short}-{GRAPH_TYPE}"

base = f"maggot_models/notebooks/outs/{FNAME}/objs/"
filename = base + "dc" + name_base + ".pickle"
clean_start = False
if os.path.isfile(filename) and not clean_start:
    print("Attempting to load file")
    with open(filename, "rb") as f:
        dc = pickle.load(f)
    print(f"Loaded file from {filename}")
else:
    print("Fitting DivisiveCluster model")
    start = timer()
    dc = DivisiveCluster(n_init=N_INIT, cluster_method=CLUSTER_METHOD)
    dc.fit(cluster_latent)
    end = end = timer()
    print()
    print(f"DivisiveCluster took {(end - start)/60.0} minutes to fit")
    print()
dc.print_tree(print_val="bic_ratio")
true_pred_labels = dc.predict(cluster_latent)
pred_labels = np.empty(shape=class_labels.shape, dtype="<U100")
pred_labels[nonsensory_inds] = true_pred_labels
pred_labels[sensory_inds] = sensory_labels


# %% [markdown]
# # Plotting and saving divisive cluster hierarchy results for GraspyGMM

stashobj(dc, "dc" + name_base)

n_classes = len(np.unique(class_labels))
class_color_dict = get_color_dict(class_labels, pal=cc.glasbey_cool)
pred_color_dict = get_color_dict(true_pred_labels, pal=cc.glasbey_warm)
all_color_dict = {**class_color_dict, **pred_color_dict}
stashobj(all_color_dict, "all_color_dict" + name_base)

class_colors = np.array(itemgetter(*class_labels)(class_color_dict))

# stashskel("known-skels" + name_base, skeleton_labels, class_labels, class_colors)
# stashskel(
#     "known-skels" + name_base,
#     skeleton_labels,
#     class_labels,
#     class_colors,
#     multiout=True,
# )


title = (
    f"Divisive hierarchical clustering,"
    + f" {cluster_type}, {embed_type} ({latent_dim} + {latent_dim}), {ptr_type},"
    + f" {brain_type}, {graph_type}"
)

fig, ax = plt.subplots(1, 1, figsize=(20, 30))
sankey(
    ax,
    class_labels[~is_sensory],
    true_pred_labels,
    aspect=20,
    fontsize=16,
    colorDict=all_color_dict,
)
ax.axis("off")
ax.set_title(title, fontsize=30)
stashfig("sankey" + name_base)

fig, ax = plt.subplots(1, 1, figsize=(20, 30))
sankey(
    ax,
    true_pred_labels,
    class_labels[~is_sensory],
    aspect=20,
    fontsize=16,
    colorDict=all_color_dict,
)
ax.axis("off")
ax.set_title(title, fontsize=30)
stashfig("sankey-inv" + name_base)

sns.set_context("talk", font_scale=0.8)

_, _, leaf_names = bartreeplot(
    dc,
    class_labels[~is_sensory],
    true_pred_labels,
    show_props=True,
    print_props=False,
    inverse_memberships=False,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-props" + name_base)
bartreeplot(
    dc,
    class_labels[~is_sensory],
    true_pred_labels,
    show_props=False,
    print_props=True,
    inverse_memberships=False,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-counts" + name_base)
bartreeplot(
    dc,
    class_labels[~is_sensory],
    true_pred_labels,
    show_props=True,
    inverse_memberships=True,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-props-inv" + name_base)
bartreeplot(
    dc,
    class_labels[~is_sensory],
    true_pred_labels,
    show_props=False,
    inverse_memberships=True,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-counts-inv" + name_base)


# pred_colors = np.array(itemgetter(*pred_labels)(pred_color_dict))
# stashskel("pred-skels" + name_base, skeleton_labels, pred_labels, pred_colors)
# stashskel(
#     "pred-skels" + name_base, skeleton_labels, pred_labels, pred_colors, multiout=True
# )

# %% [markdown]
# # Separate sensory modalities

# %% [markdown]
# # Plot the degree distribution for each predicted cluster

adj = sym_adj
in_edgesums = adj.sum(axis=0)
out_edgesums = adj.sum(axis=1)
in_degree = np.count_nonzero(adj, axis=0)
out_degree = np.count_nonzero(adj, axis=1)
out = []
for arr in [in_edgesums, out_edgesums, in_degree, out_degree]:
    mean = (arr[:n_pairs] + arr[n_pairs : 2 * n_pairs]) / 2
    out_arr = np.concatenate((mean, arr[2 * n_pairs :]))
    out.append(out_arr)
in_edgesums, out_edgesums, in_degree, out_degree = out

degree_df = pd.DataFrame()
degree_df["In edgesum"] = in_edgesums
degree_df["Out edgesum"] = out_edgesums
degree_df["In degree"] = in_degree
degree_df["Out degree"] = out_degree
degree_df["Total degree"] = in_degree + out_degree
degree_df["Cluster"] = pred_labels
degree_df["Class"] = class_labels

fg = sns.FacetGrid(
    degree_df,
    col="Cluster",
    col_wrap=6,
    sharex=True,
    sharey=False,
    hue="Cluster",
    hue_order=pred_color_dict.keys(),
    palette=pred_color_dict.values(),
    col_order=leaf_names,
)

fg.map(sns.distplot, "In edgesum", norm_hist=True, kde=False)
fg.set(yticks=[], yticklabels=[])
stashfig("in-edgesum" + name_base)

fg = sns.FacetGrid(
    degree_df,
    col="Cluster",
    col_wrap=6,
    sharex=True,
    sharey=False,
    hue="Cluster",
    hue_order=pred_color_dict.keys(),
    palette=pred_color_dict.values(),
    col_order=leaf_names,
)

fg.map(sns.distplot, "Out edgesum", norm_hist=True, kde=False)
fg.set(yticks=[], yticklabels=[])
stashfig("out-edgesum" + name_base)

fg = sns.FacetGrid(
    degree_df,
    col="Cluster",
    col_wrap=6,
    sharex=True,
    sharey=False,
    hue="Cluster",
    hue_order=pred_color_dict.keys(),
    palette=pred_color_dict.values(),
    col_order=leaf_names,
)

fg.map(sns.distplot, "In degree", norm_hist=True, kde=False)
fg.set(yticks=[], yticklabels=[])
stashfig("in-degree" + name_base)

fg = sns.FacetGrid(
    degree_df,
    col="Cluster",
    col_wrap=6,
    sharex=True,
    sharey=False,
    hue="Cluster",
    hue_order=pred_color_dict.keys(),
    palette=pred_color_dict.values(),
    col_order=leaf_names,
)

fg.map(sns.distplot, "Out degree", norm_hist=True, kde=False)
fg.set(yticks=[], yticklabels=[])
stashfig("out-degree" + name_base)

fg = sns.FacetGrid(
    degree_df,
    col="Cluster",
    col_wrap=6,
    sharex=True,
    sharey=False,
    hue="Cluster",
    hue_order=pred_color_dict.keys(),
    palette=pred_color_dict.values(),
    col_order=leaf_names,
)

fg.map(sns.distplot, "Total degree", norm_hist=True, kde=False)
fg.set(yticks=[], yticklabels=[])
stashfig("total-degree" + name_base)
# %% [markdown]
# #


blockmodel_df = get_blockmodel_df(
    adj, pred_labels, use_weights=True, return_counts=False
)
plt.figure(figsize=(20, 20))
sns.heatmap(blockmodel_df, cmap="Reds")


g = nx.from_pandas_adjacency(blockmodel_df, create_using=nx.DiGraph())
uni_labels, counts = np.unique(pred_labels, return_counts=True)
size_scaler = 5
size_map = dict(zip(uni_labels, size_scaler * counts))
nx.set_node_attributes(g, size_map, name="Size")
mini_adj = nx.to_numpy_array(g, nodelist=uni_labels)
node_signal_flow = signal_flow(mini_adj)
sf_map = dict(zip(uni_labels, node_signal_flow))
nx.set_node_attributes(g, sf_map, name="Signal Flow")
sym_adj = symmetrize(mini_adj)
node_lap = LaplacianSpectralEmbed(n_components=1).fit_transform(sym_adj)
node_lap = np.squeeze(node_lap)
lap_map = dict(zip(uni_labels, node_lap))
nx.set_node_attributes(g, lap_map, name="Laplacian-2")
color_map = dict(zip(uni_labels, cc.glasbey_light))
nx.set_node_attributes(g, color_map, name="Color")
g.nodes(data=True)
nx.write_graphml(g, f"maggot_models/notebooks/outs/{FNAME}/mini_g.graphml")


# %% sort minigraph based on signal flow


sort_inds = np.argsort(node_signal_flow)[::-1]

temp_labels = blockmodel_df.index.values
temp_labels = temp_labels[sort_inds]

temp_adj = blockmodel_df.values
temp_adj = temp_adj[np.ix_(sort_inds, sort_inds)]
temp_df = pd.DataFrame(data=temp_adj, columns=temp_labels, index=temp_labels)
plt.figure(figsize=(20, 20))
sns.heatmap(temp_df, cmap="Reds")
stashfig("probplot")


def draw_networkx_nice(
    g,
    x_pos,
    y_pos,
    sizes=None,
    colors=None,
    nodelist=None,
    cmap="Blues",
    ax=None,
    x_boost=0,
    y_boost=0,
    draw_axes_arrows=False,
):
    if nodelist is None:
        nodelist = g.nodes()
    weights = nx.get_edge_attributes(g, "weight")

    x_attr_dict = nx.get_node_attributes(g, x_pos)
    y_attr_dict = nx.get_node_attributes(g, y_pos)

    pos = {}
    label_pos = {}
    for n in nodelist:
        pos[n] = (x_attr_dict[n], y_attr_dict[n])
        label_pos[n] = (x_attr_dict[n] + x_boost, y_attr_dict[n] + y_boost)

    if sizes is not None:
        size_attr_dict = nx.get_node_attributes(g, sizes)
        node_size = []
        for n in nodelist:
            node_size.append(size_attr_dict[n])

    if colors is not None:
        color_attr_dict = nx.get_node_attributes(g, colors)
        node_color = []
        for n in nodelist:
            node_color.append(color_attr_dict[n])

    weight_array = np.array(list(weights.values()))
    norm = mplc.Normalize(vmin=0, vmax=weight_array.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cmap = sm.to_rgba(weight_array)

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 30), frameon=False)

    node_collection = nx.draw_networkx_nodes(
        g, pos, node_color=node_color, node_size=node_size, with_labels=False, ax=ax
    )
    n_squared = len(nodelist) ** 2  # maximum z-order so far
    node_collection.set_zorder(n_squared)

    nx.draw_networkx_edges(
        g,
        pos,
        edge_color=cmap,
        connectionstyle="arc3,rad=0.2",
        arrows=True,
        width=1.5,
        ax=ax,
    )

    text_items = nx.draw_networkx_labels(g, label_pos, ax=ax)

    # make sure the labels are above all in z order
    for _, t in text_items.items():
        t.set_zorder(n_squared + 1)

    ax.set_xlabel(x_pos)
    ax.set_ylabel(y_pos)
    # plt.box(False)
    fig.set_facecolor("w")
    return ax


draw_networkx_nice(
    g, "Laplacian-2", "Signal Flow", nodelist=uni_labels, sizes="Size", colors="Color"
)
stashfig("nice-mini-graph")


# %%
