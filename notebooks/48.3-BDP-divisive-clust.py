# %% [markdown]
# # Imports
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

from src.cluster import DivisiveCluster
from src.data import load_everything
from src.embed import lse, preprocess_graph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_sbm_prob
from src.visualization import bartreeplot, get_color_dict, get_colors, sankey, screeplot

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2019-12-18"

SAVEFIGS = True
SAVESKELS = True
SAVEOBJS = True

PTR = True
if PTR:
    ptr_type = "PTR"
else:
    ptr_type = "Raw"

ONLY_RIGHT = False
if ONLY_RIGHT:
    brain_type = "Right Hemisphere"
    brain_type_short = "righthemi"
else:
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
# # Load the data

adj, class_labels, side_labels, skeleton_labels = load_everything(
    "Gad",
    version=BRAIN_VERSION,
    return_keys=["Merge Class", "Hemisphere"],
    return_ids=True,
)


# select the right hemisphere
if ONLY_RIGHT:
    right_inds = np.where(side_labels == "R")[0]
    adj = adj[np.ix_(right_inds, right_inds)]
    class_labels = class_labels[right_inds]
    skeleton_labels = skeleton_labels[right_inds]

adj, class_labels, skeleton_labels = preprocess_graph(
    adj, class_labels, skeleton_labels
)
known_inds = np.where(class_labels != "Unk")[0]

# %% [markdown]
# # Embedding
n_verts = adj.shape[0]
latent, laplacian = lse(adj, N_COMPONENTS, regularizer=None, ptr=PTR)
latent_dim = latent.shape[1] // 2
screeplot(
    laplacian, title=f"Laplacian scree plot, R-DAD (ZG2 = {latent_dim} + {latent_dim})"
)
print(f"ZG chose dimension {latent_dim} + {latent_dim}")
# %% [markdown]
# # Fitting divisive cluster model with GraspyGMM
name_base = f"-{cluster_type}-{embed_type}-{ptr_type}-{brain_type_short}-{GRAPH_TYPE}"

base = f"maggot_models/notebooks/outs/{FNAME}/objs/"
filename = base + "dc" + name_base + ".pickle"
if os.path.isfile(filename):
    print("Attempting to load file")
    with open(filename, "rb") as f:
        dc = pickle.load(f)
    print(f"Loaded file from {filename}")
else:
    print("Fitting DivisiveCluster model")
    start = timer()
    dc = DivisiveCluster(n_init=N_INIT, cluster_method=CLUSTER_METHOD)
    dc.fit(latent)
    end = end = timer()
    print()
    print(f"DivisiveCluster took {(end - start)/60.0} minutes to fit")
    print()
dc.print_tree(print_val="bic_ratio")
pred_labels = dc.predict(latent)

# %% [markdown]
# # Plotting and saving divisive cluster hierarchy results for GraspyGMM

stashobj(dc, "dc" + name_base)

n_classes = len(np.unique(class_labels))
class_color_dict = get_color_dict(class_labels, pal=cc.glasbey_cool)
pred_color_dict = get_color_dict(pred_labels, pal=cc.glasbey_warm)
all_color_dict = {**class_color_dict, **pred_color_dict}
stashobj(all_color_dict, "all_color_dict" + name_base)

class_colors = np.array(itemgetter(*class_labels)(class_color_dict))

stashskel("known-skels" + name_base, skeleton_labels, class_labels, class_colors)
stashskel(
    "known-skels" + name_base,
    skeleton_labels,
    class_labels,
    class_colors,
    multiout=True,
)


title = (
    f"Divisive hierarchical clustering,"
    + f" {cluster_type}, {embed_type} ({latent_dim} + {latent_dim}), {ptr_type},"
    + f" {brain_type}, {graph_type}"
)

fig, ax = plt.subplots(1, 1, figsize=(20, 30))
sankey(ax, class_labels, pred_labels, aspect=20, fontsize=16, colorDict=all_color_dict)
ax.axis("off")
ax.set_title(title, fontsize=30)
stashfig("sankey" + name_base)

fig, ax = plt.subplots(1, 1, figsize=(20, 30))
sankey(ax, pred_labels, class_labels, aspect=20, fontsize=16, colorDict=all_color_dict)
ax.axis("off")
ax.set_title(title, fontsize=30)
stashfig("sankey-inv" + name_base)

sns.set_context("talk", font_scale=0.8)

_, _, leaf_names = bartreeplot(
    dc,
    class_labels,
    pred_labels,
    show_props=True,
    print_props=False,
    inverse_memberships=False,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-props" + name_base)
bartreeplot(
    dc,
    class_labels,
    pred_labels,
    show_props=False,
    print_props=True,
    inverse_memberships=False,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-counts" + name_base)
bartreeplot(
    dc,
    class_labels,
    pred_labels,
    show_props=True,
    inverse_memberships=True,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-props-inv" + name_base)
bartreeplot(
    dc,
    class_labels,
    pred_labels,
    show_props=False,
    inverse_memberships=True,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-counts-inv" + name_base)


pred_colors = np.array(itemgetter(*pred_labels)(pred_color_dict))
stashskel("pred-skels" + name_base, skeleton_labels, pred_labels, pred_colors)
stashskel(
    "pred-skels" + name_base, skeleton_labels, pred_labels, pred_colors, multiout=True
)


# %% [markdown]
# # Plot the degree distribution for each predicted cluster


in_edgesums = adj.sum(axis=0)
out_edgesums = adj.sum(axis=1)
in_degree = np.count_nonzero(adj, axis=0)
out_degree = np.count_nonzero(adj, axis=1)
degree_df = pd.DataFrame()
degree_df["In edgesum"] = in_edgesums
degree_df["Out edgesum"] = out_edgesums
degree_df["In degree"] = in_degree
degree_df["Out degree"] = out_degree
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


# %% [markdown]
# #


def _calculate_block_edgesum(graph, block_inds, block_vert_inds):
    """
    graph : input n x n graph 
    block_inds : list of length n_communities
    block_vert_inds : list of list, for each block index, gives every node in that block
    return_counts : whether to calculate counts rather than proportions
    """

    n_blocks = len(block_inds)
    block_pairs = cartprod(block_inds, block_inds)
    block_p = np.zeros((n_blocks, n_blocks))

    for p in block_pairs:
        from_block = p[0]
        to_block = p[1]
        from_inds = block_vert_inds[from_block]
        to_inds = block_vert_inds[to_block]
        block = graph[from_inds, :][:, to_inds]
        p = np.sum(block)
        p = p / block.size
        block_p[from_block, to_block] = p

    return block_p


def get_block_edgesums(adj, pred_labels, sort_blocks):
    block_vert_inds, block_inds, block_inv = _get_block_indices(pred_labels)
    block_sums = _calculate_block_edgesum(adj, block_inds, block_vert_inds)
    block_sums = block_sums[np.ix_(sort_blocks, sort_blocks)]
    block_sum_df = pd.DataFrame(data=block_sums, columns=sort_blocks, index=sort_blocks)
    return block_sum_df


def probplot(
    prob_df,
    ax=None,
    title=None,
    log_scale=False,
    cmap="Purples",
    vmin=None,
    vmax=None,
    figsize=(10, 10),
):
    cbar_kws = {"fraction": 0.08, "shrink": 0.8, "pad": 0.03}

    data = prob_df.values

    if log_scale:
        data = data + 0.001

        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        cbar_ticks = [
            math.pow(10, i)
            for i in range(
                math.floor(math.log10(data.min().min())),
                1 + math.ceil(math.log10(data.max().max())),
            )
        ]
        cbar_kws["ticks"] = cbar_ticks

    if ax is None:
        _ = plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.set_title(title, pad=30, fontsize=30)

    sns.set_context("talk", font_scale=1)

    heatmap_kws = dict(
        cbar_kws=cbar_kws,
        annot=True,
        square=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fmt=".0f",
    )
    if log_scale:
        heatmap_kws["norm"] = log_norm
    if ax is not None:
        heatmap_kws["ax"] = ax

    ax = sns.heatmap(prob_df, **heatmap_kws)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return ax


prob_df = get_sbm_prob(adj, pred_labels)
# block_sum_df = get_block_edgesums(adj, pred_labels, prob_df.columns.values)
# %% [markdown]
# #
fig, ax = plt.subplots(figsize=(30, 30))
probplot(100 * prob_df, ax=ax, title="Connection percentage")
stashfig("conn-prob" + name_base)
# plt.figure(figsize=(10, 10))
# probplot(block_sum_df, title="Average synapses")


# %% [markdown]
# # Plot SBM minigraph with signal flow vs lap. Try to split out sensory inputs here
sbm_prob = get_sbm_prob(adj, pred_labels)

from src.hierarchy import signal_flow
import networkx as nx
from graspy.embed import LaplacianSpectralEmbed
from graspy.utils import symmetrize
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from graspy.utils import cartprod


def _get_block_indices(y):
    """
    y is a length n_verts vector of labels

    returns a length n_verts vector in the same order as the input
    indicates which block each node is
    """
    block_labels, block_inv, block_sizes = np.unique(
        y, return_inverse=True, return_counts=True
    )

    n_blocks = len(block_labels)
    block_inds = range(n_blocks)

    block_vert_inds = []
    for i in block_inds:
        # get the inds from the original graph
        inds = np.where(block_inv == i)[0]
        block_vert_inds.append(inds)
    return block_labels, block_vert_inds, block_inds


def _calculate_block_counts(graph, block_inds, block_vert_inds, return_counts=True):
    """
    graph : input n x n graph 
    block_inds : list of length n_communities
    block_vert_inds : list of list, for each block index, gives every node in that block
    return_counts : whether to calculate counts rather than proportions
    """

    n_blocks = len(block_inds)
    block_pairs = cartprod(block_inds, block_inds)
    block_p = np.zeros((n_blocks, n_blocks))

    for p in block_pairs:
        from_block = p[0]
        to_block = p[1]
        from_inds = block_vert_inds[from_block]
        to_inds = block_vert_inds[to_block]
        block = graph[from_inds, :][:, to_inds]
        if return_counts:
            p = np.sum(block)
        block_p[from_block, to_block] = p
    return block_p


block_labels, block_vert_inds, block_inds = _get_block_indices(pred_labels)
block_counts = _calculate_block_counts(adj, block_inds, block_vert_inds)
block_count_df = pd.DataFrame(
    index=block_labels, columns=block_labels, data=block_counts
)
#%%
# uni_pred_labels, counts = np.unique(pred_labels, return_counts=True)
# uni_ints = range(len(uni_pred_labels))
# label_map = dict(zip(uni_pred_labels, uni_ints))
# int_labels = np.array(itemgetter(*uni_pred_labels)(label_map))
# synapse_counts = _calculate_block_counts(adj, uni_ints, pred_labels)

block_df = sbm_prob
block_adj = sbm_prob.values
block_labels = sbm_prob.index.values
sym_adj = symmetrize(block_adj)
lse_embed = LaplacianSpectralEmbed(form="DAD", n_components=1)
latent = lse_embed.fit_transform(sym_adj)
latent = np.squeeze(latent)

block_signal_flow = signal_flow(block_adj)
block_g = nx.from_pandas_adjacency(block_df, create_using=nx.DiGraph())
pos = dict(zip(block_labels, zip(latent, block_signal_flow)))
weights = nx.get_edge_attributes(block_g, "weight")

node_colors = np.array(itemgetter(*block_labels)(pred_color_dict))

uni_pred_labels, pred_counts = np.unique(pred_labels, return_counts=True)
size_map = dict(zip(uni_pred_labels, pred_counts))
node_sizes = np.array(itemgetter(*block_labels)(size_map))
node_sizes *= 4
norm = mpl.colors.Normalize(vmin=0.1, vmax=block_adj.max())
sm = ScalarMappable(cmap="Blues", norm=norm)
cmap = sm.to_rgba(np.array(list(weights.values())))
# cmap = mpl.colors.LinearSegmentedColormap("Blues", block_counts.ravel()).to_rgba(
#     np.array(list(labels.values()))
# )
fig, ax = plt.subplots(figsize=(30, 30))
node_collection = nx.draw_networkx_nodes(
    block_g,
    pos,
    edge_color=cmap,
    node_color=node_colors,
    node_size=node_sizes,
    with_labels=False,
)
n_squared = len(node_sizes) ** 2
node_collection.set_zorder(n_squared)

arrow_collection = nx.draw_networkx_edges(
    block_g,
    pos,
    edge_color=cmap,
    connectionstyle="arc3,rad=0.2",
    arrows=True,
    width=1.5,
)

boost = 0.005
for (key, val), size in zip(pos.items(), node_sizes):
    new_val = (val[0] + boost, val[1])
    pos[key] = new_val

text_items = nx.draw_networkx_labels(
    block_g,
    pos,
    edge_color=cmap,
    node_color=node_colors,
    node_size=node_sizes,
    width=1.5,
)
for _, t in text_items.items():
    t.set_zorder(n_squared + 1)

ax.axis("off")
stashfig("synapse-count-drawing" + name_base)


# %% [markdown]
# # Compute ARI of left vs right for the paired neurons

# %% [markdown]
# # Try to flag the places where there is disagreement left/right
