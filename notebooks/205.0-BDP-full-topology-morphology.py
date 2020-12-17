#%% [markdown]
# # Joint embedding of connectivity and morphology
# > Here I try to accomplish this on the maggot data using the MASE approach.
#
# - toc: true
# - badges: true
# - categories: [pedigo, maggot, graspologic]
# - hide: false
# - search_exclude: false
#%%
# collapse
import logging
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import QuantileTransformer
from umap import UMAP

from graspologic.embed import (
    AdjacencySpectralEmbed,
    LaplacianSpectralEmbed,
    select_dimension,
    selectSVD,
)
from graspologic.plot import pairplot
from graspologic.utils import pass_to_ranks, symmetrize
from src.data import load_metagraph
from src.io import savefig
from src.pymaid import start_instance
from src.visualization import CLASS_COLOR_DICT, adjplot, set_theme

# REF: https://stackoverflow.com/questions/35326814/change-level-logged-to-ipython-jupyter-notebook
logger = logging.getLogger()
assert len(logger.handlers) == 1
handler = logger.handlers[0]
handler.setLevel(logging.ERROR)

t0 = time.time()

# some plotting settings
set_theme()
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["text.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["xtick.color"] = "black"


CLASS_COLOR_DICT["sens"] = CLASS_COLOR_DICT["sens-ORN"]
CLASS_COLOR_DICT["bLN"] = CLASS_COLOR_DICT["pLN"]
palette = CLASS_COLOR_DICT


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


# for pymaid to pull neurons
start_instance()

#%% [markdown]
# ## The data
# Here I select a subset of the *Drosophila* larva connectome. Specifically:

# For the purposes of the network, I use the "summed" or union graph (not separating by
# axon/dendrite). I also restrict this analysis to one hemisphere (the left).
#%% load connectivity data
# collapse


data_dir = Path("maggot_models/experiments/nblast/outs")
side = "left"

# load graph
mg = load_metagraph("G")
meta = mg.meta
meta = meta[meta[side]]
mg = mg.reindex(meta.index, use_ids=True)
mg = mg.remove_pdiff()
mg = mg.make_lcc()
meta = mg.meta
print(f"{len(meta)} neurons in selected largest connected component on {side}")

# load nblast similarities
nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-scores.csv", index_col=0)
nblast_sim.columns = nblast_sim.columns.values.astype(int)
print(f"{len(nblast_sim)} neurons in NBLAST data on {side}")

# get neurons that are in both
intersect_index = np.intersect1d(meta.index, nblast_sim.index)
print(f"{len(intersect_index)} neurons in intersection on {side}")

# reindex appropriately
nblast_sim = nblast_sim.reindex(index=intersect_index, columns=intersect_index)
mg = mg.reindex(intersect_index, use_ids=True)
meta = mg.meta
adj = mg.adj
ptr_adj = pass_to_ranks(adj)
ptr_morph = symmetrize(nblast_sim.values)

#%% work on preprocessing the scores

transform = "quantile"
embedding = "LSE"
n_components = 6


distance = nblast_sim.values  # the raw nblast scores are dissimilarities/distances
sym_distance = symmetrize(distance)  # the raw scores are not symmetric
# make the distances between 0 and 1
sym_distance /= sym_distance.max()
sym_distance -= sym_distance.min()
# and then convert to similarity
morph_sim = 1 - sym_distance

indices = np.triu_indices_from(morph_sim, k=1)

if transform == "quantile":
    quant = QuantileTransformer()
    transformed_vals = quant.fit_transform(morph_sim[indices].reshape(-1, 1))
    transformed_vals = np.squeeze(transformed_vals)
    transformed_morph = np.ones_like(morph_sim)
    transformed_morph[indices] = transformed_vals
    transformed_morph[indices[::-1]] = transformed_vals
elif transform == "log":
    raise NotImplementedError()
else:
    transformed_morph = morph_sim

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]
sns.histplot(morph_sim[indices], ax=ax)
ax = axs[1]
sns.histplot(transformed_morph[indices], ax=ax)
plt.tight_layout()

Z = linkage(
    squareform(1 - transformed_morph), method="average"
)  # has to be distance here
colors = np.vectorize(palette.get)(meta["merge_class"])
sns.clustermap(
    transformed_morph,
    row_linkage=Z,
    col_linkage=Z,
    row_colors=colors,
    col_colors=colors,
    cmap="RdBu_r",
    center=0,
)

adjplot_kws = dict(
    meta=meta,
    sort_class=["simple_class"],
    item_order=["simple_class", "merge_class"],
    colors=["simple_class", "merge_class"],
    palette=palette,
    ticks=False,
)
adjplot(
    transformed_morph,
    title="Morphology (pairwise NBLAST)",
    cbar=False,
    **adjplot_kws,
)

embed_kws = dict(concat=False)


if embedding == "ASE":
    Embedder = AdjacencySpectralEmbed
elif embedding == "LSE":
    Embedder = LaplacianSpectralEmbed
    embed_kws["form"] = "R-DAD"

if embedding == "ASE":
    embed_kws["diag_aug"] = False


morph_model = Embedder(n_components=n_components, **embed_kws)
morph_embed = morph_model.fit_transform(transformed_morph)

pairplot(
    morph_embed,
    labels=meta["simple_class"].values,
    palette=palette,
    title="Morphology embedding",
    diag_kind=None,
)

umap = UMAP(n_neighbors=20, metric="precomputed", min_dist=0.75)
umap_embed = umap.fit_transform(1 - transformed_morph)


def simple_scatterplot(X, labels=None, palette="deep", ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plot_df = pd.DataFrame(data=X[:, :2], columns=["0", "1"])
    plot_df["labels"] = labels
    sns.scatterplot(
        data=plot_df, x="0", y="1", hue="labels", palette=palette, ax=ax, s=15
    )
    ax.set(xlabel="", ylabel="", title=title, xticks=[], yticks=[])
    ax.get_legend().remove()
    return fig, ax


simple_scatterplot(
    umap_embed,
    labels=meta["merge_class"].values,
    palette=palette,
    title=r"UMAP $\circ$ morphology dissimilarities",
)

#%% [markdown]
# ## Plotting the data for both modalities
# **Left**: the adjacency matrix for this subgraph after pass-to-ranks.
#
# **Right**: the similarity matrix obtained from NBLAST after some post-processing,
# including a quantile transform (like pass-to-ranks). Also likely subject to change.
#%%
# #%%

# inds = np.triu_indices_from(morph_sim, k=1)
# flat_morph_sim = morph_sim[inds]
# arg_zero = np.argwhere(morph_sim[inds] == 0).ravel()
# rows = inds[0][arg_zero]
# cols = inds[1][arg_zero]
# print(intersect_index[rows])
# print(intersect_index[cols])
# flat_morph_sim[arg_zero] = 1
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# ax = axs[0]
# sns.histplot(flat_morph_sim, ax=ax)
# ax = axs[1]
# sns.histplot(np.exp(-(flat_morph_sim ** 10)), ax=ax)
# plt.tight_layout()

#%%
sns.set_context("talk", font_scale=1.25)

adjplot_kws = dict(
    meta=meta,
    sort_class=["simple_class"],
    item_order=["simple_class", "merge_class"],
    colors=["simple_class", "merge_class"],
    palette=palette,
    ticks=False,
)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
adjplot(
    ptr_adj,
    plot_type="scattermap",
    ax=ax,
    title="Connectivity (adjacency matrix)",
    sizes=(1, 2),
    **adjplot_kws,
)

ax = axs[1]
adjplot(
    transformed_morph,
    ax=ax,
    title="Morphology (pairwise NBLAST)",
    cbar=False,
    **adjplot_kws,
)
stashfig("adj-morpho-heatmaps")


#%% [markdown]
# ## The goal and the method
# We aim to learn a single embedding which is representative of differences in both
# neuron connectivity and morphology. Ultimately, we wish to cluster (or otherwise look
# at which neurons are similar/different) using that combined representation.
#
# To do so we will use a MASE-like approach. It consists of two simple stages:
#
# 1. Embed the adjacency matrix (or Lapliacian) and embed the NBLAST similarity matrix
# 2. Concatenate the two embeddings from step 1, and then embed that matrix for our
# final representation.
#
#
# Since the connectivity embedding is directed, for the concatenation step in 2., I use
# $$W = \left [ U_A, V_A, U_M \right ]$$
# where $U_A$ and $V_A$ are the left and right singular vectors of the adjacency or
# Laplacian matrix, and $U_M$ is the singular vectors of the morphology similarity
# matrix, and $W$ is the matrix which is going to be embedded again for the final
# representation.
#%% first stage - embeddings
# collapse


def careys_rule(X):
    """Get the number of singular values to check"""
    return int(np.ceil(np.log2(np.min(X.shape))))


zg_n_components = careys_rule(ptr_adj)
max_n_components = 40  # basically just how many to show for screeplots

embed_kws = dict(concat=False)

embedding = "LSE"
if embedding == "ASE":
    Embedder = AdjacencySpectralEmbed
elif embedding == "LSE":
    Embedder = LaplacianSpectralEmbed
    embed_kws["form"] = "R-DAD"

adj_model = Embedder(n_components=max_n_components, **embed_kws)
adj_embed = adj_model.fit_transform(ptr_adj)

if embedding == "ASE":
    embed_kws["diag_aug"] = False

morph_model = Embedder(n_components=max_n_components, **embed_kws)
morph_embed = morph_model.fit_transform(transformed_morph)

#%% [markdown]
# ## Screeplots for the first stage
#%%
# collapse


def textplot(x, y, text, ax=None, x_pad=0, y_pad=0, **kwargs):
    """Plot a iterables of x, y, text with matplotlib's ax.text"""
    if ax is None:
        ax = plt.gca()
    for x_loc, y_loc, s in zip(x, y, text):
        ax.text(
            x_loc + x_pad,
            y_loc + y_pad,
            s,
            transform=ax.transData,
            **kwargs,
        )


def screeplot(
    singular_values, check_n_components=None, ax=None, title="Screeplot", n_elbows=4
):
    if ax is None:
        ax = plt.gca()

    elbows, elbow_vals = select_dimension(
        singular_values[:check_n_components], n_elbows=n_elbows
    )

    index = np.arange(1, len(singular_values) + 1)

    sns.lineplot(x=index, y=singular_values, ax=ax, zorder=1)
    sns.scatterplot(
        x=elbows,
        y=elbow_vals,
        color="darkred",
        marker="x",
        ax=ax,
        zorder=2,
        s=80,
        linewidth=2,
    )
    textplot(
        elbows,
        elbow_vals,
        elbows,
        ax=ax,
        color="darkred",
        fontsize="small",
        x_pad=0.5,
        y_pad=0,
        zorder=3,
    )
    ax.set(title=title, xlabel="Index", ylabel="Singular value")
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    return ax


sns.set_context(font_scale=1)
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
ax = axs[0]
screeplot(
    adj_model.singular_values_,
    check_n_components=zg_n_components,
    ax=ax,
    title="Connectivity",
)

ax = axs[1]
screeplot(
    morph_model.singular_values_,
    check_n_components=zg_n_components,
    ax=ax,
    title="Morphology",
)
plt.tight_layout()
stashfig("stage1-scree")
#%% [markdown]
# Based on the screeplots above, I selected 6 as the embedding dimension to use for
# stage 2.
#%% [markdown]
# ## Plot the connectivity and morphology embeddings from the first stage
# Here I show the first 6 dimensions of the embeddings (or the first 3 + 3 for a
# directed embedding).
#%%
# collapse
n_show = 6
n_show_per = n_show // 2
plot_adj_embed = np.concatenate(
    (adj_embed[0][:, :n_show_per], adj_embed[1][:, :n_show_per]), axis=1
)
col_names = [f"Out dimension {i + 1}" for i in range(n_show_per)]
col_names += [f"In dimension {i + 1}" for i in range(n_show_per)]
labels = meta["simple_class"].values
pairplot(
    plot_adj_embed,
    labels=labels,
    col_names=col_names,
    palette=palette,
    title="Connectivity screeplot",
    diag_kind=None,
)
stashfig("connectivity-pairplot")

pairplot(
    morph_embed[:, :n_show],
    labels=labels,
    palette=palette,
    title="Morphology screeplot",
    diag_kind=None,
)
stashfig("morphology-pairplot")
#%%
# collapse


def unscale(X):
    # TODO implement as a setting in graspologic
    norms = np.linalg.norm(X, axis=0)
    X = X / norms[None, :]
    return X


adj_embed = [unscale(a) for a in adj_embed]
unscale(morph_embed)

n_components = 6  # setting based on the plots above

concat_embed = np.concatenate(
    (
        adj_embed[0][:, :n_components],
        adj_embed[1][:, :n_components],
        morph_embed[:, :n_components],
    ),
    axis=1,
)

joint_embed, joint_singular_values, _ = selectSVD(
    concat_embed, n_components=concat_embed.shape[1], algorithm="full"
)
#%% [markdown]
# ## Screeplot for the second stage
#%%
# collapse
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
screeplot(
    joint_singular_values,
    check_n_components=careys_rule(concat_embed),
    ax=ax,
    title="Joint embedding screeplot",
)
stashfig("stage2-scree")
#%% [markdown]
# ## Plot the joint embedding from stage 2
#%%
# collapse
pairplot(
    joint_embed[:, :n_show],
    labels=labels,
    palette=palette,
    title="Joint embedding",
    diag_kind="hist",
)
stashfig("joint-pairplot")
#%%
# collapse
n_components = 6
umap = UMAP(min_dist=0.6, n_neighbors=30, metric="euclidean")
umap_embed = umap.fit_transform(joint_embed[:, :n_components])


def simple_scatterplot(X, labels=labels, palette="deep", ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plot_df = pd.DataFrame(data=X[:, :2], columns=["0", "1"])
    plot_df["labels"] = labels
    sns.scatterplot(data=plot_df, x="0", y="1", hue="labels", palette=palette, ax=ax)
    ax.set(xlabel="", ylabel="", title=title, xticks=[], yticks=[])
    ax.get_legend().remove()
    return fig, ax


simple_scatterplot(
    umap_embed,
    labels=meta["merge_class"].values,
    palette=palette,
    title=r"UMAP $\circ$ joint embedding",
)
stashfig("joint-umap")

#%%

from graspologic.cluster import DivisiveCluster

X = joint_embed[:, :n_components]
dc = DivisiveCluster(max_level=8)
hier_pred_labels = dc.fit_predict(X)

cluster_meta = mg.meta.copy()
for level in range(hier_pred_labels.shape[1]):
    uni_pred_labels, indicator = np.unique(
        hier_pred_labels[:, :level], axis=0, return_inverse=True
    )
    cluster_meta[f"lvl{level}_labels"] = indicator
#%%

for level_index in range(hier_pred_labels.shape[1]):
    out = pd.crosstab(
        index=cluster_meta[f"lvl{level_index}_labels"],
        columns=cluster_meta["merge_class"],
    )


def stacked_barplot(
    data: pd.Series,
    center=0,
    thickness=0.5,
    orient="v",
    ax=None,
    palette="deep",
    start=0,
    outline=False,
):
    curr_start = start
    if orient == "v":
        drawer = ax.bar
    elif orient == "h":
        drawer = ax.barh
    for item, count in data.iteritems():
        drawer(center, count, thickness, curr_start, color=palette[item], zorder=2)
        curr_start += count
    if outline:
        drawer(
            center,
            data.sum(),
            thickness,
            start,
            edgecolor="black",
            linewidth=1,
            color="none",
            zorder=1,
        )


pad = 20
n_levels = hier_pred_labels.shape[1]
sns.set_context("talk", 1)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection="polar")
ax.set(ylim=(n_levels - 0.5, -0.5), yticks=np.arange(n_levels), ylabel="Level")

for level in range(n_levels):
    counts_by_cluster = pd.crosstab(
        index=cluster_meta[f"lvl{level}_labels"],
        columns=cluster_meta["merge_class"],
    )
    start = 0
    for cluster_idx, row in counts_by_cluster.iterrows():
        stacked_barplot(
            row,
            center=level,
            palette=palette,
            ax=ax,
            orient="h",
            start=start,
            outline=True,
        )
        start += row.sum() + pad

ax.set_xlim((-pad, start + pad))
ax.set(xticks=[])
ax.spines["bottom"].set_visible(False)
stashfig("simple-bars")

#%%

walksort_loc = "maggot_models/experiments/walk_sort/outs/meta_w_order-gt=Gad-n_init=256-hops=16-loops=False-include_reverse=False.csv"

walksort_meta = pd.read_csv(walksort_loc, index_col=0)
cluster_meta["median_node_visits"] = walksort_meta["median_node_visits"]

mg.meta["median_node_visits"] = walksort_meta["median_node_visits"]
#%%
sns.set_context("talk", font_scale=1.25)

level = 6
adjplot_kws = dict(
    meta=cluster_meta,
    sort_class=[f"lvl{level}_labels"],
    class_order=["median_node_visits"],
    item_order=["simple_class", "merge_class"],
    colors=["merge_class"],
    palette=palette,
    ticks=False,
)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
adjplot(
    ptr_adj,
    plot_type="scattermap",
    ax=ax,
    title="Connectivity (adjacency matrix)",
    sizes=(1, 2),
    **adjplot_kws,
)

ax = axs[1]
adjplot(
    transformed_morph,
    ax=ax,
    title="Morphology (pairwise NBLAST)",
    cbar=False,
    **adjplot_kws,
)
stashfig("adj-morpho-heatmaps-clustered")

#%% need to do an actual comparison to the original (just connectivity) clustering

n_components = 6  # setting based on the plots above

alphas = [0, 0.25, 0.5, 0.75, 1]

for alpha in alphas:
    print(alpha)
    concat_embed = np.concatenate(
        (
            alpha * adj_embed[0][:, :n_components],
            alpha * adj_embed[1][:, :n_components],
            (1 - alpha) * morph_embed[:, :n_components],
        ),
        axis=1,
    )

    joint_embed, joint_singular_values, _ = selectSVD(
        concat_embed, n_components=concat_embed.shape[1], algorithm="full"
    )

    X = joint_embed[:, :n_components]
    dc = DivisiveCluster(max_level=8)
    hier_pred_labels = dc.fit_predict(X)

    cluster_meta = mg.meta.copy()
    n_levels = hier_pred_labels.shape[1]
    print(f"n_levels = {n_levels}")
    for level in range(n_levels):
        uni_pred_labels, indicator = np.unique(
            hier_pred_labels[:, :level], axis=0, return_inverse=True
        )
        cluster_meta[f"lvl{level}_labels"] = indicator

    level = min(6, n_levels - 1)
    print(f"level={level}")
    adjplot_kws = dict(
        meta=cluster_meta,
        sort_class=[f"lvl{level}_labels"],
        class_order=["median_node_visits"],
        item_order=["simple_class", "merge_class"],
        colors=["merge_class"],
        palette=palette,
        ticks=False,
    )

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    ax = axs[0]
    adjplot(
        ptr_adj,
        plot_type="scattermap",
        ax=ax,
        title="Connectivity (adjacency matrix)",
        sizes=(1, 2),
        **adjplot_kws,
    )

    ax = axs[1]
    adjplot(
        transformed_morph,
        ax=ax,
        title="Morphology (pairwise NBLAST)",
        cbar=False,
        **adjplot_kws,
    )
    stashfig(f"adj-morpho-heatmaps-clustered-alpha={alpha}")

#%% need to actually quantify the above
# TODO how much variance is there from run to run of DivisiveCluster? Must just be the 
# kmeans init, right? 

# TODO need to port over the code to compute model likelihood and evaluate. Should be 
# able to run on the opposite hemisphere connectivity as well.

# TODO need to come up with a metric for evaluating how good the morphological
# clustering is? maybe just something like modularity or discriminability on the morph
# distances?

# TODO