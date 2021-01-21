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
from scipy.stats.mstats import gmean
from sklearn.preprocessing import QuantileTransformer
from umap import UMAP
import navis
import pymaid
from giskard.plot import (
    dissimilarity_clustermap,
    screeplot,
    simple_scatterplot,
    simple_umap_scatterplot,
    stacked_barplot,
)
from giskard.stats import calc_discriminability_statistic
from giskard.utils import careys_rule
from graspologic.cluster import DivisiveCluster
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, selectSVD
from graspologic.plot import pairplot
from graspologic.utils import pass_to_ranks, remap_labels, symmetrize
from src.data import load_metagraph
from src.embed import JointEmbed, unscale
from src.io import savefig
from src.metrics import calc_model_liks, plot_pairedness
from src.pymaid import start_instance
from src.visualization import CLASS_COLOR_DICT, adjplot, set_theme, simple_plot_neurons

print(f"pymaid version: {pymaid.__version__}")

# REF: https://stackoverflow.com/questions/35326814/change-level-logged-to-ipython-jupyter-notebook
logger = logging.getLogger()
assert len(logger.handlers) == 1
handler = logger.handlers[0]
handler.setLevel(logging.ERROR)

t0 = time.time()

# for pymaid to pull neurons
start_instance()

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


#%%
# collapse
walksort_loc = "maggot_models/experiments/walk_sort/outs/meta_w_order-gt=Gad-n_init=256-hops=16-loops=False-include_reverse=False.csv"
walksort_meta = pd.read_csv(walksort_loc, index_col=0)

meta = load_metagraph("G").meta
skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)


def load_side(side, nblast_type="similarities"):
    # load graph, remove partially diff, get LCC on that side
    mg = load_metagraph("G")
    mg.meta["median_node_visits"] = walksort_meta["median_node_visits"]
    meta = mg.meta
    meta = meta[meta[side]]
    mg = mg.reindex(meta.index, use_ids=True)
    mg = mg.remove_pdiff()
    mg = mg.make_lcc()
    meta = mg.meta
    print(f"{len(meta)} neurons in selected largest connected component on {side}")

    # load nblast scores/similarities
    nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-{nblast_type}.csv", index_col=0)
    nblast_sim.columns = nblast_sim.columns.values.astype(int)
    print(f"{len(nblast_sim)} neurons in NBLAST data on {side}")

    # get neurons that are in both
    intersect_index = np.intersect1d(meta.index, nblast_sim.index)
    print(f"{len(intersect_index)} neurons in intersection on {side}")

    # reindex appropriately
    nblast_sim = nblast_sim.reindex(index=intersect_index, columns=intersect_index)
    mg = mg.reindex(intersect_index, use_ids=True)
    return mg, nblast_sim


def apply_flat_labels(meta, hier_pred_labels):
    cluster_meta = meta.copy()
    n_levels = hier_pred_labels.shape[1]
    for level in range(n_levels):
        uni_pred_labels, indicator = np.unique(
            hier_pred_labels[:, :level], axis=0, return_inverse=True
        )
        cluster_meta[f"lvl{level}_labels"] = indicator
    return cluster_meta


def get_paired_subset(left, right):
    left = left.copy()
    right = right.copy()
    intersect_pairs = np.intersect1d(left["pair_id"], right["pair_id"])
    intersect_pairs = intersect_pairs[intersect_pairs != -1]
    left["inds"] = range(len(left))
    right["inds"] = range(len(right))
    left = left[left["pair_id"].isin(intersect_pairs)].sort_values("pair_id")
    right = right[right["pair_id"].isin(intersect_pairs)].sort_values("pair_id")
    assert (left["pair_id"].values == right["pair_id"].values).all()
    return left, right


#%% [markdown]
# ## The data
# Here use the *Drosophila* larva connectome.

# For the purposes of the network, I use the "summed" or union graph (not separating by
# axon/dendrite). I also restrict this analysis to one hemisphere (the left).
#%% load connectivity data
# collapse
data_dir = Path("maggot_models/experiments/nblast/outs")
side = "left"

mg, nblast_sim = load_side("left", nblast_type="scores")
meta = mg.meta
adj = mg.adj
ptr_adj = pass_to_ranks(adj)

#%% [markdown]
# ## Preprocessing NBLAST scores
#%% load and preprocessing the scores
# collapse


def preprocess_nblast(
    nblast_scores, symmetrize_mode="geom", transform="ptr", return_untransformed=False
):
    distance = nblast_scores  # the raw nblast scores are dissimilarities/distances
    indices = np.triu_indices_from(distance, k=1)

    if symmetrize_mode == "geom":
        fwd_dist = distance[indices]
        back_dist = distance[indices[::-1]]
        stack_dist = np.concatenate(
            (fwd_dist.reshape(-1, 1), back_dist.reshape(-1, 1)), axis=1
        )
        geom_mean = gmean(stack_dist, axis=1)
        sym_distance = np.zeros_like(distance)
        sym_distance[indices] = geom_mean
        sym_distance[indices[::-1]] = geom_mean
    else:  # simple average
        sym_distance = symmetrize(distance)

    # make the distances between 0 and 1
    sym_distance /= sym_distance.max()
    sym_distance -= sym_distance.min()
    # and then convert to similarity
    morph_sim = 1 - sym_distance

    if transform == "quantile":
        quant = QuantileTransformer(n_quantiles=2000)
        transformed_vals = quant.fit_transform(morph_sim[indices].reshape(-1, 1))
        transformed_vals = np.squeeze(transformed_vals)
        transformed_morph = np.ones_like(morph_sim)
        transformed_morph[indices] = transformed_vals
        transformed_morph[indices[::-1]] = transformed_vals
    elif transform == "ptr":
        transformed_morph = pass_to_ranks(morph_sim)
        np.fill_diagonal(
            transformed_morph, 1
        )  # should be exactly 1, isnt cause of ties
    elif transform == "log":
        raise NotImplementedError()
    else:
        transformed_morph = morph_sim
    if return_untransformed:
        return transformed_morph, morph_sim
    else:
        return transformed_morph


transform = "ptr"
symmetrize_mode = "geom"

transformed_morph, morph_sim = preprocess_nblast(
    nblast_sim.values,
    symmetrize_mode=symmetrize_mode,
    transform=transform,
    return_untransformed=True,
)

indices = np.triu_indices_from(morph_sim, k=1)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
ax = axs[0]
sns.histplot(morph_sim[indices], ax=ax, stat="density")
ax.set(title="Before preprocessing")
ax = axs[1]
sns.histplot(transformed_morph[indices], ax=ax, stat="density")
ax.set(title="After preprocessing")
fig.text(0.44, 0, "NBLAST similarity")
plt.tight_layout()
stashfig("nblast-score-preprocessing")

#%% [markdown]
# ## Examples of pairwise NBLAST similarities
#%%
# collapse

argsort_sim_inds = np.argsort(transformed_morph[indices])


def get_neurons_by_quantile(meta, argsort_sim_inds, quantile, boost=0):
    choice_ind = int(np.floor(len(argsort_sim_inds) * quantile)) + boost
    choice_ind = argsort_sim_inds[choice_ind]
    row_neuron_index = indices[0][choice_ind]
    col_neuron_index = indices[1][choice_ind]
    row_neuron_id = int(meta.index[row_neuron_index])
    col_neuron_id = int(meta.index[col_neuron_index])
    return row_neuron_id, col_neuron_id


quantiles = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995])
boost = 3
n_cols = 5
n_rows = int(np.ceil(len(quantiles) / n_cols))
fig = plt.figure(figsize=(10, 10 / 5 * n_rows + 1))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.1)

for i, q in enumerate(quantiles):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    neurons = get_neurons_by_quantile(meta, argsort_sim_inds, q, boost=boost)
    ax = simple_plot_neurons(neurons, ax=ax, palette=skeleton_color_dict)
    if i == 0:
        ax.set_title(f"Quantile: {q}", fontsize="small")
    else:
        ax.set_title(f"{q}", fontsize="small")
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))
stashfig(f"nblast-difference-quantiles-boost={boost}")

#%% [markdown]
# ## All pairwise NBLAST similarities after preprocessing
#%%
# collapse
dissimilarity_clustermap(
    transformed_morph,
    colors=meta["merge_class"].values,
    invert=True,
    palette=palette,
    method="ward",
)
stashfig("nblast-dissim-clustermap")

#%%
# from scipy.cluster.hierarchy import linkage, fcluster
# from scipy.spatial.distance import squareform
# import colorcet as cc

# dissimilarity = transformed_morph
# invert = True
# colors = meta["merge_class"].values
# method = "ward"
# center = 0
# cmap = "RdBu_r"
# criterion = "distance"
# t = 4.5
# if invert:
#     cluster_dissimilarity = 1 - dissimilarity
# else:
#     cluster_dissimilarity = dissimilarity
# # since it assumes a distance/dissimilarity is input, the metric kwarg doesnt matter
# Z = linkage(squareform(cluster_dissimilarity), method=method)

# if palette is not None and colors is not None:
#     colors = np.array(np.vectorize(palette.get)(colors))

# # clustergrid = sns.clustermap(
# #     dissimilarity,
# #     row_linkage=Z,
# #     col_linkage=Z,
# #     cmap=cmap,
# #     center=center,
# #     row_colors=colors,
# #     col_colors=colors,
# #     xticklabels=False,
# #     yticklabels=False,
# # )
# # inds = clustergrid.dendrogram_col.reordered_ind


# flat_labels = fcluster(Z, t, criterion=criterion)

# simple_colors = {0: "black", 1: "lightgrey"}
# bw_colors = np.vectorize(lambda x: simple_colors[x % 2])(flat_labels)

# cc_palette = dict(zip(np.unique(flat_labels), cc.glasbey_light))
# cluster_colors = np.array(np.vectorize(cc_palette.get)(flat_labels))

# clustergrid = sns.clustermap(
#     dissimilarity,
#     row_linkage=Z,
#     col_linkage=Z,
#     cmap=cmap,
#     center=center,
#     row_colors=[bw_colors, colors],
#     col_colors=[bw_colors, colors],
#     xticklabels=False,
#     yticklabels=False,
# )
# ax = clustergrid.ax_col_dendrogram
# ax.axhline(t, linewidth=1.5, linestyle=":", color="dodgerblue")

# ax = clustergrid.ax_row_dendrogram
# ax.axvline(t, linewidth=1.5, linestyle=":", color="dodgerblue")

dissimilarity_clustermap(
    transformed_morph,
    colors=meta["merge_class"].values,
    invert=True,
    palette=palette,
    method="ward",
    cut=True,
    t=0.5,
    fcluster_palette="glasbey_light",
    cmap="RdBu_r",
)

#%% [markdown]
# ## Pairwise NBLAST similarities reduced to two dimensions
#%%
# collapse
simple_umap_scatterplot(
    1 - transformed_morph,
    labels=meta["merge_class"].values,
    metric="precomputed",
    palette=CLASS_COLOR_DICT,
    title="NBLAST similarities",
)
stashfig("umap-on-nblast")

#%% [markdown]
# ## Plotting the data for both modalities
# **Left**: the adjacency matrix for this subgraph after pass-to-ranks.
#
# **Right**: the similarity matrix obtained from NBLAST after some post-processing,
# including a quantile transform (like pass-to-ranks). Also likely subject to change.

#%%
# collapse
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
#%% first stage - running the embeddings
# collapse
embedding = "ASE"

zg_n_components = careys_rule(ptr_adj)
max_n_components = 40  # basically just how many to show for screeplots

embed_kws = dict(concat=False)

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
    title="Connectivity",
)
stashfig("connectivity-pairplot")
pairplot(
    morph_embed[:, :n_show],
    labels=labels,
    palette=palette,
    title="Morphology",
)
stashfig("morphology-pairplot")
#%%
# collapse
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
simple_umap_scatterplot(
    joint_embed[:, :n_components],
    labels=meta["merge_class"].values,
    palette=palette,
    title=r"joint embedding",
)
stashfig("joint-umap")

#%% run hierarchical GMM
X = joint_embed[:, :n_components]
dc = DivisiveCluster(max_level=8)
hier_pred_labels = dc.fit_predict(X)
cluster_meta = apply_flat_labels(mg.meta, hier_pred_labels)

#%% assess hierarchical GMM
for level_index in range(hier_pred_labels.shape[1]):
    out = pd.crosstab(
        index=cluster_meta[f"lvl{level_index}_labels"],
        columns=cluster_meta["merge_class"],
    )

pad = 20
n_levels = hier_pred_labels.shape[1]
sns.set_context("talk", 1)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
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


#%% condensing all of the code and running it for multiple graphs, basically.


sides = ["left", "right"]
results = {}
for side in sides:
    currtime = time.time()

    # # load and process data
    mg, nblast_sim = load_side(side, nblast_type="similarities")
    adj = pass_to_ranks(mg.adj)
    sim = nblast_sim.values

    # # joint embedding
    je = JointEmbed(n_components=6, stage1_n_components=6, method="ase")
    print(je.method)
    joint_embedding = je.fit_transform(adj, sim, weights=[1, 1])

    # clustering, applying labels to metadata
    dc = DivisiveCluster(max_level=8)
    hier_pred_labels = dc.fit_predict(joint_embedding)
    cluster_meta = apply_flat_labels(mg.meta, hier_pred_labels)

    print(
        f"{time.time() - currtime:.3f} elapsed for embedding and clustering on {side}."
    )
    print()

    # collect
    result = {
        "side": side,
        "adj": adj,
        "sim": sim,
        "meta": cluster_meta,
        "divisive_cluster": dc,
        "joint_embedding": joint_embedding,
    }
    results[(side,)] = result

#%% get out the paired stuff
n_levels_show = 7

left_meta = results[("left",)]["meta"]
left_adj = results[("left",)]["adj"]

right_meta = results[("right",)]["meta"]
right_adj = results[("right",)]["adj"]

paired_left, paired_right = get_paired_subset(left_meta, right_meta)


#%% align the labels
for level in range(n_levels_show + 1):
    left_labels = paired_left[f"lvl{level}_labels"]
    right_labels = paired_right[f"lvl{level}_labels"]
    new_right_labels = remap_labels(left_labels, right_labels)
    paired_right[f"lvl{level}_labels"] = new_right_labels
    print(f"Level {level} pairedness:")
    print((left_labels == new_right_labels).sum() / len(left_labels))
    print()


#%% combine into one graph
old_lp_inds = paired_left["inds"]
paired_left_adj = left_adj[np.ix_(old_lp_inds, old_lp_inds)]
old_rp_inds = paired_right["inds"]
paired_right_adj = right_adj[np.ix_(old_rp_inds, old_rp_inds)]
n_pairs = len(paired_left)
full_meta = pd.concat((paired_left, paired_right))
full_adj = np.zeros((2 * n_pairs, 2 * n_pairs))
full_adj[:n_pairs, :n_pairs] = paired_left_adj
full_adj[n_pairs:, n_pairs:] = paired_right_adj
lp_inds = np.arange(n_pairs)
rp_inds = np.arange(n_pairs) + n_pairs

#%% calculate and plot likelihoods
lik_results = calc_model_liks(
    full_adj, full_meta, lp_inds, rp_inds, n_levels=n_levels_show
)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
model_name = "DCSBM"
plot_df = lik_results[lik_results["model"] == model_name]
sns.lineplot(
    data=plot_df,
    hue="test",
    x="level",
    y="norm_score",
    style="train_side",
    markers=True,
)
ax.set_ylabel(f"{model_name} normalized log lik.")
ax.set_yticks([])
ax.set_xlabel("Level")
stashfig("likelihoods")

#%% calculate and plot pairedness
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_pairedness(full_meta, lp_inds, rp_inds, ax, n_levels=n_levels_show)
stashfig("pairedness")

#%%
side = "left"
# # load and process data
mg, nblast_sim = load_side(side, nblast_type="scores")
adj = pass_to_ranks(mg.adj)
sim = preprocess_nblast(
    nblast_sim.values, symmetrize_mode=symmetrize_mode, transform=transform
)

# # joint embedding
je = JointEmbed(n_components=6, stage1_n_components=6, method="ase")
joint_embedding = je.fit_transform(adj, sim, weights=[1, 1])

# clustering, applying labels to metadata
dc = DivisiveCluster(max_level=8)
hier_pred_labels = dc.fit_predict(joint_embedding)
cluster_meta = apply_flat_labels(mg.meta, hier_pred_labels)

#%%
cluster_discrim_by_level = {}
for level in range(1, n_levels_show + 1):
    labels = cluster_meta[f"lvl{level}_labels"].values.copy()
    discrim, discrim_by_cluster = calc_discriminability_statistic(1 - sim, labels)
    cluster_discrim_by_level[level] = discrim_by_cluster
    print(f"Level {level} discriminability: {discrim}")
    np.random.shuffle(labels)
    discrim, discrim_by_cluster = calc_discriminability_statistic(1 - sim, labels)
    print(f"Level {level} shuffled discriminability: {discrim}")
    print()

#%%
rows = []
for level in range(1, n_levels_show + 1):
    labels = cluster_meta[f"lvl{level}_labels"].values.copy()
    uni_labels = np.unique(labels)
    for ul in uni_labels:
        neuron_ids = cluster_meta[cluster_meta[f"lvl{level}_labels"] == ul].index.values
        cluster_discrim = cluster_discrim_by_level[level][ul]
        row = {
            "level": level,
            "label": ul,
            "neuron_ids": neuron_ids,
            "cluster_discrim": cluster_discrim,
        }
        rows.append(row)
results = pd.DataFrame(rows)
results

#%%
level = 7
level_results = results[results["level"] == level].copy()
level_results.sort_values("cluster_discrim", inplace=True, ascending=False)

show_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
show_inds += list(len(level_results) - np.array(show_inds)[::-1] - 1)
show_inds = np.arange(len(level_results))
n_show_neurons = 1000

sns.set_context("talk", font_scale=0.75)
n_cols = 6
n_rows = int(np.ceil(len(show_inds) / n_cols))
fig = plt.figure(figsize=(12, 12 / n_cols * n_rows))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0.03, hspace=0.1)
axs = np.empty((n_rows, n_cols), dtype=object)
for i, result_idx in enumerate(show_inds):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    neurons = level_results.iloc[result_idx]["neuron_ids"][:n_show_neurons]
    simple_plot_neurons(neurons, ax=ax, palette=skeleton_color_dict)
    cluster_discrim = level_results.iloc[result_idx]["cluster_discrim"]
    if i == 0:
        ax.set_title(
            f"Discriminability:\n{cluster_discrim:0.3f} ({len(neurons)})",
        )
    else:
        ax.set_title(f"{cluster_discrim:0.3f} ({len(neurons)})")
    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))
axs[0, 0].set_ylabel("Bottom 20 clusters")
axs[0, 2].set_ylabel("Top 20 clusters")
stashfig(f"discriminability-cluster-morpho-level={level}")


# ## Points to make
# - geometric mean for symmetrizing distance is the way to go
# - discriminability can be used component-wise to sort clusters in terms of morphology
#   similarity scores

#%% need to actually quantify the above
# TODO how much variance is there from run to run of DivisiveCluster? Must just be the
# kmeans init, right?

# TODO need to port over the code to compute model likelihood and evaluate. Should be
# able to run on the opposite hemisphere connectivity as well.

# TODO need to come up with a metric for evaluating how good the morphological
# clustering is? maybe just something like modularity or discriminability on the morph
# distances?
