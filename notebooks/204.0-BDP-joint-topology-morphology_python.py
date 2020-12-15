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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymaid
import seaborn as sns
from hyppo.discrim import DiscrimOneSample
from sklearn.metrics import pairwise_distances
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
from navis import NeuronList, TreeNeuron, nblast_allbyall
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
# - Kenyon cells (KC)
# - Mushroom body input neurons (MBIN)
# - Mushroom body output neurons (MBON)
# - Uniglomerular projection neurons (uPN)
# - Thermosensitive projection neurons (tPN)
# - Visual projection neurons (vPN)
# - Broad and picky local neurons (bLN & pLN)
# - APL
# - Larval optic neuropil (LON)
# - TODO: visual sensory neurons
#
#
# For the purposes of the network, I use the "summed" or union graph (not separating by
# axon/dendrite). I also restrict this analysis to one hemisphere (the left).
#%% load connectivity data
# collapse
mg = load_metagraph("G")
meta = mg.meta
meta = meta[meta["left"]]
class1 = ["KC", "MBIN", "MBON", "uPN", "tPN", "vPN", "bLN", "pLN", "APL", "LON"]
class2 = ["ORN", "Rh5", "Rh6"]  # TODO fix and make the visual ones work?
meta = meta[meta["class1"].isin(class1) | meta["class2"].isin(class2)]
meta["merge_class"].unique()
mg = mg.reindex(meta.index, use_ids=True)
mg = mg.remove_pdiff()
mg = mg.make_lcc()
meta = mg.meta
print(f"{len(meta)} neurons in selected largest connected component")

#%% prepare data for NBLAST
# collapse
neuron_ids = [int(n) for n in meta.index]
neurons = pymaid.get_neuron(neuron_ids)  # load in with pymaid

# HACK: I am guessing there is a better way to do the below?
# TODO: I was also getting some errors about neurons with more that one soma, so I threw
# them out for now.
treenode_tables = []
for neuron_id, neuron in zip(neuron_ids, neurons):
    treenode_table = pymaid.get_treenode_table(neuron, include_details=False)
    treenode_tables.append(treenode_table)

success_neurons = []
tree_neurons = []
for neuron_id, treenode_table in zip(neuron_ids, treenode_tables):
    treenode_table.rename(columns={"parent_node_id": "parent_id"}, inplace=True)

    tree_neuron = TreeNeuron(treenode_table)
    if (tree_neuron.soma is not None) and (len(tree_neuron.soma) > 1):
        print(f"Neuron {neuron_id} has more than one soma, removing")
    else:
        tree_neurons.append(tree_neuron)
        success_neurons.append(neuron_id)

tree_neurons = NeuronList(tree_neurons)
meta = meta.loc[success_neurons]
mg = mg.reindex(success_neurons, use_ids=True)

print(f"{len(meta)} neurons ready for NBLAST")

#%% preprocessing for the adjacency
# collapse
adj = mg.adj
ptr_adj = pass_to_ranks(adj)

#%% [markdown]
# ## Running NBLAST and post-processing the scores
#%% run nblast
# collapse
currtime = time.time()
# NOTE: I've had too modify original code to allow smat=None
# NOTE: this only works when normalized=False also
scores = nblast_allbyall(tree_neurons, smat=None, normalized=False, progress=False)
print(f"{time.time() - currtime:.3f} elapsed to run NBLAST.")

#%% transform the raw nblast scores
# collapse
distance = scores.values  # the raw nblast scores are dissimilarities/distances
sym_distance = symmetrize(distance)  # the raw scores are not symmetric
# make the distances between 0 and 1
sym_distance /= sym_distance.max()
sym_distance -= sym_distance.min()
# and then convert to similarity
morph_sim = 1 - sym_distance

# rank transform the similarities
# NOTE this is very different from what native NBLAST does and could likely be improved
# upon a lot. I did this becuase it seemed like a quick way of accounting for difference
# in scale for different neurons as well as the fact that the raw distribution of
# similaritys was skewed high (very few small values)
quant = QuantileTransformer()
indices = np.triu_indices_from(morph_sim, k=1)
transformed_vals = quant.fit_transform(morph_sim[indices].reshape(-1, 1))
transformed_vals = np.squeeze(transformed_vals)
# this is a discrete version of PTR basically
ptr_morph_sim = np.ones_like(morph_sim)
ptr_morph_sim[indices] = transformed_vals
ptr_morph_sim[indices[::-1]] = transformed_vals

# # before
# plt.figure()
# sns.distplot(morph_sim[np.triu_indices_from(morph_sim, k=1)])
# # after
# plt.figure()
# sns.distplot(ptr_morph_sim[np.triu_indices_from(morph_sim, k=1)])

#%% [markdown]
# ## Plotting both modalities
# **Left**: the adjacency matrix for this subgraph after pass-to-ranks.
#
# **Right**: the similarity matrix obtained from NBLAST after some post-processing,
# including a quantile transform (like pass-to-ranks). See the code in the cell above
# for more explanation. Also likely subject to change.

#%%
# collapse
sns.set_context("talk", font_scale=1.25)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
adjplot(
    ptr_adj,
    sort_class=meta["class1"],
    colors=meta["class1"],
    palette=palette,
    tick_rot=45,
    cbar=False,
    ax=ax,
    title="Connectivity (adjacency matrix)",
)

ax = axs[1]
adjplot(
    ptr_morph_sim,
    sort_class=meta["class1"],
    colors=meta["class1"],
    palette=palette,
    tick_rot=45,
    cbar=False,
    ax=ax,
    title="Morphology (pairwise NBLAST)",
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
ptr_adj = pass_to_ranks(adj)


def careys_rule(X):
    """Get the number of singular values to check"""
    return int(np.ceil(np.log2(np.min(X.shape))))


zg_n_components = careys_rule(adj)
max_n_components = 40  # basically just how many to show for screeplots

embed_kws = dict(concat=False)

embedding = "ASE"
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
morph_embed = morph_model.fit_transform(ptr_morph_sim)

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
        y_pad=0.5,
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
# Based on the screeplots above, I selected 5 as the embedding dimension to use for
# stage 2.
#%% [markdown]
# ## Plot the connectivity and morphology embeddings from the first stage
# Here I show the first 6 dimensions of the embeddings (or the first 3 + 3 for a
# directed embedding).
#%%
# collapse
n_show = 6
labels = meta["class1"].values
n_show_per = n_show // 2
plot_adj_embed = np.concatenate(
    (adj_embed[0][:, :n_show_per], adj_embed[1][:, :n_show_per]), axis=1
)
col_names = [f"Out dimension {i + 1}" for i in range(n_show_per)]
col_names += [f"In dimension {i + 1}" for i in range(n_show_per)]
labels = meta["class1"].values
pairplot(
    plot_adj_embed,
    labels=labels,
    col_names=col_names,
    palette=palette,
    title="Connectivity screeplot",
    diag_kind="hist",
)
stashfig("connectivity-pairplot")

pairplot(
    morph_embed[:, :n_show],
    labels=labels,
    palette=palette,
    title="Morphology screeplot",
    diag_kind="hist",
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

n_components = 5  # setting based on the plots above

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
n_components = 10
umap = UMAP(min_dist=0.8, n_neighbors=20, metric="cosine")
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
    umap_embed, labels=labels, palette=palette, title=r"UMAP $\circ$ joint embedding"
)
stashfig("joint-umap")
#%% [markdown]
# ## Running discriminability
#%%
# collapse
uni_labels = np.unique(labels)
label_map = dict(zip(uni_labels, range(len(uni_labels))))
int_labels = np.vectorize(label_map.get)(labels)
stage1_n_components_range = [3, 5, 7, 10, 15]
stage2_n_components_range = [3, 5, 7, 10, 15]
metrics = ["euclidean", "cosine"]
rows = []
for stage1_n_components in stage1_n_components_range:
    concat_embed = np.concatenate(
        (
            adj_embed[0][:, :stage1_n_components],
            adj_embed[1][:, :stage1_n_components],
            morph_embed[:, :stage1_n_components],
        ),
        axis=1,
    )
    for stage2_n_components in stage2_n_components_range:
        if stage1_n_components * 3 < stage2_n_components:
            for metric in metrics:
                output = {
                    "metric": metric,
                    "stage2_n_components": stage2_n_components,
                    "stage1_n_components": stage1_n_components,
                    "discrim_tstat": None,
                    "embedding": "joint",
                }
                rows.append(output)
        else:
            joint_embed, joint_singular_values, _ = selectSVD(
                concat_embed, n_components=stage2_n_components, algorithm="full"
            )
            X = joint_embed[:, :stage2_n_components]
            for metric in metrics:
                pdist = pairwise_distances(X, metric=metric)
                discrim = DiscrimOneSample(is_dist=True)
                tstat, _ = discrim.test(pdist, int_labels, reps=0)
                output = {
                    "metric": metric,
                    "stage2_n_components": stage2_n_components,
                    "stage1_n_components": stage1_n_components,
                    "discrim_tstat": tstat,
                    "embedding": "joint",
                }
                rows.append(output)


for stage1_n_components in stage1_n_components_range:
    for embedding in ["connectivity", "morphology"]:
        if embedding == "connectivity":
            X = np.concatenate(
                (
                    adj_embed[0][:, :stage1_n_components],
                    adj_embed[1][:, :stage1_n_components],
                ),
                axis=1,
            )
        else:
            X = morph_embed[:, :stage1_n_components]
        for metric in metrics:
            pdist = pairwise_distances(X, metric=metric)
            discrim = DiscrimOneSample(is_dist=True)
            tstat, _ = discrim.test(pdist, int_labels, reps=0)
            output = {
                "metric": metric,
                "stage2_n_components": None,
                "stage1_n_components": stage1_n_components,
                "discrim_tstat": tstat,
                "embedding": embedding,
            }
            rows.append(output)

results = pd.DataFrame(rows)

#%% [markdown]
# ## Discriminability plotted as a function of stage 1 and stage 2 dimension
#%%
# collapse
pivot_kws = dict(
    index="stage1_n_components", columns="stage2_n_components", values="discrim_tstat"
)
heatmap_kws = dict(
    square=True,
    annot=True,
    cbar=False,
    vmin=results["discrim_tstat"].min(),
    vmax=1,
    cmap="RdBu_r",
    center=results["discrim_tstat"].min(),
)

fig, axs = plt.subplots(
    2,
    3,
    figsize=(12, 12),
    gridspec_kw=dict(width_ratios=[0.3, 0.3, 1]),
)

# euclidean, connectivity
ax = axs[0, 0]
plot_results = results[
    (results["embedding"] == "connectivity") & (results["metric"] == "euclidean")
].pivot(**pivot_kws)
sns.heatmap(plot_results.values, ax=ax, **heatmap_kws)
ax.set(
    ylabel="Metric = Euclidean\n\nStage 1 # components",
    yticklabels=stage1_n_components_range,
    xticks=[],
    title="Connectivity",
)

# euclidean, morphology
ax = axs[0, 1]
plot_results = results[
    (results["embedding"] == "morphology") & (results["metric"] == "euclidean")
].pivot(**pivot_kws)
sns.heatmap(plot_results.values, ax=ax, **heatmap_kws)
ax.set(
    xticks=[],
    yticks=[],
    title="Morphology",
)

# euclidean, MASE
ax = axs[0, 2]
plot_results = results[
    (results["metric"] == "euclidean") & (results["embedding"] == "joint")
].pivot(**pivot_kws)
sns.heatmap(plot_results.values, ax=ax, **heatmap_kws)
ax.set(
    xlabel="",
    yticks=[],
    xticks=[],
    title="Joint (MASE)",
)

# cosine, connectivity
ax = axs[1, 0]
plot_results = results[
    (results["embedding"] == "connectivity") & (results["metric"] == "cosine")
].pivot(**pivot_kws)
sns.heatmap(plot_results.values, ax=ax, **heatmap_kws)
ax.set(
    ylabel="Metric = cosine\n\nStage 1 # components",
    yticklabels=stage1_n_components_range,
    xticks=[],
)

# cosine, morphology
ax = axs[1, 1]
plot_results = results[
    (results["embedding"] == "morphology") & (results["metric"] == "cosine")
].pivot(**pivot_kws)
sns.heatmap(plot_results.values, ax=ax, **heatmap_kws)
ax.set(xticks=[], yticks=[])

# cosine, MASE
ax = axs[1, 2]
plot_results = results[
    (results["metric"] == "cosine") & (results["embedding"] == "joint")
].pivot(**pivot_kws)
sns.heatmap(plot_results, ax=ax, **heatmap_kws)
ax.set(
    ylabel="",
    xlabel="Stage 2 # components",
    xticklabels=stage2_n_components_range,
    yticks=[],
)

fig.suptitle("Discriminability", y=0.97)
stashfig("discrim-by-dimensionality")
#%%
# collapse
print(f"{(time.time() - t0)/60:.3f} minutes elapsed for whole notebook.")
