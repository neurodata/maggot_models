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
import pymaid
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import poisson
from scipy.stats.mstats import gmean
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from umap import UMAP

from graspologic.cluster import DivisiveCluster
from graspologic.embed import (
    AdjacencySpectralEmbed,
    LaplacianSpectralEmbed,
    select_dimension,
    selectSVD,
)
from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.plot import pairplot
from graspologic.utils import (
    binarize,
    pass_to_ranks,
    remap_labels,
    remove_loops,
    symmetrize,
)
from src.data import load_metagraph
from src.io import savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    plot_volumes,
    set_axes_equal,
    set_theme,
)

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


def careys_rule(X):
    """Get the number of singular values to check"""
    return int(np.ceil(np.log2(np.min(X.shape))))


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


def unscale(X):
    # TODO implement as a setting in graspologic
    norms = np.linalg.norm(X, axis=0)
    X = X / norms[None, :]
    return X


def simple_scatterplot(X, labels=None, palette="deep", ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plot_df = pd.DataFrame(data=X[:, :2], columns=["0", "1"])
    plot_df["labels"] = labels
    sns.scatterplot(data=plot_df, x="0", y="1", hue="labels", palette=palette, ax=ax)
    ax.set(xlabel="", ylabel="", title=title, xticks=[], yticks=[])
    ax.get_legend().remove()
    return ax


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

    # load nblast similarities
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


def _check_matrices(matrices):
    if isinstance(matrices, np.ndarray) and (matrices.ndim == 2):
        matrices = [matrices]
    return matrices


class JointEmbed(BaseEstimator):
    def __init__(
        self,
        n_components=None,
        stage1_n_components=None,
        method="ase",
        scaled=False,
        algorithm="randomized",
        embed_kws={},
    ):
        self.n_components = n_components
        self.stage1_n_components = stage1_n_components
        self.embed_kws = embed_kws
        self.method = method
        self.scaled = scaled
        self.algorithm = algorithm
        self.embed_kws = embed_kws

    def _embed_matrices(self, matrices):
        embeddings = []
        models = []
        for matrix in matrices:
            model = self._Embed(n_components=self.stage1_n_components, **self.embed_kws)
            embedding = model.fit_transform(matrix)
            embeddings.append(embedding)
            models.append(model)
        return embeddings, models

    def fit_transform(self, graphs, similarities, weights=None):
        if self.method == "ase":
            self._Embed = AdjacencySpectralEmbed
        elif self.method == "lse":
            self._Embed = LaplacianSpectralEmbed
            if self.embed_kws == {}:
                self.embed_kws["form"] = "R-DAD"

        graphs = _check_matrices(graphs)
        similarities = _check_matrices(similarities)

        if weights is None:
            weights = np.ones(len(graphs) + len(similarities))

        # could all be in one call, but want to leave open the option of treating them
        # separately.
        graph_embeddings, graph_models = self._embed_matrices(graphs)
        similarity_embeddings, similarity_models = self._embed_matrices(similarities)

        self.graph_models_ = graph_models
        self.similarity_models_ = similarity_models

        embeddings = graph_embeddings + similarity_embeddings
        concat_embeddings = []
        for i, embedding in enumerate(embeddings):
            if not isinstance(embedding, tuple):
                embedding = (embedding,)
            for e in embedding:
                if not self.scaled:
                    e = unscale(e)
                concat_embeddings.append(weights[i] * e)

        concat_embeddings = np.concatenate(concat_embeddings, axis=1)

        joint_embedding, joint_singular_values, _ = selectSVD(
            concat_embeddings, n_components=self.n_components, algorithm=self.algorithm
        )

        self.singular_values_ = joint_singular_values
        return joint_embedding


def calc_model_liks(adj, meta, lp_inds, rp_inds, n_levels=10):
    rows = []
    for level in range(n_levels + 1):
        labels = meta[f"lvl{level}_labels"].values
        left_adj = binarize(adj[np.ix_(lp_inds, lp_inds)])
        left_adj = remove_loops(left_adj)
        right_adj = binarize(adj[np.ix_(rp_inds, rp_inds)])
        right_adj = remove_loops(right_adj)
        for model, name in zip([DCSBMEstimator, SBMEstimator], ["DCSBM", "SBM"]):
            estimator = model(directed=True, loops=False)
            uni_labels, inv = np.unique(labels, return_inverse=True)
            estimator.fit(left_adj, inv[lp_inds])
            train_left_p = estimator.p_mat_
            train_left_p[train_left_p == 0] = 1 / train_left_p.size

            n_params = estimator._n_parameters() + len(uni_labels)

            score = poisson.logpmf(left_adj, train_left_p).sum()
            rows.append(
                dict(
                    train_side="Left",
                    test="Same",
                    test_side="Left",
                    score=score,
                    level=level,
                    model=name,
                    n_params=n_params,
                    norm_score=score / left_adj.sum(),
                )
            )
            score = poisson.logpmf(right_adj, train_left_p).sum()
            rows.append(
                dict(
                    train_side="Left",
                    test="Opposite",
                    test_side="Right",
                    score=score,
                    level=level,
                    model=name,
                    n_params=n_params,
                    norm_score=score / right_adj.sum(),
                )
            )

            estimator = model(directed=True, loops=False)
            estimator.fit(right_adj, inv[rp_inds])
            train_right_p = estimator.p_mat_
            train_right_p[train_right_p == 0] = 1 / train_right_p.size

            n_params = estimator._n_parameters() + len(uni_labels)

            score = poisson.logpmf(left_adj, train_right_p).sum()
            rows.append(
                dict(
                    train_side="Right",
                    test="Opposite",
                    test_side="Left",
                    score=score,
                    level=level,
                    model=name,
                    n_params=n_params,
                    norm_score=score / left_adj.sum(),
                )
            )
            score = poisson.logpmf(right_adj, train_right_p).sum()
            rows.append(
                dict(
                    train_side="Right",
                    test="Same",
                    test_side="Right",
                    score=score,
                    level=level,
                    model=name,
                    n_params=n_params,
                    norm_score=score / right_adj.sum(),
                )
            )
    return pd.DataFrame(rows)


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


def plot_pairedness(meta, lp_inds, rp_inds, ax, n_levels=10, n_shuffles=10):
    rows = []
    for level in range(n_levels + 1):
        pred_labels = meta[f"lvl{level}_labels"].values.copy()
        p_same = calc_pairedness(pred_labels, lp_inds, rp_inds)
        rows.append(dict(p_same_cluster=p_same, labels="True", level=level))
        # look at random chance
        for i in range(n_shuffles):
            np.random.shuffle(pred_labels)
            p_same = calc_pairedness(pred_labels, lp_inds, rp_inds)
            rows.append(dict(p_same_cluster=p_same, labels="Shuffled", level=level))
    plot_df = pd.DataFrame(rows)

    sns.lineplot(
        data=plot_df,
        x="level",
        y="p_same_cluster",
        ax=ax,
        hue="labels",
        markers=True,
        style="labels",
    )
    ax.set_ylabel("P same cluster")
    ax.set_xlabel("Level")


def calc_pairedness(pred_labels, lp_inds, rp_inds):
    left_labels = pred_labels[lp_inds]
    right_labels = pred_labels[rp_inds]
    right_labels = remap_labels(
        left_labels, right_labels
    )  # To make chance comparison fair
    n_same = (left_labels == right_labels).sum()
    p_same = n_same / len(lp_inds)
    return p_same


def calc_discriminability_statistic(pdist, y):
    uni_labels = np.unique(y)

    all_sample_discrims = []
    for i, row in enumerate(pdist):
        same_mask = y == y[i]
        diff_mask = ~same_mask
        same_mask[i] = False
        dist_same_label = row[same_mask]
        dist_diff_label = row[diff_mask]
        sample_discrim = np.less(
            dist_same_label[:, None], dist_diff_label[None, :]
        ).mean()
        # Takes roughly double the time to check, true ties are quite rare
        # sample_discrim += 0.5*np.equal(dist_same_label[:, None], dist_diff_label[None, :]).mean()
        all_sample_discrims.append(sample_discrim)

    all_sample_discrims = np.array(all_sample_discrims)
    total_discrim = np.mean(all_sample_discrims)

    class_discrim = {}
    for ul in uni_labels:
        class_discrim[ul] = np.mean(all_sample_discrims[y == ul])
    return total_discrim, class_discrim


def simple_plot_neurons(
    neurons, azim=-90, elev=0, dist=5, use_y=False, use_z=True, use_x=True
):
    neurons = [int(n) for n in neurons]

    skeleton_color_dict = dict(
        zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
    )

    plot_mode = "3d"

    pymaid.plot2d(
        neurons,
        color=skeleton_color_dict,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=False,
    )
    plot_volumes(volumes, ax)
    if plot_mode == "3d":
        ax.azim = azim
        ax.elev = elev  # 35
        ax.dist = dist
        set_axes_equal(ax, use_y=use_y, use_x=use_x, use_z=use_z)


#%% [markdown]
# ## The data
# Here use the *Drosophila* larva connectome.

# For the purposes of the network, I use the "summed" or union graph (not separating by
# axon/dendrite). I also restrict this analysis to one hemisphere (the left).
#%% load connectivity data
# collapse
data_dir = Path("maggot_models/experiments/nblast/outs")
side = "left"

walksort_loc = "maggot_models/experiments/walk_sort/outs/meta_w_order-gt=Gad-n_init=256-hops=16-loops=False-include_reverse=False.csv"
walksort_meta = pd.read_csv(walksort_loc, index_col=0)

mg, nblast_sim = load_side("left", nblast_type="scores")
meta = mg.meta
adj = mg.adj
ptr_adj = pass_to_ranks(adj)

#%% work on preprocessing the scores
# collapse


transform = "quantile"

symmetrize_mode = "geom"


distance = nblast_sim.values  # the raw nblast scores are dissimilarities/distances
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
else:
    sym_distance = symmetrize(distance)

# sym_distance = symmetrize(distance)  # the raw scores are not symmetric
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
elif transform == "log":
    raise NotImplementedError()
else:
    transformed_morph = morph_sim

fig, axs = plt.subplots(2, 1, figsize=(5, 10))
ax = axs[0]
sns.histplot(morph_sim[indices], ax=ax)
ax = axs[1]
sns.histplot(transformed_morph[indices], ax=ax)
plt.tight_layout()

#%%

quantile = 0.995
argsort_sim_inds = np.argsort(transformed_morph[indices])


def get_neurons_by_quantile(meta, argsort_sim_inds, quantile, boost=0):
    choice_ind = int(np.floor(len(argsort_sim_inds) * quantile)) + boost
    choice_ind = argsort_sim_inds[choice_ind]
    row_neuron_index = indices[0][choice_ind]
    col_neuron_index = indices[1][choice_ind]
    row_neuron_id = int(meta.index[row_neuron_index])
    col_neuron_id = int(meta.index[col_neuron_index])
    return row_neuron_id, col_neuron_id


volume_names = ["PS_Neuropil_manual"]
volumes = [pymaid.get_volume(v) for v in volume_names]


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
    simple_plot_neurons(neurons)
    if i == 0:
        ax.set_title(f"Quantile: {q}", fontsize="small")
    else:
        ax.set_title(f"{q}", fontsize="small")
stashfig(f"nblast-difference-quantiles-boost={boost}")
#%%
# collapse

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

#%%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
umapper = UMAP(min_dist=0.8, n_neighbors=30, metric="precomputed")
umap_morph_embedding = umapper.fit_transform(1 - transformed_morph)
sns.scatterplot(
    x=umap_morph_embedding[:, 0],
    y=umap_morph_embedding[:, 1],
    hue=meta["merge_class"].values,
    palette=CLASS_COLOR_DICT,
    s=15,
    alpha=0.5,
    linewidth=0,
    ax=ax,
    legend=False,
)
ax.axis("off")

#%% [markdown]
# ## Plotting the data for both modalities
# **Left**: the adjacency matrix for this subgraph after pass-to-ranks.
#
# **Right**: the similarity matrix obtained from NBLAST after some post-processing,
# including a quantile transform (like pass-to-ranks). Also likely subject to change.

#%%
# collapse
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
n_components = 6
umap = UMAP(min_dist=0.7, n_neighbors=30, metric="euclidean")
umap_embed = umap.fit_transform(joint_embed[:, :n_components])

simple_scatterplot(
    umap_embed,
    labels=meta["merge_class"].values,
    palette=palette,
    title=r"UMAP $\circ$ joint embedding",
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


#%% need to actually quantify the above
# TODO how much variance is there from run to run of DivisiveCluster? Must just be the
# kmeans init, right?

# TODO need to port over the code to compute model likelihood and evaluate. Should be
# able to run on the opposite hemisphere connectivity as well.

# TODO need to come up with a metric for evaluating how good the morphological
# clustering is? maybe just something like modularity or discriminability on the morph
# distances?

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
    je = JointEmbed(n_components=6, stage1_n_components=6, method="lse")
    joint_embedding = je.fit_transform(adj, sim, weights=[1, 0])

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
mg, nblast_sim = load_side(side, nblast_type="similarities")
adj = pass_to_ranks(mg.adj)
sim = nblast_sim.values

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
n_show_neurons = 1  # 000

sns.set_context("talk", font_scale=0.75)
n_cols = 6
n_rows = int(np.ceil(len(show_inds) / n_cols))
fig = plt.figure(figsize=(12, 12 / n_cols * n_rows * 1.1))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0.03, hspace=0.1)
axs = np.empty((n_rows, n_cols), dtype=object)
for i, result_idx in enumerate(show_inds):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection="3d")
    axs[inds] = ax
    neurons = level_results.iloc[result_idx]["neuron_ids"][:n_show_neurons]
    simple_plot_neurons(neurons, dist=6)
    cluster_discrim = level_results.iloc[result_idx]["cluster_discrim"]
    if i == 0:
        ax.set_title(
            f"Discriminability:\n{cluster_discrim:0.3f} ({len(neurons)})",
        )
    else:
        ax.set_title(f"{cluster_discrim:0.3f} ({len(neurons)})")
axs[0, 0].set_ylabel("Bottom 20 clusters")
axs[0, 2].set_ylabel("Top 20 clusters")
stashfig(f"discriminability-cluster-morpho-level={level}")


# ## Points to make
# - geometric mean for symmetrizing distance is the way to go
# - discriminability can be used component-wise to sort clusters in terms of morphology
#   similarity scores