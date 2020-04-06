# %% [markdown]
# # THE MIND OF A MAGGOT

# %% [markdown]
# ## Imports
import os
import time

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import rdpg
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    barplot_text,
    gridmap,
    matrixplot,
    stacked_barplot,
    adjplot,
)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

CLUSTER_SPLIT = "best"

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def get_paired_inds(meta):
    pair_meta = meta[meta["Pair"].isin(meta.index)]
    pair_group_size = pair_meta.groupby("Pair ID").size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta["Pair ID"].isin(remove_pairs)]
    assert pair_meta.groupby("Pair ID").size().min() == 2
    pair_meta.sort_values(["Pair ID", "hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["hemisphere"] == "L"]["inds"]
    rp_inds = pair_meta[pair_meta["hemisphere"] == "R"]["inds"]
    assert (
        meta.iloc[lp_inds]["Pair ID"].values == meta.iloc[rp_inds]["Pair ID"].values
    ).all()
    return lp_inds, rp_inds


# TODO broken in some cases, switched to `compute_pairedness_bipartite`
def compute_pairedness(partition, left_pair_inds, right_pair_inds, plot=False):
    uni_labels, inv = np.unique(partition, return_inverse=True)
    train_int_mat = np.zeros((len(uni_labels), len(uni_labels)))
    for i, ul in enumerate(uni_labels):
        c1_mask = inv == i
        for j, ul in enumerate(uni_labels):
            c2_mask = inv == j
            # number of times a thing in cluster 1 has a pair also in cluster 2
            pairs_in_other = np.logical_and(
                c1_mask[left_pair_inds], c2_mask[right_pair_inds]
            ).sum()
            train_int_mat[i, j] = pairs_in_other

    row_ind, col_ind = linear_sum_assignment(train_int_mat, maximize=True)
    train_pairedness = np.trace(train_int_mat[np.ix_(row_ind, col_ind)]) / np.sum(
        train_int_mat
    )  # TODO double check that this is right

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(
            train_int_mat, square=True, ax=axs[0], cbar=False, cmap="RdBu_r", center=0
        )
        int_df = pd.DataFrame(data=train_int_mat, index=uni_labels, columns=uni_labels)
        int_df = int_df.reindex(index=uni_labels[row_ind])
        int_df = int_df.reindex(columns=uni_labels[col_ind])
        sns.heatmap(int_df, square=True, ax=axs[1], cbar=False, cmap="RdBu_r", center=0)

    return train_pairedness, row_ind, col_ind


def compute_pairedness_bipartite(left_labels, right_labels):
    left_uni_labels, left_inv = np.unique(left_labels, return_inverse=True)
    right_uni_labels, right_inv = np.unique(right_labels, return_inverse=True)
    train_int_mat = np.zeros((len(left_uni_labels), len(right_uni_labels)))
    for i, ul in enumerate(left_uni_labels):
        c1_mask = left_inv == i
        for j, ul in enumerate(right_uni_labels):
            c2_mask = right_inv == j
            # number of times a thing in cluster 1 has a pair also in cluster 2
            pairs_in_other = np.logical_and(c1_mask, c2_mask).sum()
            train_int_mat[i, j] = pairs_in_other

    row_ind, col_ind = linear_sum_assignment(train_int_mat, maximize=True)
    train_pairedness = np.trace(train_int_mat[np.ix_(row_ind, col_ind)]) / np.sum(
        train_int_mat
    )  # TODO double check that this is right
    return train_pairedness, row_ind, col_ind


def fit_and_score(X_train, X_test, k, **kws):
    gc = GaussianCluster(min_components=k, max_components=k, **kws)
    gc.fit(X_train)
    model = gc.model_
    train_bic = model.bic(X_train)
    train_lik = model.score(X_train)
    test_bic = model.bic(X_test)
    test_lik = model.score(X_test)
    bic = model.bic(np.concatenate((X_train, X_test), axis=0))
    res = {
        "train_bic": -train_bic,
        "train_lik": train_lik,
        "test_bic": -test_bic,
        "test_lik": test_lik,
        "bic": -bic,
        "lik": train_lik + test_lik,
        "k": k,
        "model": gc.model_,
    }
    return res, model


def crossval_cluster(
    embed,
    left_inds,
    right_inds,
    min_clusters=2,
    max_clusters=15,
    n_init=25,
    left_pair_inds=None,
    right_pair_inds=None,
):
    left_embed = embed[left_inds]
    right_embed = embed[right_inds]
    print("Running left/right clustering with cross-validation\n")
    currtime = time.time()
    rows = []
    for k in tqdm(range(min_clusters, max_clusters)):
        # TODO add option for AutoGMM as well, might as well check
        for i in range(n_init):
            left_row, left_gc = fit_and_score(left_embed, right_embed, k)
            left_row["train"] = "left"
            right_row, right_gc = fit_and_score(right_embed, left_embed, k)
            right_row["train"] = "right"

            # pairedness computation, if available
            if left_pair_inds is not None and right_pair_inds is not None:
                # TODO double check this is right
                pred_left = left_gc.predict(embed[left_pair_inds])
                pred_right = right_gc.predict(embed[right_pair_inds])
                pness, _, _ = compute_pairedness_bipartite(pred_left, pred_right)
                left_row["pairedness"] = pness
                right_row["pairedness"] = pness

                ari = adjusted_rand_score(pred_left, pred_right)
                left_row["ARI"] = ari
                right_row["ARI"] = ari

            rows.append(left_row)
            rows.append(right_row)

    results = pd.DataFrame(rows)
    print(f"{time.time() - currtime} elapsed")
    return results


def plot_crossval_cluster(results):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax = axs[0]
    sns.lineplot(data=results, x="k", y="test_lik", hue="train", ax=ax, legend=False)
    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")
    sns.lineplot(data=results, x="k", y="train_lik", hue="train", ax=ax, legend=False)
    ax.set_ylabel("Log likelihood")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))

    ax = axs[1]
    sns.lineplot(data=results, x="k", y="test_bic", hue="train", ax=ax, legend="full")
    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")
    sns.lineplot(data=results, x="k", y="train_bic", hue="train", ax=ax, legend="full")
    ax.set_ylabel("-BIC")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))

    leg = ax.legend()
    leg.set_title("Train side")
    leg.texts[0].set_text("Test contra")
    leg.set_bbox_to_anchor((1, 1.8))
    lines = leg.get_lines()
    lines[0].set_linestyle("--")
    lines[1].set_linestyle("--")
    lines[2].set_linestyle("--")
    leg.texts[3].set_text("Test ipsi")

    ax = axs[2]
    sns.lineplot(
        data=results,
        x="k",
        y="pairedness",
        ax=ax,
        legend="full",
        color="purple",
        label="Pairedness",
    )
    sns.lineplot(
        data=results, x="k", y="ARI", ax=ax, legend="full", color="green", label="ARI"
    )
    ax.set_ylabel("Pair score")
    leg = ax.legend().remove()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # leg.loc = 2
    # leg.set_bbox_to_anchor((1, 1))

    # ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))
    # trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    # ax.text(0.8, 0.8, "Pairedness", color="purple", transform=trans)
    # ax.text(0.8, 0.6, "ARI", color="green", transform=trans)
    return fig, axs


def make_ellipses(gmm, ax, i, j, colors, alpha=0.5, equal=False, **kws):
    inds = [j, i]
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][np.ix_(inds, inds)]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[np.ix_(inds, inds)]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][inds])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, inds], v[0], v[1], 180 + angle, color=color, **kws
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ax.add_artist(ell)
        if equal:
            ax.set_aspect("equal", "datalim")


def plot_cluster_pairs(
    X, left_inds, right_inds, left_model, right_model, labels, colors=None, equal=True
):
    k = left_model.n_components
    n_dims = X.shape[1]

    if colors is None:
        colors = sns.color_palette("tab10", n_colors=k, desat=0.7)

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels  #
    pred = composite_predict(
        X, left_inds, right_inds, left_model, right_model, relabel=False
    )
    data["pred"] = pred

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.5,
                    linewidth=0,
                    s=5,
                    legend=False,
                    hue="label",
                    palette=CLASS_COLOR_DICT,
                )
                make_ellipses(left_model, ax, i, j, colors, fill=False, equal=equal)
            if i > j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=5,
                    legend=False,
                    hue="pred",
                    palette=colors,
                )
                make_ellipses(left_model, ax, i, j, colors, fill=True, equal=equal)

    plt.tight_layout()
    return fig, axs


def composite_predict(X, left_inds, right_inds, left_model, right_model, relabel=False):
    # TODO add option to boost the right numbers
    X_left = X[left_inds]
    X_right = X[right_inds]
    pred_left = left_model.predict(X_left)
    pred_right = right_model.predict(X_right)
    if relabel:
        leftify = np.vectorize(lambda x: str(x) + "L")
        rightify = np.vectorize(lambda x: str(x) + "R")
        pred_left = leftify(pred_left)
        pred_right = rightify(pred_right)
    dtype = pred_left.dtype
    pred = np.empty(len(X), dtype=dtype)
    pred[left_inds] = pred_left
    pred[right_inds] = pred_right
    return pred


def reindex_model(gmm, perm_inds):
    gmm.weights_ = gmm.weights_[perm_inds]
    gmm.means_ = gmm.means_[perm_inds]
    if gmm.covariance_type != "tied":
        gmm.covariances_ = gmm.covariances_[perm_inds]
        gmm.precisions_ = gmm.precisions_[perm_inds]
        gmm.precisions_cholesky_ = gmm.precisions_cholesky_[perm_inds]
    return gmm


def plot_metrics(results, plot_all=True):
    plot_results = results.copy()
    plot_results["k"] += np.random.normal(size=len(plot_results), scale=0.1)

    fig, axs = plt.subplots(3, 3, figsize=(20, 10), sharex=True)

    def miniplotter(var, ax):
        if plot_all:
            sns.scatterplot(
                data=plot_results,
                x="k",
                y=var,
                hue="train",
                ax=ax,
                s=8,
                linewidth=0,
                alpha=0.5,
            )
        best_inds = results.groupby(["k"])[var].idxmax()
        best_results = results.loc[best_inds].copy()
        sns.lineplot(
            data=best_results, x="k", y=var, ax=ax, color="purple", label="max"
        )
        mean_results = results.groupby(["k"]).mean()
        mean_results.reset_index(inplace=True)
        sns.lineplot(
            data=mean_results, x="k", y=var, ax=ax, color="green", label="mean"
        )
        ax.get_legend().remove()

    plot_vars = [
        "train_lik",
        "test_lik",
        "lik",
        "train_bic",
        "test_bic",
        "bic",
        "ARI",
        "pairedness",
    ]
    axs = axs.T.ravel()

    for pv, ax in zip(plot_vars, axs):
        miniplotter(pv, ax)

    axs[2].xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    axs[-2].tick_params(labelbottom=True)
    axs[-2].set_xlabel("k")

    handles, labels = axs[-2].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc="upper left")
    axs[-1].axis("off")

    return fig, axs


# %% [markdown]
# ## Load data
# In this case we are working with `G`, the directed graph formed by summing the edge
# weights of the 4 different graph types. Preprocessing here includes removing
# partially differentiated cells, and cutting out the lowest 5th percentile of nodes in
# terms of their number of incident synapses. 5th percentile ~= 12 synapses. After this,
# the largest connected component is used.

mg = load_metagraph("G", version="2020-04-01")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

# plot where we are cutting out nodes based on degree
degrees = mg.calculate_degrees()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sns.distplot(np.log10(degrees["Total edgesum"]), ax=ax)
q = np.quantile(degrees["Total edgesum"], 0.05)
ax.axvline(np.log10(q), linestyle="--", color="r")
ax.set_xlabel("log10(total synapses)")

# remove low degree neurons
idx = meta[degrees["Total edgesum"] > q].index
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)
meta = mg.meta

adj = mg.adj
adj = pass_to_ranks(adj)
meta["inds"] = range(len(meta))

left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]
lp_inds, rp_inds = get_paired_inds(meta)


# %% [markdown]
# ## Embed
# Here the embedding is ASE, with PTR and DiagAug, the number of embedding dimensions
# is for now set to ZG2 (4 + 4). Using the known pairs as "seeds", the left embedding
# is matched to the right using procrustes.
ase = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
embed = ase.fit_transform(adj)
n_components = embed[0].shape[1]  # use all of ZG2
X = np.concatenate((embed[0][:, :n_components], embed[1][:, :n_components]), axis=-1)
R, _ = orthogonal_procrustes(X[lp_inds], X[rp_inds])

if CLUSTER_SPLIT == "best":
    X[left_inds] = X[left_inds] @ R

# %% [markdown]
# ## Clustering
# Clustering is performed using Gaussian mixture modeling. At each candidate value of k,
# 50 models are trained on the left embedding, 50 models are trained on the right
# embedding (choosing the best covariance structure based on BIC on the train set).
results = crossval_cluster(
    X,
    left_inds,
    right_inds,
    left_pair_inds=lp_inds,
    right_pair_inds=rp_inds,
    max_clusters=15,
    n_init=50,
)
# best_inds = results.groupby(["k", "train"])["test_bic"].idxmax()
# best_results = results.loc[best_inds].copy()
# plot_crossval_cluster(best_results)
# stashfig(f"cross-val-n_components={n_components}")

# %% [markdown]
# ## Evaluating Clustering
# Of the 100 models we fit as described above, we now evaluate them on a variety of
# metrics:
#  - likelihood of the data the model was trained on ("train_lik")
#  - likelihood of the held out (other hemisphere) data ("test_lik")
#  - likelihood of all of the data ("lik", = "train_lik" + "test_lik")
#  - BIC using the data the model was trained on ("train_bic")
#  - BIC using the held out (other hemisphere) data ("test_bic")
#  - BIC using all of the data ("bic")
#  - ARI for pairs. Given the prediction of the model on the left data and the right
#    data, using known pairs to define a correspondence between (some) nodes, what is
#    the ARI(left_prediction, right_prediciton) for the given model
#  - Pairedness, like the above but simply the raw fraction of pairs that end up in
#    corresponding L/R clusters. Very related to ARI but not normalized.

plot_metrics(results)
stashfig(f"cluster-metrics-n_components={n_components}")


# %% [markdown]
# ## Choose a model
# A few things are clear from the above. One is that the likelihood on the train set
# continues to go up as `k` increases, but plateaus and then drops on the test set around
# k = 6 - 8. This is even slightly more clear when looking at the BIC plots, where the
# only difference is the added penalty for complexity. Based on this, I would say that
# the best k at this scale is around 6-8; however, we still need to pick a single metric
# to give us the *best* model to proceed. I'm not sure whether it makes more sense to use
# likelihood or bic here, or, to use performance on the test set or performance on all
# of the data. Here we will proceed with k=7, and choose the model with the best BIC on
# all of the data.

k = 6
metric = "bic"
basename = f"-metric={metric}-k={k}-n_components={n_components}"
basetitle = f"Metric={metric}, k={k}, n_components={n_components}"

ind = results[results["k"] == k][metric].idxmax()

print(f"Choosing model at k={k} based on best {metric}.\n")
print(f"ARI: {results.loc[ind, 'ARI']}")
print(f"Pairedness: {results.loc[ind, 'pairedness']}\n")

model = results.loc[ind, "model"]
left_model = model
right_model = model

pred = composite_predict(
    X, left_inds, right_inds, left_model, right_model, relabel=False
)
pred_side = composite_predict(
    X, left_inds, right_inds, left_model, right_model, relabel=True
)

ax = stacked_barplot(
    pred_side, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT, legend_ncol=6
)
ax.set_title(basetitle)
stashfig(f"barplot" + basename)


fig, ax = plot_cluster_pairs(
    X, left_inds, right_inds, left_model, right_model, meta["merge_class"].values
)
fig.suptitle(basetitle, y=1)

stashfig(f"pairs" + basename)


sf = signal_flow(adj)
meta["signal_flow"] = -sf
meta["pred"] = pred
meta["pred_side"] = pred_side
meta["group_signal_flow"] = meta["pred"].map(meta.groupby("pred")["signal_flow"].mean())
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    ax=ax,
    meta=meta,
    sort_class="pred_side",
    class_order="group_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
)
fig.suptitle(basetitle, y=0.94)
stashfig(f"adj-sf" + basename)

meta["te"] = -meta["Total edgesum"]
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    ax=ax,
    meta=meta,
    sort_class="pred_side",
    class_order="group_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "te"],
    plot_type="scattermap",
    sizes=(0.5, 1),
)
fig.suptitle(basetitle, y=0.94)
stashfig(f"adj-te" + basename)

meta["rand"] = np.random.uniform(size=len(meta))
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    ax=ax,
    meta=meta,
    sort_class="pred_side",
    class_order="group_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order="rand",
    plot_type="scattermap",
    sizes=(0.5, 1),
)
fig.suptitle(basetitle, y=0.94)
stashfig(f"adj-rand" + basename)

# %% [markdown]
# ## SUBCLUSTER

np.random.seed(8888)

uni_labels, inv = np.unique(pred, return_inverse=True)
all_sub_results = []
sub_data = []

reembed = False

for label in uni_labels:
    print(label)
    print()
    label_mask = pred == label
    sub_meta = meta[label_mask].copy()
    sub_meta["inds"] = range(len(sub_meta))
    sub_left_inds = sub_meta[sub_meta["left"]]["inds"].values
    sub_right_inds = sub_meta[sub_meta["right"]]["inds"].values
    sub_lp_inds, sub_rp_inds = get_paired_inds(sub_meta)
    sub_adj = adj[np.ix_(label_mask, label_mask)]

    if reembed:
        ase = AdjacencySpectralEmbed()
        # TODO look into PTR at this level as well
        sub_embed = ase.fit_transform(sub_adj)
        sub_X = np.concatenate(sub_embed, axis=1)
        sub_R, _ = orthogonal_procrustes(sub_X[sub_lp_inds], sub_X[sub_rp_inds])
        sub_X[sub_left_inds] = sub_X[sub_left_inds] @ sub_R
    else:
        sub_X = X[label_mask].copy()
        sub_R = R

    var_dict = {
        "meta": sub_meta,
        "left_inds": sub_left_inds,
        "right_inds": sub_right_inds,
        "left_pair_inds": sub_lp_inds,
        "right_pair_inds": sub_rp_inds,
        "X": sub_X,
        "adj": sub_adj,
    }

    sub_data.append(var_dict)

    sub_results = crossval_cluster(
        sub_X,
        sub_left_inds,
        sub_right_inds,
        left_pair_inds=sub_lp_inds,
        right_pair_inds=sub_rp_inds,
        max_clusters=10,
        min_clusters=1,
        n_init=50,
    )

    fig, axs = plot_metrics(sub_results, plot_all=False)
    fig.suptitle(f"Subclustering for cluster {label}, reembed={reembed}")
    stashfig(f"sub-cluster-profile-label={label}-reembed={reembed}")
    plt.close()
    all_sub_results.append(sub_results)

# %% [markdown]
# ##
# sub_ks = [(2, 4), (0,), (3, 4), (3,), (2, 3), (0,), (4,)]
# sub_kws = [(4,), (0,), (4,), (3, 4), (2, 3), (3,), (3, 4, 5)]
if not reembed:
    sub_ks = [(4,), (4,), (3,), (2, 3, 4), (0,), (3,)]
else:
    pass


for i, label in enumerate(uni_labels):
    ks = sub_ks[i]
    sub_results = all_sub_results[i]
    sub_X = sub_data[i]["X"]
    sub_left_inds = sub_data[i]["left_inds"]
    sub_right_inds = sub_data[i]["right_inds"]
    sub_lp_inds = sub_data[i]["left_pair_inds"]
    sub_rp_inds = sub_data[i]["right_pair_inds"]
    sub_meta = sub_data[i]["meta"]

    fig, axs = plot_metrics(sub_results)
    fig.suptitle(f"Subclustering for cluster {label}, reembed={reembed}")
    for ax in axs[:-1]:
        for k in ks:
            ax.axvline(k, linestyle="--", color="red", linewidth=2)
    stashfig(f"sub-cluster-metrics-label={label}-reembed={reembed}" + basename)
    plt.close()

    for k in ks:
        if k != 0:
            sub_basename = f"-label={label}-subk={k}-reembed={reembed}" + basename
            sub_basetitle = f"Subcluster for {label}, subk={k}, reembed={reembed},"
            sub_basetitle += f" metric={metric}, k={k}, n_components={n_components}"

            ind = sub_results[sub_results["k"] == k][metric].idxmax()
            sub_model = sub_results.loc[ind, "model"]
            sub_left_model = sub_model
            sub_right_model = sub_model

            sub_pred_side = composite_predict(
                sub_X,
                sub_left_inds,
                sub_right_inds,
                sub_left_model,
                sub_right_model,
                relabel=True,
            )

            ax = stacked_barplot(
                sub_pred_side,
                sub_meta["merge_class"].values,
                color_dict=CLASS_COLOR_DICT,
                legend_ncol=6,
            )
            ax.set_title(sub_basetitle)
            stashfig(f"barplot" + sub_basename)
            plt.close()

            fig, ax = plot_cluster_pairs(
                sub_X,
                sub_left_inds,
                sub_right_inds,
                sub_left_model,
                sub_right_model,
                sub_meta["merge_class"].values,
            )
            fig.suptitle(sub_basetitle, y=1)
            stashfig(f"pairs" + sub_basename)
            plt.close()

            sub_adj = sub_data[i]["adj"]
            sub_meta["sub_pred_side"] = sub_pred_side

            sub_pred_var = f"c{label}_sub_pred_side"
            meta[sub_pred_var] = ""
            meta.loc[
                pred == label, sub_pred_var
            ] = sub_pred_side  # TODO indexing is dangerous here
            meta[f"c{label}_sub_pred"] = ""
            meta.loc[pred == label, f"c{label}_sub_pred"] = composite_predict(
                sub_X,
                sub_left_inds,
                sub_right_inds,
                sub_left_model,
                sub_right_model,
                relabel=False,
            )
            meta[f"is_c{label}"] = pred == label
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            adjplot(
                adj,
                ax=ax,
                meta=meta,
                sort_class=["pred_side", sub_pred_var],
                class_order="group_signal_flow",
                colors="merge_class",
                palette=CLASS_COLOR_DICT,
                item_order=["merge_class", "signal_flow"],
                highlight=f"is_c{label}",
                highlight_kws=dict(color="red", linestyle="-", linewidth=1),
                plot_type="scattermap",
                sizes=(0.5, 1),
            )
            fig.suptitle(sub_basetitle, y=0.94)
            stashfig("full-adj" + sub_basename)
            plt.close()
# %% [markdown]
# ##

cols = meta.columns
sub_pred_side_cols = []
sub_pred_cols = []
for c in cols:
    if "_sub_pred" in c:
        if "_side" in c:
            sub_pred_side_cols.append(c)
        else:
            sub_pred_cols.append(c)

meta["total_pred"] = ""
meta["total_pred"] = meta["pred"].astype(str) + "-"
meta["total_pred_side"] = ""
meta["total_pred_side"] = meta["pred_side"].astype(str) + "-"
meta["sub_pred"] = ""
meta["sub_pred_side"] = ""

for c in sub_pred_cols:
    meta["total_pred"] += meta[c].astype(str)
    meta["sub_pred"] += meta[c].astype(str)

for c in sub_pred_side_cols:
    meta["sub_pred_side"] += meta[c].astype(str)
    meta["total_pred_side"] += meta[c].astype(str)

# %% [markdown]
# ##
meta["lvl2_signal_flow"] = meta["total_pred"].map(
    meta.groupby("total_pred")["signal_flow"].mean()
)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    ax=ax,
    meta=meta,
    sort_class=["hemisphere", "pred", "sub_pred"],
    class_order="lvl2_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
)
fig.suptitle(f"2-level hierarchy clustering, reembed={reembed}" + basetitle, y=0.94)
stashfig("lvl2-full-adj" + sub_basename)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    ax=ax,
    meta=meta,
    sort_class=["hemisphere", "pred", "sub_pred"],
    class_order="lvl2_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["rand"],
    plot_type="scattermap",
    sizes=(0.5, 1),
)
fig.suptitle(f"2-level hierarchy clustering, reembed={reembed}" + basetitle, y=0.94)
stashfig("lvl2-full-adj-rand" + sub_basename)

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(15, 20))
ax = stacked_barplot(
    meta["total_pred_side"].values,
    meta["merge_class"].values,
    color_dict=CLASS_COLOR_DICT,
    legend_ncol=6,
    ax=ax,
    norm_bar_width=False,
)

stashfig("lvl2-barplot" + sub_basename)

# %% [markdown]
# ##
import pymaid
from src.pymaid import start_instance


start_instance()

for tp in meta["total_pred"].unique()[:10]:
    ids = list(meta[meta["total_pred"] == tp].index.values)
    ids = [int(i) for i in ids]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    skeleton_color_dict = dict(
        zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
    )
    pymaid.plot2d(ids, color=skeleton_color_dict, ax=ax)
    ax.axis("equal")
    stashfig(f"test-plot2d-{tp}")

# %% [markdown]
# ##


# %%
