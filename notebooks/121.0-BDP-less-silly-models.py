# %% [markdown]
# # THE MIND OF A MAGGOT

# %% [markdown]
# ## Imports
import os
import time
import warnings

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, NodeMixin
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.testing import ignore_warnings
from tqdm import tqdm
from graspy.embed import LaplacianSpectralEmbed

import pymaid
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
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
    set_axes_equal,
    stacked_barplot,
)

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


def predict(X, left_inds, right_inds, model, relabel=False):
    # TODO add option to boost the right numbers
    X_left = X[left_inds]
    X_right = X[right_inds]
    pred_left = model.predict(X_left)
    pred_right = model.predict(X_right)
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


def fit_and_score(X_train, X_test, k, **kws):
    gc = GaussianCluster(
        min_components=k, max_components=k, covariance_type=["full", "diag"], **kws
    )
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
    X, left_inds, right_inds, model, labels, colors=None, equal=True
):
    k = model.n_components
    n_dims = X.shape[1]

    if colors is None:
        colors = sns.color_palette("tab10", n_colors=k, desat=0.7)

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels  #
    pred = predict(X, left_inds, right_inds, model, relabel=False)
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
                make_ellipses(model, ax, i, j, colors, fill=False, equal=equal)
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
                make_ellipses(model, ax, i, j, colors, fill=True, equal=equal)

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

        mean_results = results.groupby(["k"]).mean()
        mean_results.reset_index(inplace=True)
        sns.lineplot(
            data=mean_results, x="k", y=var, ax=ax, color="green", label="mean"
        )

        best_inds = results.groupby(["k"])[var].idxmax()
        best_results = results.loc[best_inds].copy()
        sns.lineplot(
            data=best_results, x="k", y=var, ax=ax, color="purple", label="max"
        )

        ymin = best_results[var].min()
        ymax = best_results[var].max()
        rng = ymax - ymin
        ymin = ymin - 0.1 * rng
        ymax = ymax + 0.02 * rng
        ax.set_ylim((ymin, ymax))

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


def level(adj, meta, pred, reembed=False, X=None, R=None, plot_all=True):
    uni_labels, inv = np.unique(pred, return_inverse=True)
    all_sub_results = []
    sub_data = []

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
            "reembed": reembed,
        }

        sub_data.append(var_dict)

        sub_results = crossval_cluster(
            sub_X,
            sub_left_inds,
            sub_right_inds,
            left_pair_inds=sub_lp_inds,
            right_pair_inds=sub_rp_inds,
            max_clusters=8,
            min_clusters=1,
            n_init=50,
        )

        fig, axs = plot_metrics(sub_results, plot_all=plot_all)
        fig.suptitle(f"Clustering for cluster {label}, reembed={reembed}")
        stashfig(f"cluster-profile-label={label}-reembed={reembed}")
        plt.close()
        all_sub_results.append(sub_results)
        return all_sub_results, sub_data


def plot_level(sub_results, sub_data, ks, label, metric="bic", basename=""):
    if isinstance(ks, int):
        ks = (ks,)

    sub_X = sub_data["X"]
    sub_left_inds = sub_data["left_inds"]
    sub_right_inds = sub_data["right_inds"]
    sub_lp_inds = sub_data["left_pair_inds"]
    sub_rp_inds = sub_data["right_pair_inds"]
    sub_meta = sub_data["meta"]
    reembed = sub_data["reembed"]

    fig, axs = plot_metrics(sub_results)
    fig.suptitle(f"Clustering for cluster {label}, reembed={reembed}")
    for ax in axs[:-1]:
        for k in ks:
            ax.axvline(k, linestyle="--", color="red", linewidth=2)
    stashfig(f"cluster-metrics-label={label}-reembed={reembed}" + basename)
    plt.close()

    for k in ks:
        if k != 0:
            sub_basename = f"-label={label}-subk={k}-reembed={reembed}" + basename
            sub_basetitle = f"Cluster for {label}, subk={k}, reembed={reembed},"
            sub_basetitle += (
                f" metric={metric}, k={k}"
            )  # , n_components={n_components}"

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

            # sub_adj = sub_data["adj"]
            # sub_meta["sub_pred_side"] = sub_pred_side

            # sub_pred_var = f"c{label}_sub_pred_side"
            # meta[sub_pred_var] = ""
            # meta.loc[
            #     pred == label, sub_pred_var
            # ] = sub_pred_side  # TODO indexing is dangerous here
            # meta[f"c{label}_sub_pred"] = ""
            # meta.loc[pred == label, f"c{label}_sub_pred"] = composite_predict(
            #     sub_X,
            #     sub_left_inds,
            #     sub_right_inds,
            #     sub_left_model,
            #     sub_right_model,
            #     relabel=False,
            # )
            # # meta[f"is_c{label}"] = pred == label
            # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            # adjplot(
            #     adj,
            #     ax=ax,
            #     meta=meta,
            #     sort_class=["pred_side", sub_pred_var],
            #     class_order="group_signal_flow",
            #     colors="merge_class",
            #     palette=CLASS_COLOR_DICT,
            #     item_order=["merge_class", "signal_flow"],
            #     # highlight=f"is_c{label}",
            #     # highlight_kws=dict(color="red", linestyle="-", linewidth=1),
            #     plot_type="scattermap",
            #     sizes=(0.5, 1),
            # )
            # fig.suptitle(sub_basetitle, y=0.94)
            # stashfig("full-adj" + sub_basename)
            # plt.close()


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
# ptr_adj = pass_to_ranks(adj)
meta["inds"] = range(len(meta))

left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]
lp_inds, rp_inds = get_paired_inds(meta)


# %% [markdown]
# ##

from graspy.embed import selectSVD
from graspy.utils import augment_diagonal


class MaggotCluster(NodeMixin):
    def __init__(
        self,
        name,
        root_inds=None,
        adj=None,
        meta=None,
        # X=None,
        n_init=50,
        reembed=False,
        parent=None,
        stashfig=None,
        min_clusters=1,
        max_clusters=15,
        n_components=None,
        n_elbows=2,
        normalize=False,
        embed="ase",
        regularizer=None,
    ):  # X=None, full_adj=None, full_meta=None):
        super(MaggotCluster, self).__init__()
        self.name = name
        self.meta = meta.copy()
        self.adj = adj.copy()
        self.parent = parent
        self.reembed = reembed
        # self.X = X
        self.meta["inds"] = range(len(self.meta))
        self.left_inds = self.meta[self.meta["left"]]["inds"]
        self.right_inds = self.meta[self.meta["right"]]["inds"]
        self.n_init = n_init
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_components = n_components
        self.n_elbows = n_elbows
        self.normalize = normalize
        self.embed = embed
        self.regularizer = regularizer

        if root_inds is None:
            print("No `root_inds` were input; assuming this is the root.")
            root_inds = meta["inds"].copy()
        self.root_inds = root_inds
        self.stashfig = stashfig

    def _stashfig(self, name):
        if self.stashfig is not None:
            basename = f"-cluster={self.name}-reembed={self.reembed}-normalize={self.normalize}"
            self.stashfig(name + basename)
            plt.close()

    def _embed(self, adj=None):
        if adj is None:
            adj = self.adj
        # TODO look into PTR at this level as well
        lp_inds, rp_inds = get_paired_inds(self.meta)

        embed_adj = pass_to_ranks(adj)
        if self.embed == "ase":
            embedder = AdjacencySpectralEmbed(
                n_components=self.n_components, n_elbows=self.n_elbows
            )
            embed = embedder.fit_transform(embed_adj)
        elif self.embed == "lse":
            embedder = LaplacianSpectralEmbed(
                n_components=self.n_components,
                n_elbows=self.n_elbows,
                regularizer=self.regularizer,
            )
            embed = embedder.fit_transform(embed_adj)
        elif self.embed == "unscaled_ase":
            embed_adj = pass_to_ranks(adj)
            embed_adj = augment_diagonal(embed_adj)
            embed = selectSVD(
                embed_adj, n_components=self.n_components, n_elbows=self.n_elbows
            )
            embed = (embed[0], embed[2].T)

        X = np.concatenate(embed, axis=1)

        fraction_paired = (len(lp_inds) + len(rp_inds)) / len(self.root_inds)
        print(f"Learning transformation with {fraction_paired} neurons paired")
        R, _ = orthogonal_procrustes(X[lp_inds], X[rp_inds])
        X[self.left_inds] = X[self.left_inds] @ R

        if self.normalize:
            row_sums = np.sum(X, axis=1)
            X /= row_sums[:, None]

        return X

    def fit_candidates(self, plot_all=True):  # mask):
        root = self.root
        meta = self.meta

        lp_inds, rp_inds = get_paired_inds(meta)

        if self.reembed is True or self.is_root:
            X = self._embed()
        elif self.reembed is False:
            X = root.X_[self.root_inds]
        elif self.reembed == "masked":
            mask = np.zeros(self.root.adj.shape, dtype=bool)
            mask[np.ix_(self.root_inds, self.root_inds)] = True
            masked_adj = np.zeros(mask.shape)
            masked_adj[mask] = self.root.adj[mask]
            X = self._embed(masked_adj)
            X = X[self.root_inds]

        self.X_ = X

        results = crossval_cluster(
            X,
            self.left_inds,
            self.right_inds,
            left_pair_inds=lp_inds,
            right_pair_inds=rp_inds,
            max_clusters=self.max_clusters,
            min_clusters=self.min_clusters,
            n_init=self.n_init,
        )
        self.results_ = results

        fig, axs = plot_metrics(results, plot_all=plot_all)
        fig.suptitle(f"Clustering for cluster {self.name}, reembed={self.reembed}")
        self._stashfig("cluster-profile")

    def _plot_pairs(self, model=None):
        if model is None:
            try:
                model = self.model_
            except AttributeError:
                raise ValueError("no model passed to _plot_pairs")
        k = model.n_components
        fig, ax = plot_cluster_pairs(
            self.X_,
            self.left_inds,
            self.right_inds,
            model,
            self.meta["merge_class"].values,
            equal=False,
        )
        fig.suptitle(f"{self.name}, k={k}", y=1)
        self._stashfig(f"pairs-k={k}")

    def _plot_bars(self, pred_side):
        ax = stacked_barplot(
            pred_side,
            self.meta["merge_class"],
            color_dict=CLASS_COLOR_DICT,
            legend_ncol=6,
            category_order=np.unique(pred_side),
        )
        k = int(len(np.unique(pred_side)) / 2)
        ax.set_title(f"{self.name}, k={k}")
        self._stashfig(f"bars-k={k}")

    def plot_model(self, k, metric="bic"):
        model, pred, pred_side = self._model_predict(k, metric=metric)
        self._plot_bars(pred_side)
        self._plot_pairs(model)

    def _model_predict(self, k, metric="bic"):
        results = self.results_
        ind = results[results["k"] == k][metric].idxmax()
        model = results.loc[ind, "model"]
        pred = predict(self.X_, self.left_inds, self.right_inds, model, relabel=False)
        pred_side = predict(
            self.X_, self.left_inds, self.right_inds, model, relabel=True
        )
        return model, pred, pred_side

    def select_model(self, k, metric="bic"):
        self.k_ = k
        self.children = []
        if k > 0:
            model, pred, pred_side = self._model_predict(k, metric=metric)
            self.model_ = model
            self.pred_ = pred
            self.pred_side_ = pred_side
            root_meta = self.root.meta
            root_meta.loc[self.root_inds.index, f"{self.name}_pred"] = pred
            root_meta.loc[self.root_inds.index, f"{self.name}_pred_side"] = pred_side

            uni_labels = np.unique(pred)

            self.children = []
            for i, ul in enumerate(uni_labels):
                new_meta = root_meta[root_meta[f"{self.name}_pred"] == ul]
                new_root_inds = new_meta["inds"]
                new_name = self.name + "-" + str(ul)
                new_adj = self.root.adj[np.ix_(new_root_inds, new_root_inds)]

                MaggotCluster(
                    new_name,
                    root_inds=new_root_inds,
                    adj=new_adj,
                    meta=new_meta,
                    reembed=self.reembed,
                    parent=self,
                    n_init=self.n_init,
                    stashfig=self.stashfig,
                    max_clusters=self.max_clusters,
                    min_clusters=self.min_clusters,
                    n_components=self.n_components,
                    n_elbows=self.n_elbows,
                )

    def plot_state(self):
        if self.k_ == 0:
            print("Nothing to plot here, k=0")
        else:
            self._plot_bars(self.pred_side_)
            self._plot_pairs(self.model_)


def get_lowest_level(node):
    level_it = LevelOrderGroupIter(node)
    last = next(level_it)
    nxt = last
    while nxt is not None:
        last = nxt
        nxt = next(level_it, None)

    return last


# Maybe tomorrow
# TODO look into my idea for partial reembeding
# TODO maybe look into my idea on subgraph + rest of graph embedding
# TODO would be cool to take the best fitting model and see how it compares in terms of
# signal flow and or cascades
# TODO look into doing DCSBM?
# TODO fix plots to only range based on the maxs
# TODO for masked embedding look into effect of d_hat
# TODO run on the subgraph instead of just masked
# TODO try running without spherical and tied

# Not tomorrow
# TODO seedless procrustes investigations


# %% [markdown]
# ##
np.random.seed(8888)
mc = MaggotCluster("0", adj=ptr_adj, meta=meta, n_init=50, stashfig=stashfig)
mc.fit_candidates()
mc.plot_model(6)
# mc.plot_model(7)  # TODO 7 might be better
mc.select_model(6)

# %% [markdown]
# ##
np.random.seed(9999)
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.fit_candidates()

# %% [markdown]
# ## Look at some models

# seems to peak very early, 2 or 3, all look tied
#
# cant read curve. mbons. 4 seems to subdivide well by class...
# 4 i guess. no intuition on these
# no ituition here either
# kcs


sub_ks = [(2, 3, 4), (2, 4), (2, 3, 4), (2, 3, 4), (3, 4), (0,)]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)

# %% [markdown]
# ## pick some models
sub_k = [3, 4, 2, 4, 3, 0]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.select_model(sub_k[i])

np.random.seed(9999)
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.fit_candidates()

# %% [markdown]
# ##
np.random.seed(1010)
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.fit_candidates()

# %% [markdown]
# ##

sub_ks = [
    (2, 6),
    (3, 4, 6),
    (2, 3),  # this one has a legit case, esp k=3
    (0,),
    (3,),
    (3, 4),
    (3,),
    (2, 3),
    (0,),
    (0,),
    (0,),
    (0,),
    (2, 3),
    (3,),
    (3,),
    (2,),
]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    for k in sub_ks[i]:
        if k != 0:
            node.plot_model(k)

# %% [markdown]
# ##

np.random.seed(8888)
mc = MaggotCluster(
    "0", adj=adj, meta=meta, n_init=50, stashfig=stashfig, reembed="masked"
)
mc.fit_candidates()
mc.plot_model(7)
# mc.plot_model(7)  # TODO 7 might be better
mc.select_model(7)


# %%
np.random.seed(9999)
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.fit_candidates()


# %%
# sub_ks = [(6,), (3, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 5)]
sub_kws = [(2, 3), (4, 6, 7), (2, 3, 4), (3, 4, 5), (2, 3, 4), (3, 4, 5), (4,)]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)


# %% [markdown]
# ## TRY SOME COMPARISONS


# %% [markdown]
# ##
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    # stashfig=stashfig,
    min_clusters=1,
    max_clusters=7,
)
mc.fit_candidates()
mc.plot_model(6)
# mc.plot_model(7)  # TODO 7 might be better

# %% [markdown]
# ##
mc.children = []
mc.select_model(6)

# %% [markdown]
# ##
np.random.seed(9999)
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.fit_candidates()

# %% [markdown]
# ## pick some models
sub_k = [3, 0, 5, 2, 2, 3]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.select_model(sub_k[i])
# %% [markdown]
# ##
mc.meta


# %%
label_meta = mc.meta.copy()
sub_cols = [f"0-{i}_pred_side" for i in range(6)]
sub_cols.remove("0-1_pred_side")
lvl_2_labels = label_meta[sub_cols].fillna("").sum(axis=1)
lvl_2_labels.name = "lvl2_pred_side"
label_meta = pd.concat((label_meta, lvl_2_labels), axis=1)


# %%
label_meta["rand"] = np.random.uniform(size=len(label_meta))
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=label_meta,
    sort_class=["0_pred_side", "lvl2_pred_side"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "rand"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("example-hierarchy-adj")
# %% [markdown]
# ##
sf = signal_flow(adj)
label_meta["signal_flow"] = -sf
label_meta["lvl2_signal_flow"] = label_meta["lvl2_pred_side"].map(
    label_meta.groupby("lvl2_pred_side")["signal_flow"].mean()
)
label_meta["lvl1_signal_flow"] = label_meta["0_pred_side"].map(
    label_meta.groupby("0_pred_side")["signal_flow"].mean()
)
# TODO fix for multilayer class_order
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=label_meta,
    sort_class=["0_pred_side", "lvl2_pred_side"],
    # class_order="lvl2_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("example-hierarchy-adj-sf")

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=label_meta,
    sort_class=["0_pred_side", "lvl2_pred_side"],
    # class_order="lvl2_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("example-hierarchy-adj-class-sf")

#%%

label_meta["te"] = -label_meta["Total edgesum"]
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=label_meta,
    sort_class=["0_pred_side", "lvl2_pred_side"],
    class_order="lvl1_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["te"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("example-hierarchy-adj-te")

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=label_meta,
    sort_class=["0_pred_side", "lvl2_pred_side"],
    class_order="lvl1_signal_flow",
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "te"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("example-hierarchy-adj-class-te")

# %% [markdown]
# ## Try with normalization
np.random.seed(8888)
mc = MaggotCluster(
    "0", adj=adj, meta=meta, n_init=50, stashfig=stashfig, normalize=True
)
mc.fit_candidates()


# %%
for k in range(3, 9):
    mc.plot_model(k)

# %% [markdown]
# ##


np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    normalize=False,
    n_elbows=2,
    max_clusters=10,
    reembed="masked",
)

mc.fit_candidates()
mc.plot_model(6)
mc.select_model(6)

for c in mc.children:
    c.n_elbows = 1

np.random.seed(9999)
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.fit_candidates()
# %% [markdown]
# ##
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    for k in range(2, 7):
        node.plot_model(k, metric="test_bic")


# %% [markdown]
# ## Try unscaled ASE
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    normalize=False,
    embed="unscaled_ase",
    regularizer=None,
    n_elbows=2,
    max_clusters=10,
    reembed="masked",
)

mc.fit_candidates()

# for k in range(3, 8):
#     mc.plot_model(k)


# %%

mc.select_model(7)
np.random.seed(9999)
print(mc.children)
print(len(mc.children))
print()
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.n_components = None
    node.n_elbows = 1
    node.fit_candidates()

# %% [markdown]
# ##
# ks = [(2,), (2, 3), (2, 3, 4), (2, 3, 4), (0,), (3, 4), (2, 4, 7)]
ks = [(2,), (2, 3, 5), (3, 4), (2, 3, 4), (0,), (2, 3, 4), (2, 3, 4, 6)]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    for k in ks[i]:
        if k > 0:
            node.plot_model(k, metric="bic")


# %%
[2, 2]

# %% [markdown]
# ##

meta[meta["class1"] == "bLN"]["Pair"]


# %%
