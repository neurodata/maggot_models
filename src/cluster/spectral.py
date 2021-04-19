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
from anytree import LevelOrderGroupIter, NodeMixin, PostOrderIter
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.testing import ignore_warnings
from tqdm import tqdm

import pymaid
from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, selectSVD
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import rdpg
from graspy.utils import augment_diagonal, binarize, pass_to_ranks
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
    set_axes_equal,
    stacked_barplot,
)
from anytree.util import leftsibling


PLOT_MODELS = True


def get_paired_inds(meta):
    pair_meta = meta[meta["pair"].isin(meta.index)]
    pair_group_size = pair_meta.groupby("pair_id").size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta["pair_id"].isin(remove_pairs)]
    assert pair_meta.groupby("pair_id").size().min() == 2
    pair_meta.sort_values(["pair_id", "hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["hemisphere"] == "L"]["inds"]
    rp_inds = pair_meta[pair_meta["hemisphere"] == "R"]["inds"]
    assert (
        meta.iloc[lp_inds]["pair_id"].values == meta.iloc[rp_inds]["pair_id"].values
    ).all()
    return lp_inds, rp_inds


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
    both_embed = np.concatenate((left_embed, right_embed), axis=0)

    # print("Running left/right clustering with cross-validation\n")
    currtime = time.time()
    rows = []
    for k in range(min_clusters, max_clusters):
        # TODO add option for AutoGMM as well, might as well check
        for i in range(n_init):
            left_row, left_gc = fit_and_score(left_embed, right_embed, k)
            left_row["train"] = "left"
            right_row, right_gc = fit_and_score(right_embed, left_embed, k)
            right_row["train"] = "right"
            both_row, both_gc = fit_and_score(both_embed, both_embed, k)
            both_row["train"] = "both"

            # pairedness computation, if available
            if left_pair_inds is not None and right_pair_inds is not None:
                # TODO double check this is right
                # TODO this no longer makes sense with the new method
                pred_left = left_gc.predict(embed[left_pair_inds])
                pred_right = left_gc.predict(embed[right_pair_inds])
                pness, _, _ = compute_pairedness_bipartite(pred_left, pred_right)
                left_row["pairedness"] = pness
                ari = adjusted_rand_score(pred_left, pred_right)
                left_row["ARI"] = ari

                pred_left = right_gc.predict(embed[left_pair_inds])
                pred_right = right_gc.predict(embed[right_pair_inds])
                pness, _, _ = compute_pairedness_bipartite(pred_left, pred_right)
                right_row["pairedness"] = pness
                ari = adjusted_rand_score(pred_left, pred_right)
                right_row["ARI"] = ari

                pred_left = both_gc.predict(embed[left_pair_inds])
                pred_right = both_gc.predict(embed[right_pair_inds])
                pness, _, _ = compute_pairedness_bipartite(pred_left, pred_right)
                both_row["pairedness"] = pness
                ari = adjusted_rand_score(pred_left, pred_right)
                both_row["ARI"] = ari

            rows.append(left_row)
            rows.append(right_row)
            rows.append(both_row)

    results = pd.DataFrame(rows)
    # print(f"{time.time() - currtime} elapsed")
    return results


def make_ellipses(gmm, ax, i, j, colors, alpha=0.5, equal=False, **kws):
    inds = [j, i]
    for n, color in enumerate(colors):
        print(color)
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


def add_connections(x1, x2, y1, y2, color="black", alpha=0.2, linewidth=0.2, ax=None):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    if ax is None:
        ax = plt.gca()
    for i in range(len(x1)):
        ax.plot(
            [x1[i], x2[i]],
            [y1[i], y2[i]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def plot_cluster_pairs(
    X,
    left_inds,
    right_inds,
    model,
    labels,
    left_pair_inds=None,
    right_pair_inds=None,
    colors=None,
    equal=False,
):
    k = model.n_components
    n_dims = X.shape[1]

    print(k)

    if colors is None:
        colors = sns.color_palette("tab10", n_colors=k, desat=0.7)

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels  #
    pred = predict(X, left_inds, right_inds, model, relabel=False)
    data["pred"] = pred

    print(len(np.unique(pred)))

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
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=CLASS_COLOR_DICT,
                )
                make_ellipses(model, ax, i, j, colors, fill=False, equal=equal)
                if left_pair_inds is not None and right_pair_inds is not None:
                    add_connections(
                        data.iloc[left_pair_inds.values, j],
                        data.iloc[right_pair_inds.values, j],
                        data.iloc[left_pair_inds.values, i],
                        data.iloc[right_pair_inds.values, i],
                        ax=ax,
                    )

            if i > j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="pred",
                    palette=colors,
                )
                make_ellipses(model, ax, i, j, colors, fill=True, equal=equal)

    plt.tight_layout()
    return fig, axs


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


class MaggotCluster(NodeMixin):
    def __init__(
        self,
        name,
        root_inds=None,
        adj=None,
        meta=None,
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
        realign=False,
        plus_c=True,
        X=None,
        bic_ratio=None,
        min_split=10,
    ):
        super().__init__()
        self.name = name
        self.meta = meta.copy()
        if adj is not None:
            self.adj = adj.copy()
        self.parent = parent
        self.reembed = reembed
        self.meta["inds"] = range(len(self.meta))
        self.left_inds = self.meta[self.meta["left"]]["inds"]
        self.right_inds = self.meta[self.meta["right"]]["inds"]
        left_pair_inds, right_pair_inds = get_paired_inds(self.meta)
        self.left_pair_inds = left_pair_inds
        self.right_pair_inds = right_pair_inds
        self.n_init = n_init
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_components = n_components
        self.n_elbows = n_elbows
        self.normalize = normalize
        self.embed = embed
        self.regularizer = regularizer
        self.realign = realign
        self.plus_c = plus_c
        self.min_split = min_split
        if X is not None and reembed == False:
            self.X = X

        if root_inds is None:
            print("No `root_inds` were input; assuming this is the root.")
            root_inds = meta["inds"].copy()
        self.root_inds = root_inds
        self.stashfig = stashfig
        self.bic_ratio = bic_ratio

    def _stashfig(self, name):
        if self.stashfig is not None:
            basename = (
                f"-cluster={self.name}"
                + f"-embed={self.embed}"
                + f"-reembed={self.reembed}"
                + f"-normalize={self.normalize}"
                + f"-realign={self.realign}"
            )
            self.stashfig(name + basename)
            plt.close()

    def _embed(self, adj=None):
        if adj is None:
            adj = self.adj

        lp_inds = self.left_pair_inds
        rp_inds = self.right_pair_inds

        embed_adj = pass_to_ranks(adj)  # TODO PTR here?
        if self.plus_c:
            embed_adj += 1 / adj.size
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
            embed_adj = augment_diagonal(embed_adj)
            embed = selectSVD(
                embed_adj, n_components=self.n_components, n_elbows=self.n_elbows
            )
            embed = (embed[0], embed[2].T)

        X = np.concatenate(embed, axis=1)

        fraction_paired = (len(lp_inds) + len(rp_inds)) / len(self.root_inds)
        print(f"Learning transformation with {fraction_paired} neurons paired")

        X = self._procrustes(X)

        if self.normalize:
            row_norms = np.linalg.norm(X, axis=1)
            X /= row_norms[:, None]

        return X

    def _procrustes(self, X):
        X = X.copy()
        R, _ = orthogonal_procrustes(X[self.left_pair_inds], X[self.right_pair_inds])
        X[self.left_inds] = X[self.left_inds] @ R
        return X

    def fit_candidates(self, plot_all=True, show_plot=True):
        if hasattr(self, "k_") and self.k_ == 0:
            return

        root = self.root

        if hasattr(self, "X"):
            X = self.X
        elif self.reembed is True or self.is_root:
            X = self._embed()
        elif self.reembed is False:
            X = root.X_[self.root_inds]
            if self.realign:
                X = self._procrustes(X)
        elif self.reembed == "masked":
            mask = np.zeros(self.root.adj.shape, dtype=bool)
            mask[np.ix_(self.root_inds, self.root_inds)] = True
            masked_adj = np.zeros(mask.shape)
            masked_adj[mask] = self.root.adj[mask]
            X = self._embed(masked_adj)
            X = X[self.root_inds]

        if np.linalg.norm(X) < 1:
            X = X / np.linalg.norm(X)

        self.X_ = X

        results = crossval_cluster(
            X,
            self.left_inds,
            self.right_inds,
            left_pair_inds=self.left_pair_inds,
            right_pair_inds=self.right_pair_inds,
            max_clusters=self.max_clusters,
            min_clusters=self.min_clusters,
            n_init=self.n_init,
        )
        self.results_ = results

        if show_plot:
            fig, axs = plot_metrics(results, plot_all=plot_all)
            fig.suptitle(f"Clustering for cluster {self.name}, reembed={self.reembed}")
            self._stashfig("cluster-profile")

    def _plot_pairs(self, model=None, lines=True):
        if model is None:
            try:
                model = self.model_
            except AttributeError:
                raise ValueError("no model passed to _plot_pairs")
        k = model.n_components
        if lines:
            left_pair_inds = self.left_pair_inds
            right_pair_inds = self.right_pair_inds
        else:
            left_pair_inds = None
            right_pair_inds = None
        fig, ax = plot_cluster_pairs(
            self.X_,
            self.left_inds,
            self.right_inds,
            model,
            self.meta["merge_class"].values,
            left_pair_inds=left_pair_inds,
            right_pair_inds=right_pair_inds,
            equal=False,
        )
        fig.suptitle(f"{self.name}, k={k}", y=1)
        self._stashfig(f"pairs-k={k}")

    def _plot_bars(self, pred_side, k):
        ax = stacked_barplot(
            pred_side,
            self.meta["merge_class"],
            color_dict=CLASS_COLOR_DICT,
            legend_ncol=6,
            category_order=np.unique(pred_side),
        )
        # k = int(pred_side / 2)
        ax.set_title(f"{self.name}, k={k}")
        self._stashfig(f"bars-k={k}")

    def plot_model(self, k, metric="bic", lines=True):
        if not PLOT_MODELS:
            return
        if k > 0:
            model, pred, pred_side = self._model_predict(k, metric=metric)
            self._plot_bars(pred_side, k)
            self._plot_pairs(model, lines=lines)

    def _model_predict(self, k, metric="bic"):
        results = self.results_
        ind = results[results["k"] == k][metric].idxmax()
        model = results.loc[ind, "model"]
        pred = predict(self.X_, self.left_inds, self.right_inds, model, relabel=False)
        pred_side = predict(
            self.X_, self.left_inds, self.right_inds, model, relabel=True
        )
        return model, pred, pred_side

    def select_model(self, k=None, metric="bic"):
        if k is None and self.bic_ratio is not None:
            model1, _, _ = self._model_predict(1, metric=metric)
            model2, _, _ = self._model_predict(2, metric=metric)
            bic1 = -model1.bic(self.X_)
            bic2 = -model2.bic(self.X_)
            diff = bic2 - bic1  # NOTE: changed this, was mistakenly a ratio before
            ratio = bic2 / bic1
            if self.bic_ratio > 0:
                if ratio > self.bic_ratio:
                    k = 2
                else:
                    k = 0
            elif self.bic_ratio == 0:
                if diff > 0:
                    k = 2
                else:
                    k = 0

        self.k_ = k
        self.children = []
        if k > 0:
            model, pred, pred_side = self._model_predict(k, metric=metric)
            self.model_ = model
            self.pred_ = pred
            self.pred_side_ = pred_side
            root_meta = self.root.meta

            pred_name = f"{self.depth + 1}_pred"
            if pred_name not in root_meta.columns:
                root_meta[pred_name] = ""
            root_meta.loc[self.root_inds.index, pred_name] = pred.astype(str)
            pred_side_name = f"{self.depth + 1}_pred_side"
            if pred_side_name not in root_meta.columns:
                root_meta[pred_side_name] = ""
            root_meta.loc[self.root_inds.index, pred_side_name] = pred_side

            uni_labels = np.unique(pred).astype(str)

            self.children = []
            for i, ul in enumerate(uni_labels):
                new_meta = root_meta[
                    (root_meta[pred_name] == ul)
                    & (root_meta.index.isin(self.root_inds.index))
                ]
                new_root_inds = new_meta["inds"]
                new_name = self.name + "-" + str(ul)
                new_adj = self.root.adj[np.ix_(new_root_inds, new_root_inds)]
                if (
                    len(new_meta[new_meta["pair"].isin(new_meta.index)]) > 2
                    and len(new_meta) > self.min_split
                ):
                    MaggotCluster(
                        new_name,
                        root_inds=new_root_inds,
                        adj=new_adj,
                        meta=new_meta,
                        parent=self,
                        reembed=self.reembed,
                        n_init=self.n_init,
                        stashfig=self.stashfig,
                        max_clusters=self.max_clusters,
                        min_clusters=self.min_clusters,
                        n_components=self.n_components,
                        n_elbows=self.n_elbows,
                        regularizer=self.regularizer,
                        realign=self.realign,
                        normalize=self.normalize,
                        embed=self.embed,
                        plus_c=self.plus_c,
                        bic_ratio=self.bic_ratio,
                        min_split=self.min_split,
                    )

    def collect_labels(self):
        meta = self.root.meta
        meta["lvl0_labels"] = "0"
        meta["lvl0_labels_side"] = meta["lvl0_labels"] + meta["hemisphere"]
        for i in range(1, self.height + 1):
            meta[f"lvl{i}_labels"] = meta[f"lvl{i-1}_labels"] + "-" + meta[f"{i}_pred"]
            meta[f"lvl{i}_labels_side"] = meta[f"lvl{i}_labels"] + meta["hemisphere"]

    def get_lowest_level(self):
        level_it = LevelOrderGroupIter(self)
        last = next(level_it)
        nxt = last
        while nxt is not None:
            last = nxt
            nxt = next(level_it, None)
        return last


class BinaryCluster(MaggotCluster):
    def __init__(
        self,
        name,
        root_inds=None,
        adj=None,
        meta=None,
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
        realign=False,
        plus_c=True,
        X=None,
        bic_ratio=None,
        min_split=10,
    ):
        super().__init__(
            name,
            root_inds=root_inds,
            adj=adj,
            meta=meta,
            n_init=n_init,
            reembed=False,
            parent=parent,
            stashfig=stashfig,
            min_clusters=1,
            max_clusters=3,
            realign=False,
            X=X,
            bic_ratio=bic_ratio,
            min_split=min_split,
        )

    def build_linkage(self):
        # get a tuple of node at each level
        levels = []
        for group in LevelOrderGroupIter(self):
            levels.append(group)

        # just find how many nodes are leaves
        # this is necessary only because we need to add n to non-leaf clusters
        num_leaves = 0
        for node in PostOrderIter(self):
            if not node.children:
                num_leaves += 1

        link_count = 0
        node_index = 0
        linkages = []
        labels = []

        for g, group in enumerate(levels[::-1][:-1]):  # reversed and skip the last
            for i in range(len(group) // 2):
                # get partner nodes
                left_node = group[2 * i]
                right_node = group[2 * i + 1]
                # just double check that these are always partners
                assert leftsibling(right_node) == left_node

                # check if leaves, need to add some new fields to track for linkage
                if not left_node.children:
                    left_node._ind = node_index
                    left_node._n_clusters = 1
                    node_index += 1
                    labels.append(left_node.name)

                if not right_node.children:
                    right_node._ind = node_index
                    right_node._n_clusters = 1
                    node_index += 1
                    labels.append(right_node.name)

                # find the parent, count samples
                parent_node = left_node.parent
                n_clusters = left_node._n_clusters + right_node._n_clusters
                parent_node._n_clusters = n_clusters

                # assign an ind to this cluster for the dendrogram
                parent_node._ind = link_count + num_leaves
                link_count += 1

                distance = g + 1  # equal height for all links

                # add a row to the linkage matrix
                linkages.append([left_node._ind, right_node._ind, distance, n_clusters])

        labels = np.array(labels)
        linkages = np.array(linkages, dtype=np.double)  # needs to be a double for scipy
        return (linkages, labels)

    def fit(self, n_levels=10, metric="bic"):
        for i in range(n_levels):
            for j, node in enumerate(self.get_lowest_level()):
                node.fit_candidates(show_plot=False)
            for j, node in enumerate(self.get_lowest_level()):
                node.select_model(k=None, metric=metric)
            self.collect_labels()
