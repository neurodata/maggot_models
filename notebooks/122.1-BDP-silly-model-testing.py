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

PLOT_MODELS = False

np.random.seed(8888)

from src.io import savecsv


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name)


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
    equal=True,
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
        # lp_inds, rp_inds = get_paired_inds(self.meta)
        lp_inds = self.left_pair_inds
        rp_inds = self.right_pair_inds

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

    def plot_model(self, k, metric="bic", lines=True):
        if not PLOT_MODELS:
            return
        if k > 0:
            model, pred, pred_side = self._model_predict(k, metric=metric)
            self._plot_bars(pred_side)
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

    def select_model(self, k, metric="bic"):
        self.k_ = k
        self.children = []
        if k > 0:
            model, pred, pred_side = self._model_predict(k, metric=metric)
            self.model_ = model
            self.pred_ = pred
            self.pred_side_ = pred_side
            root_meta = self.root.meta

            pred_name = f"{self.depth}_pred"
            if pred_name not in root_meta.columns:
                root_meta[pred_name] = ""
            root_meta.loc[self.root_inds.index, pred_name] = pred.astype(str)
            pred_side_name = f"{self.depth}_pred_side"
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


# Maybe tomorrow
# TODO maybe look into my idea on subgraph + rest of graph embedding
# TODO would be cool to take the best fitting model and see how it compares in terms of
# signal flow and or cascades
# TODO look into doing DCSBM?
# TODO for masked embedding look into effect of d_hat
# TODO run on the subgraph instead of just masked
# TODO draw lines for pairs

# Not tomorrow
# TODO seedless procrustes investigations


# %% [markdown]
# ##
np.random.seed(8888)
mc = MaggotCluster(
    "0",
    adj=adj,
    meta=meta,
    n_init=50,
    stashfig=stashfig,
    max_clusters=8,
    n_components=4,
    embed="ase",
)
mc.fit_candidates()
mc.plot_model(6)
# mc.plot_model(7)  # TODO 7 might be better
# mc.select_model(6)

# %% [markdown]
# ##
mc.select_model(6)

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

# TODO look into whether to "rematch" here by learning a new R matrix

sub_ks = [(2, 3, 4), (0,), (2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4)]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)

# %% [markdown]
# ## pick some models
# sub_k = [3, 4, 2, 4, 3, 0]
sub_k = [3, 0, 2, 2, 2, 3]
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
sub_ks = [
    (2, 3, 4),
    (0,),
    (2, 3),
    (3, 4),  # or 0,
    (2,),  # probably 0
    (2,),
    (2,),
    (2, 6),  # probably 0,
    (2,),  # probably 0,
    (2, 3),  # maybe 0
    (2, 3, 4),
    (2, 3, 4),  # probably 0,
]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    for k in sub_ks[i]:
        node.plot_model(k)
# %% [markdown]
# ##
sub_k = [2, 0, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0]
for i, node in enumerate(get_lowest_level(mc)):
    print(node.name)
    print()
    node.select_model(sub_k[i])

# %% [markdown]
# ##

meta = mc.meta.copy()
meta["rand"] = np.random.uniform(size=len(meta))
sf = signal_flow(adj)
meta["signal_flow"] = -sf
meta["te"] = -meta["Total edgesum"]
# %% [markdown]
# ## plot by class and randomly within class
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=meta,
    sort_class=["0_pred_side"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "rand"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("adj-lvl0")
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=meta,
    sort_class=["0_pred_side", "1_pred_side"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "rand"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("adj-lvl1")
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=meta,
    sort_class=["0_pred_side", "1_pred_side", "2_pred_side"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    item_order=["merge_class", "rand"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("adj-lvl2")


# %% [markdown]
# ## plot by class and signal flow within class
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=meta,
    sort_class=["0_pred_side"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["merge_class", "signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("adj-lvl0-sf-class")
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=meta,
    sort_class=["0_pred_side", "1_pred_side"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["merge_class", "signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("adj-lvl1-sf-class")
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    adj,
    meta=meta,
    sort_class=["0_pred_side", "1_pred_side", "2_pred_side"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["merge_class", "signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
    ax=ax,
)
stashfig("adj-lvl2-sf-class")
# %% [markdown]
# ##
meta["lvl0_labels"] = meta["0_pred"] + meta["hemisphere"]
meta["lvl1_labels"] = meta["0_pred"] + "-" + meta["1_pred"] + meta["hemisphere"]
meta["lvl2_labels"] = (
    meta["0_pred"] + "-" + meta["1_pred"] + "-" + meta["2_pred"] + meta["hemisphere"]
)

# %% [markdown]
# ## plot by random within a group

from graspy.models import SBMEstimator, DCSBMEstimator

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[1]
adjplot(
    adj,
    meta=meta,
    sort_class=["hemisphere", "lvl0_labels"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["te"],
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    ax=ax,
    ticks=False,
)

estimator = DCSBMEstimator(degree_directed=True, directed=True, loops=False)
estimator.fit(adj, meta["lvl0_labels"].values)
sample = np.squeeze(estimator.sample())
ax = axs[0]
adjplot(
    sample,
    meta=meta,
    sort_class=["hemisphere", "lvl0_labels"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["te"],
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    ax=ax,
    ticks=False,
)

stashfig("adj-lvl0-rand-te-hemi")
###
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
adjplot(
    adj,
    meta=meta,
    sort_class=["hemisphere", "lvl1_labels"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["te"],
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    ax=ax,
    ticks=False,
)

estimator = DCSBMEstimator(degree_directed=True, directed=True, loops=False)
estimator.fit(adj, meta["lvl1_labels"].values)
sample = np.squeeze(estimator.sample())
ax = axs[1]
adjplot(
    sample,
    meta=meta,
    sort_class=["hemisphere", "lvl1_labels"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["te"],
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    ax=ax,
    ticks=False,
)

stashfig("adj-lvl1-rand-te-hemi")
###
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[1]
adjplot(
    adj,
    meta=meta,
    sort_class=["hemisphere", "lvl2_labels"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["te"],
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    ax=ax,
    ticks=False,
)

estimator = DCSBMEstimator(degree_directed=True, directed=True, loops=False)
estimator.fit(adj, meta["lvl2_labels"].values)
sample = np.squeeze(estimator.sample())
ax = axs[0]
adjplot(
    sample,
    meta=meta,
    sort_class=["hemisphere", "lvl2_labels"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    class_order=["signal_flow"],
    item_order=["te"],
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    ax=ax,
    ticks=False,
)

stashfig("adj-lvl2-rand-te-hemi")
# %% [markdown]
# ##
from src.graph import MetaGraph


def upper_triu_prop(adj):
    inds = np.triu_indices_from(adj, k=1)
    prop = adj[inds].sum() / adj.sum()
    return prop


rows = []
n_samples = 5
lvls = ["lvl0_labels", "lvl1_labels", "lvl2_labels"]
for lvl in lvls:
    estimator = DCSBMEstimator(degree_directed=True, directed=True, loops=False)
    estimator.fit(adj, meta[lvl].values)
    for i in range(n_samples):
        sample = np.squeeze(estimator.sample())
        sample_meta = meta.copy()
        sf = signal_flow(sample)
        sample_meta["signal_flow"] = -sf
        sample_mg = MetaGraph(sample, sample_meta)
        sample_mg = sample_mg.sort_values("signal_flow", ascending=True)
        prop = upper_triu_prop(sample_mg.adj)
        print(prop)
        row = {"level": lvl.replace("_labels", ""), "prop": prop}
        rows.append(row)
    print()

bin_meta = meta.copy()
bin_adj = binarize(adj)
sf = signal_flow(bin_adj)
bin_meta["signal_flow"] = -sf
bin_mg = MetaGraph(bin_adj, bin_meta)
bin_mb = bin_mg.sort_values("signal_flow", ascending=True)
prop = upper_triu_prop(bin_mg.adj)
print(prop)

rows.append({"level": "data", "prop": prop})
prop_df = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.stripplot(data=prop_df, x="level", y="prop", ax=ax)
ax.set_ylabel("Prop. in upper triangle")
ax.set_xlabel("Model")
stashfig("ffwdness-by-model")

# %% [markdown]
# ## make barplots

from src.visualization import barplot_text


lvls = ["lvl0_labels", "lvl1_labels", "lvl2_labels"]
for lvl in lvls:
    pred_labels = meta[lvl]
    true_labels = meta["merge_class"].values
    fig, ax = plt.subplots(1, 1, figsize=(15, 20))
    stacked_barplot(
        pred_labels,
        true_labels,
        category_order=np.unique(pred_labels),
        color_dict=CLASS_COLOR_DICT,
        ax=ax,
    )
    stashfig(f"barplot-no-text-lvl-{lvl}", dpi=200)

# %% [markdown]
# ##
meta["lvl0_labels"] = meta["0_pred"]
meta["lvl1_labels"] = meta["0_pred"] + "-" + meta["1_pred"]
meta["lvl2_labels"] = meta["0_pred"] + "-" + meta["1_pred"] + "-" + meta["2_pred"]
meta["lvl0_labels_side"] = meta["lvl0_labels"] + meta["hemisphere"]
meta["lvl1_labels_side"] = meta["lvl1_labels"] + meta["hemisphere"]
meta["lvl2_labels_side"] = meta["lvl2_labels"] + meta["hemisphere"]


stashcsv(meta, "stash-label-meta")
# %% [markdown]
# ##
load = True
loc = f"maggot_models/notebooks/outs/{FNAME}/csvs/stash-label-meta.csv"
if load:
    meta = pd.read_csv(loc, index_col=0)


# %% [markdown]
# ##


start_instance()
# labels = meta["lvl2_labels"].values

for tp in meta["lvl2_labels"].unique():
    ids = list(meta[meta["lvl2_labels"] == tp].index.values)
    ids = [int(i) for i in ids]
    fig = plt.figure(figsize=(30, 10))

    gs = plt.GridSpec(2, 3, figure=fig, wspace=0, hspace=0, height_ratios=[0.8, 0.2])

    skeleton_color_dict = dict(
        zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
    )

    # ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax = fig.add_subplot(gs[0, 0], projection="3d")

    pymaid.plot2d(
        ids,
        color=skeleton_color_dict,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=True,
    )
    ax.azim = -90  # 0 for side view
    ax.elev = 0
    ax.dist = 6
    set_axes_equal(ax)

    # ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    pymaid.plot2d(
        ids,
        color=skeleton_color_dict,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=True,
    )
    ax.azim = 0  # 0 for side view
    ax.elev = 0
    ax.dist = 6
    set_axes_equal(ax)

    # ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    pymaid.plot2d(
        ids,
        color=skeleton_color_dict,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=True,
    )
    ax.azim = -90
    ax.elev = 90
    ax.dist = 6
    set_axes_equal(ax)

    ax = fig.add_subplot(gs[1, :])
    temp_meta = meta[meta["lvl2_labels"] == tp]
    cat = temp_meta["lvl2_labels_side"].values
    subcat = temp_meta["merge_class"].values
    stacked_barplot(cat, subcat, ax=ax, color_dict=CLASS_COLOR_DICT)
    ax.get_legend().remove()

    fig.suptitle(tp)

    stashfig(f"plot3d-{tp}")
    plt.close()


# %%
