# %% [markdown]
# ##
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
)

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


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def get_paired_inds(meta):
    # pair_meta = meta[meta["Pair"] != -1].copy()
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


def crossval_cluster(
    embed,
    left_inds,
    right_inds,
    R,
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
        # train left, test right
        # TODO add option for AutoGMM as well, might as well check
        left_gc = GaussianCluster(min_components=k, max_components=k, n_init=n_init)
        left_gc.fit(left_embed)
        model = left_gc.model_
        train_left_bic = model.bic(left_embed)
        train_left_lik = model.score(left_embed)
        test_left_bic = model.bic(right_embed @ R.T)
        test_left_lik = model.score(right_embed @ R.T)

        # train right, test left
        right_gc = GaussianCluster(min_components=k, max_components=k, n_init=n_init)
        right_gc.fit(right_embed)
        model = right_gc.model_
        train_right_bic = model.bic(right_embed)
        train_right_lik = model.score(right_embed)
        test_right_bic = model.bic(left_embed @ R)
        test_right_lik = model.score(left_embed @ R)

        left_row = {
            "k": k,
            "contra_bic": -test_left_bic,
            "contra_lik": test_left_lik,
            "ipsi_bic": -train_left_bic,
            "ipsi_lik": train_left_lik,
            "cluster": left_gc,
            "train": "left",
            "n_components": n_components,
        }
        right_row = {
            "k": k,
            "contra_bic": -test_right_bic,
            "contra_lik": test_right_lik,
            "ipsi_bic": -train_right_bic,
            "ipsi_lik": train_right_lik,
            "cluster": right_gc,
            "train": "right",
            "n_components": n_components,
        }

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
    sns.lineplot(data=results, x="k", y="contra_lik", hue="train", ax=ax, legend=False)
    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")
    sns.lineplot(data=results, x="k", y="ipsi_lik", hue="train", ax=ax, legend=False)
    ax.set_ylabel("Log likelihood")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))

    ax = axs[1]
    sns.lineplot(data=results, x="k", y="contra_bic", hue="train", ax=ax, legend="full")
    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")
    sns.lineplot(data=results, x="k", y="ipsi_bic", hue="train", ax=ax, legend="full")
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
    X,
    left_inds,
    right_inds,
    left_model,
    right_model,
    labels,
    left_colors=None,
    right_colors=None,
    equal=True,
):
    k = left_model.n_components
    n_dims = X.shape[1]

    if left_colors is None and right_colors is None:
        tab20 = sns.color_palette("tab20", n_colors=2 * k, desat=0.7)
        left_colors = tab20[::2]
        right_colors = tab20[1::2]

    colors = left_colors + right_colors

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels  #
    # TODO fill this in with composite_predict here
    pred_left = left_model.predict(X[left_inds])
    pred_right = right_model.predict(X[right_inds]) + pred_left.max() + 1
    pred = np.empty(len(data), dtype=int)
    pred[left_inds] = pred_left
    pred[right_inds] = pred_right
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
                make_ellipses(
                    left_model, ax, i, j, left_colors, fill=False, equal=equal
                )
                make_ellipses(
                    right_model, ax, i, j, right_colors, fill=False, equal=equal
                )
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
                make_ellipses(left_model, ax, i, j, left_colors, fill=True, equal=equal)
                make_ellipses(
                    right_model, ax, i, j, right_colors, fill=True, equal=equal
                )

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


# %% [markdown]
# ## Load data


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
meta = mg.meta

adj = mg.adj
adj = pass_to_ranks(adj)
meta["inds"] = range(len(meta))

left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]
lp_inds, rp_inds = get_paired_inds(meta)

# %% [markdown]
# ## Embed
ase = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
embed = ase.fit_transform(adj)
n_components = embed[0].shape[1]  # use all of ZG2
X = np.concatenate((embed[0][:, :n_components], embed[1][:, :n_components]), axis=-1)
R, _ = orthogonal_procrustes(X[lp_inds], X[rp_inds])

# %% [markdown]
# ## Cluster, doing train/test on left/right
results = crossval_cluster(
    X, left_inds, right_inds, R, left_pair_inds=lp_inds, right_pair_inds=rp_inds
)

plot_crossval_cluster(results)
stashfig(f"cross-val-n_components={n_components}")

# %% [markdown]
# ## Choose a k
k = 6

# %% [markdown]
# ## Take the best models and match them left/right

models = results[results["k"] == k]["cluster"].values
left_model = models[0].model_
right_model = models[1].model_

pred = composite_predict(X, left_inds, right_inds, left_model, right_model)  # TODO
_, row_inds, col_inds = compute_pairedness_bipartite(pred[lp_inds], pred[rp_inds])

left_model = reindex_model(left_model, row_inds)
right_model = reindex_model(right_model, col_inds)

pred = composite_predict(
    X, left_inds, right_inds, left_model, right_model, relabel=True
)

stacked_barplot(
    pred, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT, legend_ncol=4
)

stashfig(f"gmm-crossval-barplot-k={k}-n_components={n_components}")

# %% [markdown]
# ## Pairplot with gaussian blobs for the clusters

plot_cluster_pairs(
    X, left_inds, right_inds, left_model, right_model, meta["merge_class"].values
)

stashfig(f"gmm-crossval-pairs-k={k}-n_components={n_components}")
stashfig(f"gmm-crossval-pairs-k={k}-n_components={n_components}", fmt="pdf")

# %% [markdown]
# ## plot the adjacency matrix


sf = signal_flow(adj)
meta["signal_flow"] = -sf
meta["pred"] = pred
meta["group_signal_flow"] = meta["pred"].map(meta.groupby("pred")["signal_flow"].mean())

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
matrixplot(
    adj,
    ax=ax,
    row_meta=meta,
    col_meta=meta,
    row_sort_class="pred",
    col_sort_class="pred",
    row_class_order="group_signal_flow",
    col_class_order="group_signal_flow",
    row_colors="merge_class",
    col_colors="merge_class",
    row_palette=CLASS_COLOR_DICT,
    col_palette=CLASS_COLOR_DICT,
    row_item_order=["merge_class", "signal_flow"],
    col_item_order=["merge_class", "signal_flow"],
    plot_type="scattermap",
    sizes=(0.5, 1),
)
stashfig(f"adj-k={k}-n_components={n_components}")

# %% [markdown]
# ## SUBCLUSTER with reembedding!

pred = composite_predict(
    X, left_inds, right_inds, left_model, right_model, relabel=False
)
uni_labels, inv = np.unique(pred, return_inverse=True)
all_sub_results = []
sub_data = []

reembed = True

for label in uni_labels:
    print(label)
    print()
    label_mask = inv == label
    sub_meta = meta[label_mask].copy()
    sub_meta["inds"] = range(len(sub_meta))
    sub_left_inds = sub_meta[sub_meta["left"]]["inds"].values
    sub_right_inds = sub_meta[sub_meta["right"]]["inds"].values
    sub_lp_inds, sub_rp_inds = get_paired_inds(sub_meta)

    if reembed:
        ase = AdjacencySpectralEmbed()
        # TODO look into PTR at this level as well
        sub_adj = adj[np.ix_(label_mask, label_mask)]
        sub_embed = ase.fit_transform(sub_adj)
        sub_X = np.concatenate(sub_embed, axis=1)
        sub_R, _ = orthogonal_procrustes(sub_X[sub_lp_inds], sub_X[sub_rp_inds])
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
    }
    if reembed:
        var_dict["adj"] = sub_adj
    sub_data.append(var_dict)

    sub_results = crossval_cluster(
        sub_X,
        sub_left_inds,
        sub_right_inds,
        sub_R,
        left_pair_inds=sub_lp_inds,
        right_pair_inds=sub_rp_inds,
        max_clusters=10,
        min_clusters=1,
    )

    fig, axs = plot_crossval_cluster(sub_results)
    fig.suptitle(f"Subclustering for cluster {label}, reembed={reembed}")
    stashfig(f"sub-cluster-profile-label={label}-reembed={reembed}")
    all_sub_results.append(sub_results)

# %% [markdown]
# ##


sub_ks = [(2, 4), (6,), (2, 5), (2, 5), (2, 4), (2, 3, 5)]

for i, label in enumerate(uni_labels):
    ks = sub_ks[i]
    sub_results = all_sub_results[i]
    sub_X = sub_data[i]["X"]
    sub_left_inds = sub_data[i]["left_inds"]
    sub_right_inds = sub_data[i]["right_inds"]
    sub_lp_inds = sub_data[i]["left_pair_inds"]
    sub_rp_inds = sub_data[i]["right_pair_inds"]
    sub_meta = sub_data[i]["meta"]

    fig, axs = plot_crossval_cluster(sub_results)
    fig.suptitle(f"Subclustering for cluster {label}, reembed={reembed}")
    for ax in axs:
        for k in ks:
            ax.axvline(k, linestyle="--", color="red", linewidth=2)
    stashfig(f"sub-cluster-profile-label={label}-k={k}-reembed={reembed}")
    plt.close()

    for k in ks:
        models = sub_results[sub_results["k"] == k]["cluster"].values
        sub_left_model = models[0].model_
        sub_right_model = models[1].model_

        sub_pred = composite_predict(
            sub_X, sub_left_inds, sub_right_inds, sub_left_model, sub_right_model
        )
        _, row_inds, col_inds = compute_pairedness_bipartite(
            sub_pred[sub_lp_inds], sub_pred[sub_rp_inds]
        )

        # there is a weird edge case for KCs
        if len(row_inds) < len(np.unique(sub_pred)):
            row_inds = np.unique(sub_pred)
            col_inds = np.unique(sub_pred)

        sub_left_model = reindex_model(sub_left_model, row_inds)
        sub_right_model = reindex_model(sub_right_model, col_inds)

        fig, axs = plot_cluster_pairs(
            sub_X,
            sub_left_inds,
            sub_right_inds,
            sub_left_model,
            sub_right_model,
            sub_meta["merge_class"].values,
        )
        fig.suptitle(f"Subclustering for cluster {label}, sub-K={k}, reembed={reembed}")
        stashfig(f"subcluster-pairs-label={label}-k={k}-reembed={reembed}")
        plt.close()

        sub_pred = composite_predict(
            sub_X,
            sub_left_inds,
            sub_right_inds,
            sub_left_model,
            sub_right_model,
            relabel=True,
        )

        ax = stacked_barplot(
            sub_pred,
            sub_meta["merge_class"].values,
            color_dict=CLASS_COLOR_DICT,
            legend_ncol=4,
        )
        ax.set_title(f"Subclusters for cluster {label}, sub-K={k}, reembed={reembed}")
        stashfig(f"subcluster-barplot-label={label}-k={k}-reembed={reembed}")
        plt.close()

        if "adj" in sub_data[i]:
            sub_adj = sub_data[i]["adj"]
            sub_meta["sub_pred"] = sub_pred
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            matrixplot(
                sub_adj,
                ax=ax,
                row_meta=sub_meta,
                col_meta=sub_meta,
                row_sort_class=["hemisphere", "sub_pred"],
                col_sort_class=["hemisphere", "sub_pred"],
                row_class_order=None,
                col_class_order=None,
                row_colors="merge_class",
                col_colors="merge_class",
                row_palette=CLASS_COLOR_DICT,
                col_palette=CLASS_COLOR_DICT,
                row_item_order="merge_class",
                col_item_order="merge_class",
                plot_type="scattermap",
                sizes=(5, 7),
            )
            stashfig(f"subcluster-adj-label={label}-k={k}-reembed={reembed}")
            plt.close()

# %% [markdown]
# ## SUBCLUSTER without reembedding

all_sub_results = []
sub_data = []

reembed = False

for label in uni_labels:
    print(label)
    print()
    label_mask = inv == label
    sub_meta = meta[label_mask].copy()
    sub_meta["inds"] = range(len(sub_meta))
    sub_left_inds = sub_meta[sub_meta["left"]]["inds"].values
    sub_right_inds = sub_meta[sub_meta["right"]]["inds"].values
    sub_lp_inds, sub_rp_inds = get_paired_inds(sub_meta)

    if reembed:
        ase = AdjacencySpectralEmbed()
        # TODO look into PTR at this level as well
        sub_adj = adj[np.ix_(label_mask, label_mask)]
        sub_embed = ase.fit_transform(sub_adj)
        sub_X = np.concatenate(sub_embed, axis=1)
        sub_R, _ = orthogonal_procrustes(sub_X[sub_lp_inds], sub_X[sub_rp_inds])
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
    }
    if reembed:
        var_dict["adj"] = sub_adj
    sub_data.append(var_dict)

    sub_results = crossval_cluster(
        sub_X,
        sub_left_inds,
        sub_right_inds,
        sub_R,
        left_pair_inds=sub_lp_inds,
        right_pair_inds=sub_rp_inds,
        max_clusters=10,
        min_clusters=1,
    )
    fig, axs = plot_crossval_cluster(sub_results)
    fig.suptitle(f"Subclustering for cluster {label}, reembed={reembed}")
    stashfig(f"sub-cluster-profile-label={label}-reembed={reembed}")
    all_sub_results.append(sub_results)

# %% [markdown]
# ##

sub_ks = [(2, 4, 5), (3,), (2, 4, 5), (2, 3, 4), (2, 3, 7), (3, 6)]

for i, label in enumerate(uni_labels):
    ks = sub_ks[i]
    sub_results = all_sub_results[i]
    sub_X = sub_data[i]["X"]
    sub_left_inds = sub_data[i]["left_inds"]
    sub_right_inds = sub_data[i]["right_inds"]
    sub_lp_inds = sub_data[i]["left_pair_inds"]
    sub_rp_inds = sub_data[i]["right_pair_inds"]
    sub_meta = sub_data[i]["meta"]

    fig, axs = plot_crossval_cluster(sub_results)
    fig.suptitle(f"Subclustering for cluster {label}, reembed={reembed}")
    for ax in axs:
        for k in ks:
            ax.axvline(k, linestyle="--", color="red", linewidth=2)
    stashfig(f"sub-cluster-profile-label={label}-k={k}-reembed={reembed}")
    plt.close()

    for k in ks:
        models = sub_results[sub_results["k"] == k]["cluster"].values
        sub_left_model = models[0].model_
        sub_right_model = models[1].model_

        sub_pred = composite_predict(
            sub_X, sub_left_inds, sub_right_inds, sub_left_model, sub_right_model
        )
        _, row_inds, col_inds = compute_pairedness_bipartite(
            sub_pred[sub_lp_inds], sub_pred[sub_rp_inds]
        )

        # there is a weird edge case for KCs
        if len(row_inds) < len(np.unique(sub_pred)):
            row_inds = np.unique(sub_pred)
            col_inds = np.unique(sub_pred)

        sub_left_model = reindex_model(sub_left_model, row_inds)
        sub_right_model = reindex_model(sub_right_model, col_inds)

        fig, axs = plot_cluster_pairs(
            sub_X,
            sub_left_inds,
            sub_right_inds,
            sub_left_model,
            sub_right_model,
            sub_meta["merge_class"].values,
        )
        fig.suptitle(f"Subclustering for cluster {label}, sub-K={k}, reembed={reembed}")
        stashfig(f"subcluster-pairs-label={label}-k={k}-reembed={reembed}")
        plt.close()

        sub_pred = composite_predict(
            sub_X,
            sub_left_inds,
            sub_right_inds,
            sub_left_model,
            sub_right_model,
            relabel=True,
        )

        ax = stacked_barplot(
            sub_pred,
            sub_meta["merge_class"].values,
            color_dict=CLASS_COLOR_DICT,
            legend_ncol=4,
        )
        ax.set_title(f"Subclusters for cluster {label}, sub-K={k}, reembed={reembed}")
        stashfig(f"subcluster-barplot-label={label}-k={k}-reembed={reembed}")
        plt.close()

        if "adj" in sub_data[i]:
            sub_adj = sub_data[i]["adj"]
            sub_meta["sub_pred"] = sub_pred
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            matrixplot(
                sub_adj,
                ax=ax,
                row_meta=sub_meta,
                col_meta=sub_meta,
                row_sort_class=["hemisphere", "sub_pred"],
                col_sort_class=["hemisphere", "sub_pred"],
                row_class_order=None,
                col_class_order=None,
                row_colors="merge_class",
                col_colors="merge_class",
                row_palette=CLASS_COLOR_DICT,
                col_palette=CLASS_COLOR_DICT,
                row_item_order="merge_class",
                col_item_order="merge_class",
                plot_type="scattermap",
                sizes=(5, 7),
            )
            stashfig(f"subcluster-adj-label={label}-k={k}-reembed={reembed}")
            plt.close()
