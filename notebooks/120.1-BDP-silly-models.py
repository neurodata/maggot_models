# %% [markdown]
# ##
import os
import time

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import orthogonal_procrustes
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
from src.visualization import CLASS_COLOR_DICT, gridmap, matrixplot

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

sns.set_context("talk", font_scale=1.5)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, dpi=300, **kws)


def get_paired_inds(meta):
    pair_meta = meta[meta["Pair"] != -1].copy()
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


# %% [markdown]
# ## Load
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

degrees = mg.calculate_degrees()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.distplot(np.log10(degrees["Total edgesum"]), ax=ax)
q = np.quantile(degrees["Total edgesum"], 0.05)
ax.axvline(np.log10(q), linestyle="--", color="r")
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


# %% [markdown]
# ## Cluster, doing train/test on left/right

currtime = time.time()

n_init = 25

rows = []
n_components = embed[0].shape[1]
print(n_components)
print()
train_embed = np.concatenate(
    (embed[0][:, :n_components], embed[1][:, :n_components]), axis=-1
)
R, _ = orthogonal_procrustes(train_embed[lp_inds], train_embed[rp_inds])


# %% [markdown]
# ##

left_embed = train_embed[left_inds]
right_embed = train_embed[right_inds]

for k in tqdm(range(2, 15)):
    # train left, test right
    # TODO add option for AutoGMM as well, might as well check
    left_gc = GaussianCluster(min_components=k, max_components=k, n_init=n_init)
    left_gc.fit(left_embed)
    model = left_gc.model_
    train_left_bic = model.bic(left_embed)
    train_left_lik = model.score(left_embed)
    test_left_bic = model.bic(right_embed @ R.T)
    test_left_lik = model.score(right_embed @ R.T)

    row = {
        "k": k,
        "contra_bic": test_left_bic,
        "contra_lik": test_left_lik,
        "ipsi_bic": train_left_bic,
        "ipsi_lik": train_left_lik,
        "cluster": left_gc,
        "train": "left",
        "n_components": n_components,
    }
    rows.append(row)

    # train right, test left
    right_gc = GaussianCluster(min_components=k, max_components=k, n_init=n_init)
    right_gc.fit(right_embed)
    model = right_gc.model_
    train_right_bic = model.bic(right_embed)
    train_right_lik = model.score(right_embed)
    test_right_bic = model.bic(left_embed @ R)
    test_right_lik = model.score(left_embed @ R)

    row = {
        "k": k,
        "contra_bic": test_right_bic,
        "contra_lik": test_right_lik,
        "ipsi_bic": train_right_bic,
        "ipsi_lik": train_right_lik,
        "cluster": right_gc,
        "train": "right",
        "n_components": n_components,
    }
    rows.append(row)

print(f"{time.time() - currtime} elapsed")

results = pd.DataFrame(rows)
n_components = n_components
small_results = results[results["n_components"] == n_components]
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax = axs[0]
sns.lineplot(data=small_results, x="k", y="contra_bic", hue="train", ax=ax)
ax.lines[0].set_linestyle("--")
ax.lines[1].set_linestyle("--")
sns.lineplot(data=small_results, x="k", y="ipsi_bic", hue="train", ax=ax)
ax.set_ylabel("BIC")
leg = ax.get_legend().remove()
ax.set_title(f"n_components={n_components}")

ax = axs[1]
sns.lineplot(data=small_results, x="k", y="contra_lik", hue="train", ax=ax)
ax.lines[0].set_linestyle("--")
ax.lines[1].set_linestyle("--")
sns.lineplot(data=small_results, x="k", y="ipsi_lik", hue="train", ax=ax)
ax.set_ylabel("Log likelihood")

leg = ax.get_legend()
leg.set_title("Train side")
leg.texts[0].set_text("Test contra")
leg.set_bbox_to_anchor((1, 1.5))
lines = leg.get_lines()
lines[0].set_linestyle("--")
lines[1].set_linestyle("--")
lines[2].set_linestyle("--")
leg.texts[3].set_text("Test ipsi")

stashfig(f"cross-val-n_components={n_components}")


# %% [markdown]
# ## Set k = 8, set n_components = 4


k = 6
n_per_hemisphere = 1000
res = small_results[small_results["k"] == k]
models = res["cluster"].values

X1, y1 = models[0].model_.sample(n_per_hemisphere)
X2, y2 = models[1].model_.sample(n_per_hemisphere)
y2 += y1.max()
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)
pairplot(X, labels=y, palette=cc.glasbey_light)
# %% [markdown]
# ##
# pairplot(train_embed, labels=meta["merge_class"].values, palette=CLASS_COLOR_DICT)

# %% [markdown]
# ##

import matplotlib as mpl

left_model = models[0].model_
right_model = models[1].model_
colors = cc.glasbey_light[:k]


def make_ellipses(gmm, ax, i, j, colors, alpha=0.5, **kws):
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
        # ax.set_aspect("equal", "datalim")


n_dims = X.shape[1]


fig, axs = plt.subplots(
    n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20), frameon=False
)
X = train_embed
data = pd.DataFrame(data=X)
data["label"] = meta["merge_class"].values
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
        colors = cc.glasbey_light[: 2 * k]
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
            make_ellipses(left_model, ax, i, j, colors[:k], fill=False)
            make_ellipses(right_model, ax, i, j, colors[k:], fill=False)
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
                palette=colors
                # color="black",
                # palette=CLASS_COLOR_DICT,
            )
            make_ellipses(left_model, ax, i, j, colors[:k], fill=True)
            make_ellipses(right_model, ax, i, j, colors[k:], fill=True)

plt.tight_layout()

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"


stashfig(f"gmm-crossval-pairs-k={k}-n_components={n_components}")
stashfig(f"gmm-crossval-pairs-k={k}-n_components={n_components}", fmt="pdf")

# %% [markdown]
# ##

from src.visualization import barplot_text, stacked_barplot

# barplot_text(pred, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT)
stacked_barplot(
    pred, meta["merge_class"].values, color_dict=CLASS_COLOR_DICT, legend_ncol=4
)

stashfig(f"gmm-crossval-barplot-k={k}-n_components={n_components}")

# %% [markdown]
# ## SUBCLUSTER !

from scipy.optimize import linear_sum_assignment


def compute_pairedness(partition, meta, rand_adjust=False, plot=False):
    partition = partition.copy()
    meta = meta.copy()

    uni_labels, inv = np.unique(partition, return_inverse=True)

    train_int_mat = np.zeros((len(uni_labels), len(uni_labels)))
    meta = meta.loc[partition.index]

    for i, ul in enumerate(uni_labels):
        c1_mask = inv == i

        c1_pairs = meta.loc[c1_mask, "Pair"]
        c1_pairs.drop(
            c1_pairs[c1_pairs == -1].index
        )  # HACK must be a better pandas sol

        for j, ul in enumerate(uni_labels):
            c2_mask = inv == j
            c2_inds = meta.loc[c2_mask].index
            train_pairs_in_other = np.sum(c1_pairs.isin(c2_inds))
            train_int_mat[i, j] = train_pairs_in_other

    row_ind, col_ind = linear_sum_assignment(train_int_mat, maximize=True)
    train_pairedness = np.trace(train_int_mat[np.ix_(row_ind, col_ind)]) / np.sum(
        train_int_mat
    )

    if plot:
        # FIXME broken
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(
            train_int_mat, square=True, ax=axs[0], cbar=False, cmap="RdBu_r", center=0
        )
        int_df = pd.DataFrame(data=train_int_mat, index=uni_labels, columns=uni_labels)
        int_df = int_df.reindex(index=uni_labels[row_ind])
        int_df = int_df.reindex(columns=uni_labels[col_ind])
        sns.heatmap(int_df, square=True, ax=axs[1], cbar=False, cmap="RdBu_r", center=0)

    if rand_adjust:
        # attempt to correct for difference in matchings as result of random chance
        # TODO this could be analytic somehow
        part_vals = partition.values
        np.random.shuffle(part_vals)
        partition = pd.Series(data=part_vals, index=partition.index)
        rand_train_pairedness, rand_test_pairedness = compute_pairedness(
            partition, meta, rand_adjust=False, plot=False, holdout=holdout
        )
        test_pairedness -= rand_test_pairedness
        train_pairedness -= rand_train_pairedness
        # pairedness = pairedness - rand_pairedness
    return train_pairedness, row_ind, col_ind


pness, row_ind, col_ind = compute_pairedness(
    pd.Series(index=meta.index, data=pred), meta, plot=True
)

dict(zip(row_ind, col_ind))

# %% [markdown]
# ##

pair = (3, 6)

left_sub_inds = np.where(pred == pair[0])[0]
right_sub_inds = np.where(pred == pair[1])[0]
inds = np.concatenate((left_sub_inds, right_sub_inds))

ase = AdjacencySpectralEmbed()
embed = ase.fit_transform(adj[np.ix_(inds, inds)])

left_embed = train_embed[left_sub_inds]
right_embed = train_embed[right_sub_inds]
rows = []
for k in tqdm(range(2, 15)):
    # train left, test right
    # TODO add option for AutoGMM as well, might as well check
    left_gc = GaussianCluster(min_components=k, max_components=k, n_init=n_init)
    left_gc.fit(left_embed)
    model = left_gc.model_
    train_left_bic = model.bic(left_embed)
    train_left_lik = model.score(left_embed)
    test_left_bic = model.bic(right_embed @ R.T)
    test_left_lik = model.score(right_embed @ R.T)

    row = {
        "k": k,
        "contra_bic": test_left_bic,
        "contra_lik": test_left_lik,
        "ipsi_bic": train_left_bic,
        "ipsi_lik": train_left_lik,
        "cluster": left_gc,
        "train": "left",
        "n_components": n_components,
    }
    rows.append(row)

    # train right, test left
    right_gc = GaussianCluster(min_components=k, max_components=k, n_init=n_init)
    right_gc.fit(right_embed)
    model = right_gc.model_
    train_right_bic = model.bic(right_embed)
    train_right_lik = model.score(right_embed)
    test_right_bic = model.bic(left_embed @ R)
    test_right_lik = model.score(left_embed @ R)

    row = {
        "k": k,
        "contra_bic": test_right_bic,
        "contra_lik": test_right_lik,
        "ipsi_bic": train_right_bic,
        "ipsi_lik": train_right_lik,
        "cluster": right_gc,
        "train": "right",
        "n_components": n_components,
    }
    rows.append(row)

print(f"{time.time() - currtime} elapsed")

results = pd.DataFrame(rows)


small_results = results[results["n_components"] == n_components]
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax = axs[0]
sns.lineplot(data=small_results, x="k", y="contra_bic", hue="train", ax=ax)
ax.lines[0].set_linestyle("--")
ax.lines[1].set_linestyle("--")
sns.lineplot(data=small_results, x="k", y="ipsi_bic", hue="train", ax=ax)
ax.set_ylabel("BIC")
leg = ax.get_legend().remove()
ax.set_title(f"n_components={n_components}")

ax = axs[1]
sns.lineplot(data=small_results, x="k", y="contra_lik", hue="train", ax=ax)
ax.lines[0].set_linestyle("--")
ax.lines[1].set_linestyle("--")
sns.lineplot(data=small_results, x="k", y="ipsi_lik", hue="train", ax=ax)
ax.set_ylabel("Log likelihood")

leg = ax.get_legend()
leg.set_title("Train side")
leg.texts[0].set_text("Test contra")
leg.set_bbox_to_anchor((1, 1.5))
lines = leg.get_lines()
lines[0].set_linestyle("--")
lines[1].set_linestyle("--")
lines[2].set_linestyle("--")
leg.texts[3].set_text("Test ipsi")
# %% [markdown]
# ##
# X = np.concatenate((left_embed, right_embed))
k = 5
# data = pd.DataFrame
X = np.concatenate((left_embed, right_embed))
n_per_hemisphere = 1000
res = small_results[small_results["k"] == k]
models = res["cluster"].values
left_model = models[0].model_
right_model = models[1].model_
labels = meta["merge_class"].values[inds]
pred = np.empty(len(labels), dtype=int)
left_pred = left_model.predict(left_embed)
right_pred = right_model.predict(right_embed) + left_pred.max() + 1
pred[: len(left_sub_inds)] = left_pred
pred[len(left_sub_inds) :] = right_pred
stacked_barplot(pred, labels, color_dict=CLASS_COLOR_DICT, legend_ncol=4)
stashfig("example-subclust")
