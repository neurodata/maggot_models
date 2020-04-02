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
    savefig(name, foldername=FNAME, save_on=True, dpi=200, **kws)


mg = load_metagraph("G")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
# %% [markdown]
# ##


adj = mg.adj
adj = pass_to_ranks(adj)
meta = mg.meta

ase = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
embed = ase.fit_transform(adj)


# %% [markdown]
# ##


meta["inds"] = range(len(meta))
left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]


# %% [markdown]
# ##
#
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


lp_inds, rp_inds = get_paired_inds(meta)

# meta[meta["left"] & (meta["Pair"] != -1)]


# %% [markdown]
# ##


currtime = time.time()

n_init = 50


rows = []
Rs = []
n_components = embed[0].shape[1]
print(n_components)
print()
train_embed = np.concatenate(
    (embed[0][:, :n_components], embed[1][:, :n_components]), axis=-1
)
R, _ = orthogonal_procrustes(train_embed[lp_inds], train_embed[rp_inds])
Rs.append(R)
left_embed = train_embed[left_inds]
left_embed = left_embed @ R
right_embed = train_embed[right_inds]

for k in tqdm(range(2, 15)):
    # train left, test right
    left_gc = GaussianCluster(min_components=k, max_components=k, n_init=n_init)
    left_gc.fit(left_embed)
    model = left_gc.model_
    train_left_bic = model.bic(left_embed)
    train_left_lik = model.score(left_embed)
    test_left_bic = model.bic(right_embed)
    test_left_lik = model.score(right_embed)

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
    test_right_bic = model.bic(left_embed)
    test_right_lik = model.score(left_embed)

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

# %% [markdown]
# ##


results = pd.DataFrame(rows)
mean_results = results.groupby(["n_components", "k"]).mean()
mean_results.reset_index(inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.lineplot(
    data=mean_results, x="k", y="contra_bic", hue="n_components", ax=ax, palette="Reds"
)
sns.lineplot(
    data=mean_results, x="k", y="ipsi_bic", hue="n_components", ax=ax, palette="Blues"
)
plt.legend(bbox_to_anchor=(1, 1))

#%%
small_results = results[results["n_components"] == 7]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.scatterplot(data=small_results, x="k", y="contra_bic", hue="train", ax=ax)
small_mean_results = small_results.groupby("k").mean()
small_mean_results.reset_index(inplace=True)
sns.lineplot(
    data=small_mean_results,
    x="k",
    y="contra_bic",
    label="mean",
    color="purple",
    ax=ax,
    # marker="+",
    # s=100,
)
plt.legend(bbox_to_anchor=(1, 1))
ax.set_ylabel("BIC")

# %% [markdown]
# ##
n_components = 4
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


k = 8
n_per_hemisphere = 1000
R = Rs[3]
R_inv = np.linalg.inv(R)
res = small_results[small_results["k"] == k]
models = res["cluster"].values

X1, y1 = models[0].model_.sample(n_per_hemisphere)
X2, y2 = models[1].model_.sample(n_per_hemisphere)
X = np.concatenate((X1, X2), axis=0)
np.random.shuffle(X)
X[:n_per_hemisphere] = X[:n_per_hemisphere] @ R_inv

pairplot(X)

X_left = X[:, :n_components]
X_right = X[:, n_components:]


graph = rdpg(X_left, X_right, rescale=False, directed=True, loops=True)
# %% [markdown]
# ##
n_components = 4
train_embed = np.concatenate(
    (embed[0][:, :n_components], embed[1][:, :n_components]), axis=-1
)
R, _ = orthogonal_procrustes(train_embed[lp_inds], train_embed[rp_inds])
Rs.append(R)
left_embed = train_embed[left_inds]
left_embed = left_embed @ R
right_embed = train_embed[right_inds]

pred_left = models[0].model_.predict(left_embed)
pred_right = models[1].model_.predict(right_embed)
pred_left += len(np.unique(pred_right)) + 1

pred = np.empty(len(embed[0]))
pred[left_inds] = pred_left
pred[right_inds] = pred_right
meta["joint_pred"] = pred

ax, _, tax, _ = matrixplot(
    binarize(adj),
    plot_type="scattermap",
    sizes=(0.25, 0.5),
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_meta=meta,
    col_sort_class=["hemisphere", "joint_pred"],
    col_ticks=False,
    # col_class_order="block_sf",
    col_item_order="adj_sf",
    row_ticks=False,
    row_colors="merge_class",
    row_palette=CLASS_COLOR_DICT,
    row_meta=meta,
    row_sort_class=["hemisphere", "joint_pred"],
    # row_class_order="block_sf",
    row_item_order="adj_sf",
)
# %% [markdown]
# ##


# %% [markdown]
# ##

# %% [markdown]
# ##


pairplot(embed, labels=pred_labels, palette=cc.glasbey_light)

# %% [markdown]
# ##


sbm = DCSBMEstimator(directed=True, degree_directed=True, loops=False, max_comm=30)
sbm.fit(binarize(adj))
pred_labels = sbm.vertex_assignments_
print(len(np.unique(pred_labels)))

meta["pred_labels"] = pred_labels

graph = np.squeeze(sbm.sample())


meta["adj_sf"] = -signal_flow(binarize(adj))

block_sf = -signal_flow(sbm.block_p_)
block_map = pd.Series(data=block_sf)
meta["block_sf"] = meta["pred_labels"].map(block_map)

#%%
graph_type = "G"
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
ax, _, tax, _ = matrixplot(
    binarize(adj),
    ax=ax,
    plot_type="scattermap",
    sizes=(0.25, 0.5),
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_meta=meta,
    col_sort_class="pred_labels",
    col_ticks=False,
    col_class_order="block_sf",
    col_item_order="adj_sf",
    row_ticks=False,
    row_colors="merge_class",
    row_palette=CLASS_COLOR_DICT,
    row_meta=meta,
    row_sort_class="pred_labels",
    row_class_order="block_sf",
    row_item_order="adj_sf",
)
tax.set_title(f"Data ({graph_type})")
ax = axs[1]
ax, _, tax, _ = matrixplot(
    graph,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.25, 0.5),
    col_colors="merge_class",
    col_palette=CLASS_COLOR_DICT,
    col_meta=meta,
    col_ticks=False,
    col_class_order="block_sf",
    row_ticks=False,
    col_sort_class="pred_labels",
    col_item_order="adj_sf",
    row_colors="merge_class",
    row_palette=CLASS_COLOR_DICT,
    row_meta=meta,
    row_sort_class="pred_labels",
    row_class_order="block_sf",
    row_item_order="adj_sf",
)
tax.set_title(f"Model sample (dDCSBM)")

stashfig("null-model-adjs")

# %% [markdown]
# ##
block_p = sbm.block_p_
block_meta = pd.DataFrame(index=range(len(block_p)))
block_meta["signal_flow"] = block_sf
matrixplot(
    block_p,
    col_meta=block_meta,
    row_meta=block_meta,
    col_item_order="signal_flow",
    row_item_order="signal_flow",
)


# %% [markdown]
# ##

sbm = DCSBMEstimator(directed=True, degree_directed=True, loops=False, max_comm=30)
sbm.fit(binarize(adj), y=pred)
