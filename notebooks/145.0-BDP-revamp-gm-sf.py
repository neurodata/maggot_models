# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

from graspy.match import GraphMatch
from graspy.plot import heatmap
from graspy.simulations import sbm
from src.data import load_metagraph
from src.graph import preprocess
from src.io import savecsv, savefig
from src.utils import invert_permutation
from src.visualization import CLASS_COLOR_DICT, adjplot

print(scipy.__version__)

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


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


# %% [markdown]
# ## generate SBM
low_p = 0.01
diag_p = 0.1
feedforward_p = 0.3
n_blocks = 20
n_per_block = 25  # 50
community_sizes = n_blocks * [n_per_block]

basename = f"-n_blocks={n_blocks}-n_per_block={n_per_block}"

block_probs = get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=n_blocks)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False, ax=axs[0], square=True)
axs[0].xaxis.tick_top()
axs[0].set_title("Block probability matrix", pad=25)


np.random.seed(88)
adj, labels = sbm(
    community_sizes, block_probs, directed=True, loops=False, return_labels=True
)
n_verts = adj.shape[0]

adjplot(adj, sort_class=labels, cbar=False, ax=axs[1], square=True)
axs[1].set_title("Adjacency matrix", pad=25)
plt.tight_layout()
stashfig("sbm" + basename)


# %% [markdown]
# ## Create the matching matrix


def diag_indices(length, k=0):
    return (np.arange(length - k), np.arange(k, length))


def make_flat_match(length, **kws):
    match_mat = np.zeros((length, length))
    match_mat[np.triu_indices(length, k=1)] = 1
    return match_mat


def make_linear_match(length, offset=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = length - k + offset
    return match_mat


def make_exp_match(length, alpha=0.5, offset=0, **kws):
    match_mat = np.zeros((length, length))
    for k in np.arange(1, length):
        match_mat[diag_indices(length, k)] = np.exp(-alpha * (k - 1)) + offset
    return match_mat


def normalize_match(graph, match_mat):
    return match_mat / match_mat.sum() * graph.sum()


# %% [markdown]
# ##

methods = [make_flat_match, make_linear_match, make_exp_match]
names = ["Flat", "Linear", "Exp"]

gm = GraphMatch(
    n_init=50, init_method="rand", max_iter=100, eps=0.05, shuffle_input=True
)
alpha = 0.005
match_mats = []
permutations = []
for method, name in zip(methods, names):
    print(name)
    match_mat = method(len(adj), alpha=alpha)
    match_mat = normalize_match(adj, match_mat)
    match_mats.append(match_mat)
    gm.fit(match_mat, adj)
    permutations.append(gm.perm_inds_)

# %% [markdown]
# ##
from src.hierarchy import signal_flow
from src.visualization import remove_axis
import pandas as pd

n_verts = len(adj)
sf = signal_flow(adj)
sf_perm = np.argsort(-sf)
inds = np.arange(n_verts)

plot_df = pd.DataFrame()
plot_df["labels"] = labels
plot_df["x"] = inds


def format_order_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("True order")
    ax.axis("square")


if n_blocks > 10:
    pal = "tab20"
else:
    pal = "tab10"
color_dict = dict(zip(np.unique(labels), sns.color_palette(pal, n_colors=n_blocks)))


def plot_diag_boxes(ax):
    for i in range(n_blocks):
        low = i * n_per_block - 0.5
        high = (i + 1) * n_per_block + 0.5
        xs = [low, high, high, low, low]
        ys = [low, low, high, high, low]
        ax.plot(xs, ys, color=color_dict[i], linestyle="--", linewidth=0.7, alpha=0.7)


def calc_accuracy(block_preds):
    acc = (block_preds == labels).astype(float).mean()
    return acc


def calc_abs_dist(block_preds):
    mae = np.abs(block_preds - labels).mean()
    return mae


def calc_euc_dist(block_preds):
    sse = np.sqrt(((block_preds - labels) ** 2).sum())
    mse = sse / len(block_preds)
    return mse


def plot_scores(perm, ax):
    block_preds = perm // n_per_block
    acc = calc_accuracy(block_preds)
    mae = calc_abs_dist(block_preds)
    mse = calc_euc_dist(block_preds)
    ax.text(
        0.75,
        0.07,
        f"Acc. {acc:.2f}\nMAE {mae:.2f}\nMSE {mse:.2f}",
        transform=ax.transAxes,
    )


# model
fig, axs = plt.subplots(3, 6, figsize=(30, 15))

scatter_kws = dict(
    x="x",
    y="y",
    hue="labels",
    s=7,
    linewidth=0,
    palette=color_dict,
    legend=False,
    alpha=1,
)
first = 0
ax = axs[0, first]
ax.set_title("Model (truth)")
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False, ax=ax, square=True)
show_annot_array = np.zeros_like(block_probs, dtype=bool)
show_annot_array[0, :3] = 1
for text, show_annot in zip(
    ax.texts, (element for row in show_annot_array for element in row)
):
    text.set_visible(show_annot)
ax.set_xticks([])
ax.set_yticks([])

adjplot(adj, colors=labels, ax=axs[1, first], cbar=False)
plot_df["y"] = inds
ax = axs[2, first]
sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
format_order_ax(ax)
ax.set_ylabel("Predicted order")
plot_diag_boxes(ax)
plot_scores(inds, ax)

# random
first = 1
remove_axis(axs[0, first])
axs[0, first].set_title("Random")
perm = inds.copy()
np.random.shuffle(perm)
adjplot(adj[np.ix_(perm, perm)], colors=labels[perm], ax=axs[1, first], cbar=False)
plot_df["y"] = perm
ax = axs[2, first]
sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
format_order_ax(ax)
plot_diag_boxes(ax)
plot_scores(perm, ax)

# signal flow
first = 2
remove_axis(axs[0, first])
axs[0, first].set_title("Signal flow")
adjplot(
    adj[np.ix_(sf_perm, sf_perm)], colors=labels[sf_perm], ax=axs[1, first], cbar=False
)
plot_df["y"] = sf_perm
ax = axs[2, first]
sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
format_order_ax(ax)
plot_diag_boxes(ax)
plot_scores(sf_perm, ax)


# graph matching
first = 3
for i, (match, perm) in enumerate(zip(match_mats, permutations)):
    axs[0, i + first].set_title(names[i])
    # matching matrix
    adjplot(match, ax=axs[0, i + first], cbar=False)
    # adjacency
    adjplot(
        adj[np.ix_(perm, perm)], colors=labels[perm], ax=axs[1, i + first], cbar=False
    )
    # ranks
    plot_df["y"] = perm
    ax = axs[2, i + first]
    sns.scatterplot(data=plot_df, ax=ax, **scatter_kws)
    format_order_ax(ax)
    plot_diag_boxes(ax)
    plot_scores(perm, ax)


plt.tight_layout()
stashfig("sbm-ordering" + basename)


# axs[0, first].set_title("Signal flow")
# axs[0, first].set_ylabel("Match matrix")
# axs[1, first].set_ylabel("Sorted adjacency")
# %%
