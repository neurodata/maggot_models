# %% [markdown]
# ##
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.simulations import sbm
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savefig
from src.traverse import (
    Cascade,
    RandomWalk,
    TraverseDispatcher,
    to_markov_matrix,
    to_transmission_matrix,
)
from src.visualization import CLASS_COLOR_DICT, matrixplot

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

sns.set_context("talk", font_scale=1.5)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, dpi=200, **kws)


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


def hist_from_cascade(
    transition_probs,
    start_nodes,
    stop_nodes=[],
    allow_loops=False,
    n_init=100,
    simultaneous=True,
    max_hops=10,
):
    dispatcher = TraverseDispatcher(
        Cascade,
        transition_probs,
        allow_loops=allow_loops,
        n_init=n_init,
        max_hops=max_hops,
        stop_nodes=stop_nodes,
    )
    if simultaneous:
        hop_hist = dispatcher.start(start_nodes)
    else:
        n_verts = len(transition_probs)
        hop_hist = np.zeros((n_verts, max_hops))
        for s in start_nodes:
            hop_hist += dispatcher.start(s)
    return hop_hist


low_p = 0.01
diag_p = 0.1
feedforward_p = 0.3
n_blocks = 10
basename = f"-{feedforward_p}-{diag_p}-{low_p}-{n_blocks}"


block_probs = get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=n_blocks)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False, ax=axs[0], square=True)
axs[0].xaxis.tick_top()
axs[0].set_title("Block probability matrix", pad=25)

community_sizes = np.empty(2 * n_blocks, dtype=int)
n_feedforward = 100
community_sizes = n_blocks * [n_feedforward]

np.random.seed(88)
adj, labels = sbm(
    community_sizes, block_probs, directed=True, loops=False, return_labels=True
)
n_verts = adj.shape[0]

matrixplot(
    adj,
    row_sort_class=labels,
    col_sort_class=labels,
    cbar=False,
    ax=axs[1],
    square=True,
)
axs[1].set_title("Adjacency matrix", pad=25)
plt.tight_layout()
stashfig("sbm" + basename)

# %% [markdown]
# ##

max_hops = 10
n_init = 100
p = 0.03

# random walk
transition_probs = to_markov_matrix(adj)  # row normalize!
traverse = RandomWalk
simultaneous = False
td = TraverseDispatcher(
    traverse,
    transition_probs,
    n_init=n_init,
    simultaneous=simultaneous,
    stop_nodes=np.arange(n_feedforward * (n_blocks - 1), n_feedforward * (n_blocks)),
    max_hops=max_hops,
    allow_loops=False,
)
rw_hop_hist = td.multistart(np.arange(n_feedforward))
rw_hop_hist = rw_hop_hist.T
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(rw_hop_hist, ax=ax, col_sort_class=labels)

# forward cascade
transition_probs = to_transmission_matrix(adj, p)
traverse = Cascade
simultaneous = True
td = TraverseDispatcher(
    traverse,
    transition_probs,
    n_init=n_init,
    simultaneous=simultaneous,
    stop_nodes=np.arange(n_feedforward * (n_blocks - 1), n_feedforward * (n_blocks)),
    max_hops=max_hops,
    allow_loops=False,
)
casc_hop_hist = td.multistart(np.arange(n_feedforward))
casc_hop_hist = casc_hop_hist.T
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(casc_hop_hist, ax=ax, col_sort_class=labels)

# backward cascade
td = TraverseDispatcher(
    traverse,
    transition_probs.T,
    n_init=n_init,
    simultaneous=simultaneous,
    stop_nodes=np.arange(n_feedforward),
    max_hops=max_hops,
    allow_loops=False,
)
back_hop_hist = td.multistart(
    np.arange(n_feedforward * (n_blocks - 1), n_feedforward * (n_blocks))
)
back_hop_hist = back_hop_hist.T
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(back_hop_hist, ax=ax, col_sort_class=labels)

col_df = pd.DataFrame(labels, columns=["label"])
# %% [markdown]
# ##


def compute_mean_visit(hop_hist):
    n_visits = np.sum(hop_hist, axis=0)
    weight_sum_visits = (np.arange(1, max_hops + 1)[:, None] * hop_hist).sum(axis=0)
    mean_visit = weight_sum_visits / n_visits
    return mean_visit


col_df["rw_mean_visit"] = compute_mean_visit(rw_hop_hist)
col_df["casc_mean_visit"] = compute_mean_visit(casc_hop_hist)
col_df["back_mean_visit"] = compute_mean_visit(back_hop_hist)
col_df["diff"] = col_df["casc_mean_visit"] - col_df["back_mean_visit"]
col_df["signal_flow"] = -signal_flow(adj)


# %% [markdown]
# ##
sns.set_context("talk", font_scale=1.5)
pad = 30
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
ax = axs[0, 0]
method = "Signal flow"
matrixplot(
    adj,
    ax=ax,
    col_meta=col_df,
    col_colors="label",
    col_item_order="signal_flow",
    row_meta=col_df,
    row_colors="label",
    row_item_order="signal_flow",
    square=True,
    cbar=False,
)
ax.set_title(f"{method} sort", pad=pad)


ax = axs[0, 1]
method = "Random walk"
matrixplot(
    adj,
    ax=ax,
    col_meta=col_df,
    col_colors="label",
    col_item_order="rw_mean_visit",
    row_meta=col_df,
    row_colors="label",
    row_item_order="rw_mean_visit",
    square=True,
    cbar=False,
)
ax.set_title(f"{method} sort", pad=pad)

ax = axs[1, 0]
method = "Cascade"
matrixplot(
    adj,
    ax=ax,
    col_meta=col_df,
    col_colors="label",
    col_item_order="casc_mean_visit",
    row_meta=col_df,
    row_colors="label",
    row_item_order="casc_mean_visit",
    square=True,
    cbar=False,
)
ax.set_title(f"{method} sort", pad=pad)

ax = axs[1, 1]
method = "Diff"
matrixplot(
    adj,
    ax=ax,
    col_meta=col_df,
    col_colors="label",
    col_item_order="diff",
    row_meta=col_df,
    row_colors="label",
    row_item_order="diff",
    square=True,
    cbar=False,
)
ax.set_title(f"{method} sort", pad=pad)
plt.tight_layout()
stashfig(f"sbm-flow-compare" + basename)

