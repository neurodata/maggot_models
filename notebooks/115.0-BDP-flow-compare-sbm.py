# %% [markdown]
# ##
from src.hierarchy import signal_flow
from src.data import load_metagraph
from src.visualization import matrixplot
from src.visualization import CLASS_COLOR_DICT
from src.io import savefig
import os
from src.graph import preprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

sns.set_context("talk")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


from src.traverse import Cascade, RandomWalk, TraverseDispatcher


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


low_p = 0.05
diag_p = 0.1
feedforward_p = 0.3
n_blocks = 5
basename = f"-{feedforward_p}-{diag_p}-{low_p}-{n_blocks}"


block_probs = get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=n_blocks)
plt.figure(figsize=(10, 10))
sns.heatmap(block_probs, annot=True, cmap="Reds", cbar=False)
plt.title("Feedforward block probability matrix")

#%%
from graspy.simulations import sbm

community_sizes = np.empty(2 * n_blocks, dtype=int)
n_feedforward = 100
community_sizes = n_blocks * [n_feedforward]

np.random.seed(88)
adj, labels = sbm(
    community_sizes, block_probs, directed=True, loops=False, return_labels=True
)
n_verts = adj.shape[0]

matrixplot(adj, row_sort_class=labels, col_sort_class=labels, cbar=False)
stashfig("sbm" + basename)

# %% [markdown]
# ##
from src.traverse import to_transmission_matrix
from src.traverse import to_markov_matrix, RandomWalk

method = "Cascade"
max_hops = 10
n_init = 100
p = 0.03

if method == "Cascade":
    transition_probs = to_transmission_matrix(adj, p)
    traverse = Cascade
    simultaneous = True
elif method == "Random walk":
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
hop_hist = td.multistart(np.arange(n_feedforward))
hop_hist = hop_hist.T

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(hop_hist, ax=ax, col_sort_class=labels)

col_df = pd.DataFrame(labels, columns=["label"])
# %% [markdown]
# ##

n_visits = np.sum(hop_hist, axis=0)
weight_sum_visits = (np.arange(1, max_hops + 1)[:, None] * hop_hist).sum(axis=0)
mean_visit = weight_sum_visits / n_visits

col_df["mean_visit"] = mean_visit
group_means = col_df.groupby("label")["mean_visit"].mean()
col_df["group_mean_visit"] = col_df["label"].map(group_means)
# col_meta.append(col_df)

sf = signal_flow(adj)
col_df["signal_flow"] = -sf

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(
    hop_hist,
    ax=ax,
    col_meta=col_df,
    col_sort_class=["label"],
    col_class_order="group_mean_visit",
    col_item_order="mean_visit",
)
stashfig(f"hop-hist-{method}" + basename)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ax = axs[0]
matrixplot(
    adj,
    ax=ax,
    col_meta=col_df,
    col_colors="label",
    col_item_order="mean_visit",
    row_meta=col_df,
    row_colors="label",
    row_item_order="mean_visit",
    square=True,
    cbar=False,
)
ax.set_title(f"{method} sort", pad=22)


ax = axs[1]
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
ax.set_title("Signal flow sort", pad=22)

stashfig(f"sbm-flow-compare-{method}" + basename)

