# %% [markdown]
# # Load and import
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.plot import gridplot, heatmap, pairplot
from graspy.simulations import sbm
from graspy.utils import binarize, get_lcc, is_fully_connected
from src.data import load_everything
from src.hierarchy import normalized_laplacian, signal_flow
from src.io import savefig
from src.visualization import gridmap
from src.data import load_metagraph
import matplotlib as mpl

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
SAVEFIGS = True
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


GRAPH_VERSION = "2020-01-21"


graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
graph_type_labels = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
sns.set_context("talk", font_scale=1)


def compute_triu_prop(A, return_edges=False):
    sum_triu_edges = np.sum(np.triu(A, k=1))
    sum_tril_edges = np.sum(np.tril(A, k=-1))
    triu_prop = sum_triu_edges / (sum_triu_edges + sum_tril_edges)
    if return_edges:
        return triu_prop, sum_triu_edges, sum_tril_edges
    else:
        return triu_prop


def shuffle_edges(A):
    fake_A = A.copy().ravel()
    np.random.shuffle(fake_A)
    fake_A = fake_A.reshape((n_verts, n_verts))
    return fake_A


def signal_flow_sort(A, return_inds=False):
    A = A.copy()
    nodes_signal_flow = signal_flow(A)
    sort_inds = np.argsort(nodes_signal_flow)[::-1]
    sorted_A = A[np.ix_(sort_inds, sort_inds)]
    if return_inds:
        return sorted_A, sort_inds
    else:
        return sorted_A


# %% [markdown]
# #
mpl.rcParams["axes.titlepad"] = 10

shuffled_triu_outs = []
true_triu_outs = []
n_shuffles = 20
palette = sns.color_palette("deep", len(graph_types))

for i, (g, name) in enumerate(zip(graph_types, graph_type_labels)):

    mg = load_metagraph(g, version=GRAPH_VERSION)
    mg.make_lcc()
    adj = mg.adj
    n_verts = adj.shape[0]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # shuffle original adj just to avoid sorting spandrels
    perm_inds = np.random.permutation(n_verts)
    adj = adj[np.ix_(perm_inds, perm_inds)]

    # compare to a fake network with same weights
    for _ in range(n_shuffles):
        fake_adj = shuffle_edges(adj)
        fake_adj = signal_flow_sort(fake_adj)
        fake_triu_prop = compute_triu_prop(fake_adj)
        out_dict = {"Proportion": fake_triu_prop, "Graph": name, "Type": "Shuffled"}
        shuffled_triu_outs.append(out_dict)
        print(
            f"{g} shuffled graph sorted proportion in upper triangle: {fake_triu_prop}"
        )
    gridmap_kws = {"c": [palette[i]], "sizes": (5, 10)}

    gridmap(fake_adj, ax=axs[0], **gridmap_kws)
    axs[0].set_title("Shuffled edges")
    axs[0].plot(
        [0, n_verts], [0, n_verts], color="grey", linewidth=2, linestyle="--", alpha=0.8
    )

    z = signal_flow(adj)
    sort_inds = np.argsort(z)[::-1]
    adj = adj[np.ix_(sort_inds, sort_inds)]

    true_triu_prop = compute_triu_prop(adj)

    out_dict = {"Proportion": true_triu_prop, "Graph": name, "Type": "True"}
    true_triu_outs.append(out_dict)
    print(f"{g} graph sorted proportion in upper triangle: {true_triu_prop}")

    gridmap(adj, ax=axs[1], **gridmap_kws)
    axs[1].set_title("True edges")
    axs[1].plot(
        [0, n_verts], [0, n_verts], color="grey", linewidth=2, linestyle="--", alpha=0.8
    )

    fig.suptitle(f"{name} ({n_verts})", fontsize=40, y=1.04)
    plt.tight_layout()
    stashfig(f"{g}-gridplot-sf-sorted")
    print()

#%%

shuffle_df = pd.DataFrame(shuffled_triu_outs)
true_df = pd.DataFrame(true_triu_outs)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax = sns.stripplot(
    data=shuffle_df,
    x="Graph",
    y="Proportion",
    linewidth=1,
    alpha=0.4,
    jitter=0.3,
    size=5,
    ax=ax,
)
# ax = sns.violinplot(data=shuffle_df, x="Graph", y="Proportion", ax=ax)

ax = sns.stripplot(
    data=true_df,
    x="Graph",
    y="Proportion",
    marker="_",
    linewidth=2,
    s=90,
    ax=ax,
    label="True",
    jitter=False,
)

shuffle_marker = plt.scatter([], [], marker=".", c="k", label="Shuffled")
true_marker = plt.scatter([], [], marker="_", linewidth=3, s=400, c="k", label="True")
ax.legend(handles=[shuffle_marker, true_marker])
ax.set_ylabel("Proportion synapses in upper triangle")
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_title("Signal flow graph sorting", fontsize=25, pad=15)
stashfig("sf-proportions")
#%%
prop_df = pd.DataFrame()
prop_df["Proportion"] = np.array(shuffled_triu_props + true_triu_props)
prop_df["Type"] = np.array(4 * ["Shuffled"] + 4 * ["True"])
prop_df["Graph"] = np.array(graph_type_labels + graph_type_labels)

ax = sns.pointplot(
    data=prop_df, x="Graph", y="Proportion", hue="Type", ci=None, join=False
)
ax.set_ylim((0.4, 1.05))
ax.axhline(0.5, linestyle="--")
ax.set_ylabel("Proportion upper triangular")
ax.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.0)
ax.set_title("")
# sns.pointplot()
# %% [markdown]
# # null simulation


def get_feedforward_B(low_p, diag_p, feedforward_p, n_blocks=5):
    B = np.zeros((n_blocks, n_blocks))
    B += low_p
    B -= np.diag(np.diag(B))
    B -= np.diag(np.diag(B, k=1), k=1)
    B += np.diag(diag_p * np.ones(n_blocks))
    B += np.diag(feedforward_p * np.ones(n_blocks - 1), k=1)
    return B


def n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


low_p = 0.01
diag_p = 0.1
feedforward_p = 0.2
community_sizes = np.array(5 * [50])
block_probs = get_feedforward_B(low_p, diag_p, feedforward_p)
A = sbm(community_sizes, block_probs, directed=True, loops=False)
n_verts = A.shape[0]

perm_inds = np.random.permutation(n_verts)
A_perm = A[np.ix_(perm_inds, perm_inds)]
heatmap(A, cbar=False, title="Feedforward SBM")
stashfig("ffSBM")
heatmap(A_perm, cbar=False, title="Feedforward SBM, shuffled")
stashfig("ffSBM-shuffle")
z = signal_flow(A)
sort_inds = np.argsort(z)[::-1]
heatmap(
    A[np.ix_(sort_inds, sort_inds)],
    cbar=False,
    title="Feedforward SBM, sorted by signal flow",
)
stashfig("ffSBM-sf")
A_fake = A.copy().ravel()
np.random.shuffle(A_fake)
A_fake = A_fake.reshape((n_verts, n_verts))
z = signal_flow(A_fake)
sort_inds = np.argsort(z)[::-1]
heatmap(
    A_fake[np.ix_(sort_inds, sort_inds)],
    cbar=False,
    title="Random network, sorted by signal flow",
)
stashfig("random-sf")

