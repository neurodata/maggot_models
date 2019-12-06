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
from src.utils import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
SAVEFIGS = False
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


GRAPH_VERSION = "2019-09-18-v2"
adj, class_labels, side_labels = load_everything(
    "Gad", GRAPH_VERSION, return_class=True, return_side=True
)

adj, inds = get_lcc(adj, return_inds=True)
class_labels = class_labels[inds]
side_labels = side_labels[inds]
n_verts = adj.shape[0]

# %% [markdown]
# # Sort shuffled AD graph on signal flow
fake_adj = adj.copy()
np.random.shuffle(fake_adj.ravel())
fake_adj = fake_adj.reshape((n_verts, n_verts))

z = signal_flow(fake_adj)
sort_inds = np.argsort(z)[::-1]

gridplot([fake_adj[np.ix_(sort_inds, sort_inds)]], height=20)
stashfig("gridplot-sf-sorted-fake")


# %% [markdown]
# # Sort true AD graph on signal flow

z = signal_flow(adj)
sort_inds = np.argsort(z)[::-1]

gridplot([adj[np.ix_(sort_inds, sort_inds)]], height=20)
stashfig("gridplot-sf-sorted")

# %% [markdown]
# # Repeat


# %% [markdown]
# # Look at graph laplacians

evecs, evals = normalized_laplacian(
    adj, n_components=5, return_evals=True, normalize_evecs=True
)


scatter_mat = np.concatenate((z[:, np.newaxis], evecs), axis=1)
pairplot(scatter_mat, labels=class_labels, palette="tab20")

# %% [markdown]
# # Examine the 2nd eigenvector
degree = ((adj + adj.T) / 2).sum(axis=1)

evecs, evals = normalized_laplacian(
    adj, n_components=2, return_evals=True, normalize_evecs=True
)
evec2_norm = evecs[:, 1]

evecs, evals = normalized_laplacian(
    adj, n_components=2, return_evals=True, normalize_evecs=False
)
evec2_unnorm = evecs[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(evec2_norm)

plt.figure(figsize=(10, 6))
plt.plot(evec2_unnorm)

plt.figure(figsize=(10, 6))
plt.scatter(evec2_norm, degree)

plt.figure(figsize=(10, 6))
plt.scatter(evec2_unnorm, degree)


# %% [markdown]
# #
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
graph_type_labels = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]

GRAPH_VERSION = "2019-09-18-v2"
sns.set_context("talk", font_scale=1)


def gridmap(A, ax=None, legend=False, sizes=(10, 70)):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 20))
    n_verts = A.shape[0]
    inds = np.nonzero(A)
    edges = A[inds]
    scatter_df = pd.DataFrame()
    scatter_df["Weight"] = edges
    scatter_df["x"] = inds[1]
    scatter_df["y"] = inds[0]
    ax = sns.scatterplot(
        data=scatter_df,
        x="x",
        y="y",
        size="Weight",
        legend=legend,
        sizes=sizes,
        ax=ax,
        linewidth=0.3,
    )
    ax.axis("equal")
    ax.set_xlim((0, n_verts))
    ax.set_ylim((n_verts, 0))
    ax.axis("off")
    return ax


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
    np.random.shuffle(Aj)
    A = A.reshape((n_verts, n_verts))
    return A


def signal_flow_sort(A, return_inds=False):
    nodes_signal_flow = signal_flow(A)
    sort_inds = np.argsort(nodes_signal_flow)[::-1]
    sorted_A = A[np.ix_(sort_inds, sort_inds)]
    if return_inds:
        return A, sort_inds
    else:
        return sorted_A


shuffled_triu_props = []
true_triu_props = []

for g, name in zip(graph_types, graph_type_labels):
    adj = load_everything(g, GRAPH_VERSION)
    adj, inds = get_lcc(adj, return_inds=True)
    n_verts = adj.shape[0]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # shuffle original adj just to avoid sorting spandrels
    perm_inds = np.random.permutation(n_verts)
    adj = adj[np.ix_(perm_inds, perm_inds)]

    # compare to a fake network with same weights
    fake_adj = shuffle_edges(adj)
    fake_adj = signal_flow_sort(fake_adj)
    fake_triu_prop = compute_triu_prop(fake_adj)

    shuffled_triu_props.append(fake_triu_prop)
    # print(f"{g} shuffled graph sorted upper triangle synapses: {fake_sum_triu_edges}")
    # print(f"{g} shuffled graph sorted upper triange synapses: {fake_sum_tril_edges}")
    print(f"{g} shuffled graph sorted proportion in upper triangle: {fake_triu_prop}")

    gridmap(fake_adj, ax=axs[0])
    axs[0].set_title("Shuffled edges")

    z = signal_flow(adj)
    sort_inds = np.argsort(z)[::-1]
    adj = adj[np.ix_(sort_inds, sort_inds)]

    true_triu_prop = compute_triu_prop(adj)
    true_triu_props.append(true_triu_prop)
    # print(f"Is {g} graph fully connected: {is_fully_connected(adj)}")
    # print(f"{g} graph sorted upper triangle synapses: {sum_triu_edges}")
    # print(f"{g} graph sorted upper triange synapses: {sum_tril_edges}")
    print(f"{g} graph sorted proportion in upper triangle: {true_triu_prop}")

    gridmap(adj, ax=axs[1])
    axs[1].set_title("True edges")

    fig.suptitle(f"{name} ({n_verts})", fontsize=40, y=1.02)
    plt.tight_layout()
    stashfig(f"{g}-gridplot-sf-sorted")
    print()

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

