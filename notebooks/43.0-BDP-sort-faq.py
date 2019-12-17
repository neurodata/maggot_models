# %% [markdown]
# # Load and import
import os
from operator import itemgetter
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.match import FastApproximateQAP

from graspy.plot import gridplot, heatmap, pairplot
from graspy.simulations import sbm
from graspy.utils import binarize, get_lcc, is_fully_connected
from src.data import load_everything
from src.hierarchy import normalized_laplacian, signal_flow
from src.utils import savefig, shuffle_edges

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


GRAPH_VERSION = "2019-09-18-v2"
adj, class_labels, side_labels = load_everything(
    "Gad", GRAPH_VERSION, return_keys=["Class", "Hemisphere"]
)
adj = adj[:100, :100]
adj, inds = get_lcc(adj, return_inds=True)
class_labels = class_labels[inds]
side_labels = side_labels[inds]
n_verts = adj.shape[0]


# %% [markdown]
# #
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
graph_type_labels = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]

GRAPH_VERSION = "2019-12-09"
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


def signal_flow_sort(A, return_inds=False):
    A = A.copy()
    nodes_signal_flow = signal_flow(A)
    sort_inds = np.argsort(nodes_signal_flow)[::-1]
    sorted_A = A[np.ix_(sort_inds, sort_inds)]
    if return_inds:
        return sorted_A, sort_inds
    else:
        return sorted_A


def get_template_mat(A):
    total_synapses = np.sum(A)
    upper_triu_inds = np.triu_indices_from(A, k=1)
    filler = total_synapses / len(upper_triu_inds[0])
    upper_triu_template = np.zeros_like(A)
    upper_triu_template[upper_triu_inds] = filler
    return upper_triu_template


def faq_sort(A, return_inds=False, n_init=1):
    template = get_template_mat(A)
    faq = FastApproximateQAP(
        max_iter=30,
        eps=0.0001,
        init_method="rand",
        n_init=n_init,
        shuffle_input=True,
        gmp=True,
    )
    start = timer()
    faq.fit(template, adj)
    end = timer()
    print(f"FAQ took {(end - start)/60.0} minutes")

    perm_inds = faq.perm_inds_
    sort_A = A[np.ix_(perm_inds, perm_inds)]
    return sort_A


shuffled_triu_outs = []
true_triu_outs = []
n_shuffles = 1

for g, name in zip(graph_types, graph_type_labels):
    print(f"Running for {g} graph")

    adj = load_everything(g, GRAPH_VERSION)
    adj, inds = get_lcc(adj, return_inds=True)
    n_verts = adj.shape[0]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # compare to a fake network with same weights
    for i in range(n_shuffles):
        fake_adj = shuffle_edges(adj)
        fake_adj = faq_sort(fake_adj)
        fake_triu_prop = compute_triu_prop(fake_adj)
        out_dict = {"Proportion": fake_triu_prop, "Graph": name, "Type": "Shuffled"}
        shuffled_triu_outs.append(out_dict)
        print(
            f"{g} shuffled graph sorted proportion in upper triangle: {fake_triu_prop}"
        )

    gridmap(fake_adj, ax=axs[0])
    axs[0].set_title("Shuffled edges")

    adj = faq_sort(adj)
    true_triu_prop = compute_triu_prop(adj)

    out_dict = {"Proportion": true_triu_prop, "Graph": name, "Type": "True"}
    true_triu_outs.append(out_dict)
    print(f"{g} graph sorted proportion in upper triangle: {true_triu_prop}")

    gridmap(adj, ax=axs[1])
    axs[1].set_title("True edges")

    fig.suptitle(f"{name} ({n_verts})", fontsize=40, y=1.02)
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


# #%%
# prop_df = pd.DataFrame()
# prop_df["Proportion"] = np.array(shuffled_triu_props + true_triu_props)
# prop_df["Type"] = np.array(4 * ["Shuffled"] + 4 * ["True"])
# prop_df["Graph"] = np.array(graph_type_labels + graph_type_labels)

# ax = sns.pointplot(
#     data=prop_df, x="Graph", y="Proportion", hue="Type", ci=None, join=False
# )
# ax.set_ylim((0.4, 1.05))
# ax.axhline(0.5, linestyle="--")
# ax.set_ylabel("Proportion upper triangular")
# ax.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.0)
# ax.set_title("")

