# %% [markdown]
# #
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.embed import LaplacianSpectralEmbed
from graspy.plot import heatmap, pairplot
from graspy.simulations import sbm
from graspy.utils import get_lcc

from src.data import load_everything
from src.utils import savefig


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
SAVEFIGS = True
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

BRAIN_VERSION = "2019-09-18-v2"
GRAPH_TYPES = ["G", "Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [
    "Summed 4 types",
    r"A $\to$ D",
    r"A $\to$ A",
    r"D $\to$ D",
    r"D $\to$ A",
]


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


# %% [markdown]
# #
sns.set_context("talk", font_scale=1.2)

fig, axs = plt.subplots(4, 5, figsize=(20, 20))
for i in range(len(GRAPH_TYPES)):
    g_type = GRAPH_TYPES[i]
    g_type_label = GRAPH_TYPE_LABELS[i]

    adj = load_everything(g_type, version=BRAIN_VERSION)

    in_sum = np.sort(adj.sum(axis=0))
    out_sum = np.sort(adj.sum(axis=1))

    ax = sns.scatterplot(
        x=range(len(in_sum)), y=in_sum + 1, s=10, linewidth=0, ax=axs[0, i]
    )
    ax.set(yscale="log")
    ax.set_xticklabels([])

    ax = sns.scatterplot(
        x=range(len(out_sum)), y=out_sum + 1, s=10, linewidth=0, ax=axs[1, i]
    )
    ax.set(yscale="log")
    ax.set_xticklabels([])

    in_degree = np.sort(np.count_nonzero(adj, axis=0))
    out_degree = np.sort(np.count_nonzero(adj, axis=1))

    ax = sns.scatterplot(
        x=range(len(in_degree)), y=in_degree + 1, s=10, linewidth=0, ax=axs[2, i]
    )
    ax.set(yscale="log")
    ax.set_xticklabels([])

    ax = sns.scatterplot(
        x=range(len(out_degree)), y=out_degree + 1, s=10, linewidth=0, ax=axs[3, i]
    )
    ax.set(yscale="log")

for i, g in enumerate(GRAPH_TYPE_LABELS):
    axs[0, i].set_title(g)

plot_labels = [
    "# input synapses + 1",
    "# output synapses + 1",
    "# input edges + 1",
    "# output edges + 1",
]

for i, n in enumerate(plot_labels):
    axs[i, 0].set_ylabel(n)

for i in range(len(GRAPH_TYPE_LABELS)):
    axs[3, i].set_xlabel("Neuron index")

plt.suptitle("Degree and synapse count sequences", fontsize=35, y=1.02)
plt.tight_layout()
stashfig("degree-synapse-sequences")


# %%
fig, axs = plt.subplots(4, 5, figsize=(20, 20))
for i in range(len(GRAPH_TYPES)):
    g_type = GRAPH_TYPES[i]
    g_type_label = GRAPH_TYPE_LABELS[i]

    adj = load_everything(g_type, version=BRAIN_VERSION)

    in_sum = np.sort(adj.sum(axis=0))
    out_sum = np.sort(adj.sum(axis=1))

    ax = sns.distplot(in_sum, ax=axs[0, i])
    # ax.set(yscale="log")
    ax.set_xticklabels([])

    ax = sns.distplot(out_sum, ax=axs[1, i])
    # ax.set(yscale="log")
    ax.set_xticklabels([])

    in_degree = np.sort(np.count_nonzero(adj, axis=0))
    out_degree = np.sort(np.count_nonzero(adj, axis=1))

    ax = sns.distplot(in_degree, ax=axs[2, i])
    # ax.set(yscale="log")
    ax.set_xticklabels([])

    ax = sns.distplot(out_degree, ax=axs[3, i])
    # ax.set(yscale="log")

for i, g in enumerate(GRAPH_TYPE_LABELS):
    axs[0, i].set_title(g)

plot_labels = [
    "# input synapses + 1",
    "# output synapses + 1",
    "# input edges + 1",
    "# output edges + 1",
]

# for i, n in enumerate(plot_labels):
#     axs[i, 0].set_ylabel(n)

# for i in range(len(GRAPH_TYPE_LABELS)):
#     axs[3, i].set_xlabel("Neuron index")

plt.suptitle("Degree and synapse count sequences", fontsize=35, y=1.02)
plt.tight_layout()
# stashfig("degree-synapse-sequences")


# %%
from powerlaw import plot_ccdf, plot_cdf, plot_pdf


fig, axs = plt.subplots(4, 5, figsize=(20, 20), sharex=True)
for i in range(len(GRAPH_TYPES)):
    g_type = GRAPH_TYPES[i]
    g_type_label = GRAPH_TYPE_LABELS[i]

    adj = load_everything(g_type, version=BRAIN_VERSION)

    in_sum = np.sort(adj.sum(axis=0))
    out_sum = np.sort(adj.sum(axis=1))

    ax = plot_ccdf(in_sum, ax=axs[0, i])
    # ax.set(yscale="log")
    ax.set_xticklabels([])

    ax = plot_ccdf(out_sum, ax=axs[1, i])
    # ax.set(yscale="log")
    ax.set_xticklabels([])

    in_degree = np.sort(np.count_nonzero(adj, axis=0))
    out_degree = np.sort(np.count_nonzero(adj, axis=1))

    ax = plot_ccdf(in_degree, ax=axs[2, i])
    # ax.set(yscale="log")
    ax.set_xticklabels([])

    ax = plot_ccdf(out_degree, ax=axs[3, i])
    # ax.set(yscale="log")

for i, g in enumerate(GRAPH_TYPE_LABELS):
    axs[0, i].set_title(g)

plot_labels = [
    "# input synapses",
    "# output synapses + 1",
    "# input edges + 1",
    "# output edges + 1",
]
for i, n in enumerate(plot_labels):
    axs[i, 0].set_ylabel(n)

plt.tight_layout()

# %%
dfs = []
for g_type, name in zip(GRAPH_TYPES, GRAPH_TYPE_LABELS):
    adj = load_everything(g_type, version=BRAIN_VERSION)

    temp_df = pd.DataFrame()
    temp_df["Prob"] = adj.sum(axis=0)
    temp_df["Type"] = "In sum"
    temp_df["Graph"] = name
    dfs.append(temp_df)

    temp_df = pd.DataFrame()
    temp_df["Prob"] = adj.sum(axis=1)
    temp_df["Type"] = "In sum"
    temp_df["Graph"] = name
    dfs.append(temp_df)

    temp_df = pd.DataFrame()
    temp_df["Prob"] = np.count_nonzero(adj, axis=0)
    temp_df["Type"] = "In sum"
    temp_df["Graph"] = name
    dfs.append(temp_df)

    temp_df = pd.DataFrame()
    temp_df["Prob"] = np.count_nonzero(adj, axis=1)
    temp_df["Type"] = "In sum"
    temp_df["Graph"] = name
    dfs.append(temp_df)

sum_df = pd.concat(dfs)

fg = sns.FacetGrid(sum_df, row="Graph", col="Prob")
fg.map(plot_ccdf)

#%%
from powerlaw import Fit

fig, axs = plt.subplots(5, 4, figsize=(20, 20), sharex=True, sharey=True)


def in_sum(adj):
    return np.sum(adj, axis=0)


def out_sum(adj):
    return np.sum(adj, axis=1)


def in_degree(adj):
    return np.count_nonzero(adj, axis=0)


def out_degree(adj):
    return np.count_nonzero(adj, axis=1)


funcs = [in_sum, out_sum, in_degree, out_degree]

for i in range(len(GRAPH_TYPES)):
    g_type = GRAPH_TYPES[i]
    g_type_label = GRAPH_TYPE_LABELS[i]
    adj = load_everything(g_type, version=BRAIN_VERSION)
    for j in range(len(funcs)):
        vals = funcs[j](adj)
        vals = vals[vals > 0]
        plot_ccdf(data=vals, ax=axs[i, j])
        results = Fit(vals)
        line = results.power_law.plot_ccdf(
            ax=axs[i, j],
            c="r",
            shift_by="original_data",
            linestyle="--",
            label="Power law",
        )
        results.lognormal.plot_ccdf(
            ax=axs[i, j],
            c="g",
            shift_by="original_data",
            linestyle="--",
            label="Lognormal",
        )


func_labels = [
    "# input synapses",
    "# output synapses",
    "# input edges",
    "# output edges",
]
[a.set_ylabel(g) for a, g in zip(axs[:, 0], GRAPH_TYPE_LABELS)]
[a.set_xlabel(g) for a, g in zip(axs[4, :], func_labels)]

plt.tight_layout()
plt.legend()
plt.suptitle("Degree and synapse survival functions", fontsize=35, y=1.02)
stashfig("degree-sum-ccdfs")
# r"P(X $\geq$ x)"

