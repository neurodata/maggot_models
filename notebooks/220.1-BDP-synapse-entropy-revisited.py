# %% [markdown]
# ##
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from giskard.utils import powerset
from src.data import load_metagraph
from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, format="png", dpi=300, **kws)


colors = sns.color_palette("deep")

mg = load_metagraph("G")
mg.make_lcc()

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]  # "Gs"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g)
    # this line is important, to make the graphs aligned
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)


# %% [markdown]
# ##

adj_tensor = np.array(adjs)
sums = adj_tensor.sum(axis=0)
mask = sums > 0
edges_by_type = []
for a in adjs:
    edges = a[mask]
    edges_by_type.append(edges)
edges_by_type = np.array(edges_by_type).T
edges_by_type


#%%

edge_types = np.array([et[1:] for et in graph_types])
edge_type_combos = powerset(edge_types, ignore_empty=True)

bool_edges_by_type = edges_by_type > 0
bool_edges_by_type = pd.DataFrame(data=bool_edges_by_type, columns=edge_types)
combo_data = bool_edges_by_type.groupby(list(edge_types)).size()
combo_data.name = "count"
combo_data = combo_data.to_frame()

for edge_type_combo in edge_type_combos:
    bool_edge_type_combo = tuple(np.isin(edge_types, edge_type_combo))
    count = combo_data.loc[bool_edge_type_combo]
# edge_combos, counts = np.unique(bool_edgs_by_type, axis=0, return_counts=True)

# edge_combos = powerset(edge_types, ignore_empty=True)

#%%

combo_data["inv_n_in_combo"] = 0
flat_rank = 2 ** np.arange(len(edge_types))
for edge_type_combo, row in combo_data.iterrows():
    rank = flat_rank[np.array(edge_type_combo)]
    rank_sum = rank.sum()
    n_in_combo = len(rank)
    row["inv_n_in_combo"] = -n_in_combo

combo_data = (
    combo_data.reset_index()
    .sort_values(["inv_n_in_combo"] + list(edge_types), ascending=False)
    .set_index(list(edge_types))
)
combo_data = combo_data.drop("inv_n_in_combo", axis=1)
combo_data

#%% [markdown]
# ## A simple null model

marginal_ps = {}
for i, a in enumerate(adjs):
    p = np.count_nonzero(a) / a.size
    marginal_ps[edge_types[i]] = p

#%%
# null_probs = {}
combo_data["null_prob"] = -1.0
combo_data["null_count"] = -1.0
for bool_edge_type_combo in combo_data.index:
    edge_type_combo = edge_types[np.array(bool_edge_type_combo)]
    p = 1
    for edge_type in edge_types:
        if edge_type in edge_type_combo:
            p *= marginal_ps[edge_type]
        else:
            p *= 1 - marginal_ps[edge_type]
    combo_data.loc[bool_edge_type_combo, "null_prob"] = p
    combo_data.loc[bool_edge_type_combo, "null_count"] = p * (len(adjs[0]) ** 2)

combo_data

# %%


def plot_upset_indicators(
    intersections,
    ax=None,
    facecolor="black",
    element_size=None,
    with_lines=True,
    horizontal=True,
    height_pad=0.7,
):
    # REF: https://github.com/jnothman/UpSetPlot/blob/e6f66883e980332452041cd1a6ba986d6d8d2ae5/upsetplot/plotting.py#L428
    """Plot the matrix of intersection indicators onto ax"""
    data = intersections
    index = data.index
    index = index.reorder_levels(index.names[::-1])
    n_cats = index.nlevels

    idx = np.flatnonzero(index.to_frame()[index.names].values)[::-1]
    c = np.array(["lightgrey"] * len(data) * n_cats, dtype="O")
    c[idx] = facecolor
    x = np.repeat(np.arange(len(data)), n_cats)
    y = np.tile(np.arange(n_cats), len(data))
    if element_size is not None:
        s = (element_size * 0.35) ** 2
    else:
        # TODO: make s relative to colw
        s = 200
    ax.scatter(x, y, c=c.tolist(), linewidth=0, s=s)

    if with_lines:
        line_data = (
            pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
        )
        ax.vlines(
            line_data.index.values,
            line_data["min"],
            line_data["max"],
            lw=2,
            colors=facecolor,
        )

    tick_axis = ax.yaxis
    tick_axis.set_ticks(np.arange(n_cats))
    tick_axis.set_ticklabels(index.names, rotation=0 if horizontal else -90)
    # ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    if not horizontal:
        ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_ylim((-height_pad, n_cats - 1 + height_pad))


upset_ratio = 0.4
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
divider = make_axes_locatable(ax)
for i, (idx, row) in enumerate(combo_data.iterrows()):
    count = row["count"]
    expected_count = row["null_count"]
    true = ax.bar(
        i - 0.2, count, color=colors[0], width=0.3, edgecolor=colors[0], linewidth=1
    )
    null = ax.bar(
        i + 0.2,
        expected_count,
        color="grey",
        width=0.3,
        hatch="//",
        facecolor="whitesmoke",
        edgecolor="grey",
        linewidth=1,
    )
ax.set(xticks=[], xticklabels=[], xlabel="")
labels = ["Observed", "Null"]
ax.legend(handles=[true, null], labels=labels)
upset_ax = divider.append_axes("bottom", size=f"{upset_ratio*100}%", sharex=ax)
plot_upset_indicators(combo_data, ax=upset_ax)
ax.set(ylabel="# of edges", ylim=(1, 1e5))  # ax.get_ylim()[-1]
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
stashfig("edge-type-count-upset")
ax.set_yscale("log")
stashfig("edge-type-count-upset-log")

#%%
# TODO
# do the binomial tests

# for i, (idx, row) in enumerate(combo_data.iterrows()):
import networkx as nx

rows = []
for adj, edge_type in zip(adjs, edge_types):
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    for cc in nx.strongly_connected_components(g):
        size = len(cc)
        rows.append({"size": size, "edge_type": edge_type})
results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
jitter = 0.2
sns.stripplot(data=results, x="edge_type", y="size", s=4, jitter=jitter)
ax.set_yscale("log")
ax.set(ylim=(0.5, ax.get_ylim()[1]), ylabel="Size of weakly connected component")
nums = [1, 2]
for i, edge_type in enumerate(edge_types):
    for num in nums:
        n_in_bin = len(
            results[(results["edge_type"] == edge_type) & (results["size"] == num)]
        )
        if n_in_bin > 0:
            ax.text(i + jitter + 0.02, num, n_in_bin, va='center')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
stashfig("weak-cc-size-distributions")