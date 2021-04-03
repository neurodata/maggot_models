# %% [markdown]
# ##
import pandas as pd
import numpy as np

# connector_loc = "maggot_models/data/processed/2020-06-10/connectors.csv"
# connectors = pd.read_csv(connector_loc, index_col=0)

# sizes = connectors.groupby(
#     ["connector_id", "presynaptic_type", "postsynaptic_type"]
# ).size()
# sizes.name = "count"

# sizes = sizes.reset_index()

# sizes["subconnector_type"] = (
#     sizes["presynaptic_type"] + "-" + sizes["postsynaptic_type"]
# )
# n_postsynaptic = sizes.groupby("connector_id")["count"].sum()
# sizes["n_postsynaptic"] = sizes["connector_id"].map(n_postsynaptic)
# sizes["prop"] = sizes["count"] / sizes["n_postsynaptic"]

# connector_entrop = {}
# for connector_id, group in sizes.groupby("connector_id"):
#     entrop = -np.sum(group["prop"] * np.log(group["prop"]) / np.log(4))
#     connector_entrop[connector_id] = entrop

# %% [markdown]
# ##

from src.data import load_metagraph

mg = load_metagraph("G")
mg.make_lcc()

edge_types = ["Gaa", "Gad", "Gda", "Gdd"]  # "Gs"]
adjs = []
for g in edge_types:
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

bool_edges_by_type = edges_by_type > 0
edge_combos, counts = np.unique(bool_edges_by_type, axis=0, return_counts=True)
edge_types = np.array([et[1:] for et in edge_types])

# #%%
# edge_types_broadcast = np.repeat(edge_types, len(edge_combos), axis=0)

# edge_type_labels = edge_types_broadcast[edge_combos]
#%%
data = pd.DataFrame(data=edge_combos, columns=edge_types)
data["count"] = counts
data = data.set_index(keys=list(edge_types)[::-1])["count"]
data = data.sort_values(ascending=False)
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
    n_cats = data.index.nlevels

    idx = np.flatnonzero(data.index.to_frame()[data.index.names].values)
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
    tick_axis.set_ticklabels(data.index.names, rotation=0 if horizontal else -90)
    # ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    if not horizontal:
        ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_ylim((-height_pad, n_cats - 1 + height_pad))


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from src.io import savefig


FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, format="png", dpi=300, **kws)


upset_ratio = 0.4
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
divider = make_axes_locatable(ax)
for i, (idx, count) in enumerate(data.iteritems()):
    ax.bar(i, count, color="grey", width=0.4)
ax.set(xticks=[], xticklabels=[], xlabel="")
upset_ax = divider.append_axes("bottom", size=f"{upset_ratio*100}%", pad=0, sharex=ax)
plot_upset_indicators(data, ax=upset_ax)
ax.set(ylabel="# of edges")
stashfig("edge-type-count-upset")
ax.set_yscale("log")
stashfig("edge-type-count-upset-log")
