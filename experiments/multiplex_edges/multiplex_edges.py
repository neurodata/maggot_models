# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from giskard.utils import powerset
from src.data import load_maggot_graph
from src.io import savefig
from src.visualization import set_theme
from graspologic.utils import remove_loops
from scipy.stats import binom
from scipy.stats import binom_test  # binom_test is depricated
from graspologic.utils import binarize

t0 = time.time()
set_theme()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/multiplex_edges/figs",
        format="pdf",
        **kws,
    )
    savefig(
        name,
        pathname="./maggot_models/experiments/multiplex_edges/figs",
        format="png",
        **kws,
    )


colors = sns.color_palette("deep")

mg = load_maggot_graph()

graph_types = ["ad", "aa", "dd", "da"]
adjs = []
for gt in graph_types:
    adj = mg.to_edge_type_graph(gt).adj
    adj = binarize(adj)
    adjs.append(adj)

# %%

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

edge_types = np.array([et[:] for et in graph_types])
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

#%%
marginal_ps = {}
for i, a in enumerate(adjs):
    p = np.count_nonzero(a) / a.size
    marginal_ps[edge_types[i]] = p

#%%
# null_probs = {}
combo_data["prob"] = combo_data["count"] / (len(adjs[0]) ** 2)
combo_data["null_count"] = -1.0
combo_data["null_prob"] = -1.0
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


n_potential_edges = len(adjs[0]) ** 2
upset_ratio = 0.4
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
divider = make_axes_locatable(ax)
for i, (idx, row) in enumerate(combo_data.iterrows()):
    count = row["count"]
    expected_count = row["null_count"]
    expected_prob = row["null_prob"]
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
    a, b = binom.interval(0.999, n_potential_edges, expected_prob)
    line = ax.plot([i + 0.2, i + 0.2], [a, b], color="black")[0]
    print([a, b])

ax.set(xticks=[], xticklabels=[], xlabel="")
labels = ["Observed", "Null", "99.9% CI"]
ax.legend(handles=[true, null, line], labels=labels)
upset_ax = divider.append_axes("bottom", size=f"{upset_ratio*100}%", sharex=ax)
plot_upset_indicators(combo_data, ax=upset_ax)
ax.set(ylabel="# of edges", ylim=(1, 1e5))  # ax.get_ylim()[-1]
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
stashfig("edge-type-count-upset")
ax.set_yscale("log")
stashfig("edge-type-count-upset-log")

#%%
for idx, row in combo_data.iterrows():
    k = row["count"]
    n = len(adjs[0]) ** 2
    p = row["null_prob"]
    pvalue = binom_test(k, n, p, alternative="two-sided")
    print(pvalue)

#%%

# graph_types = ["Gad", "Gaa", "Gdd", "Gda"]  # "Gs"]
# adjs = []

rows = []
n = len(adjs[0])
n_possible = n * (n - 1)  # ignore the diagonals
for i, fwd_adj in enumerate(adjs):
    fwd_adj = remove_loops(fwd_adj)
    fwd_inds = np.nonzero(fwd_adj)
    p_fwd = len(fwd_inds[0]) / n_possible
    for j, back_adj in enumerate(adjs):
        back_adj = remove_loops(back_adj)
        p_back = np.count_nonzero(back_adj) / n_possible
        # this takes the indices where there were edges in the forward edge type
        # (fwd_inds)
        # then grabs the reversed elements of the backward adj
        # then just counts how many of THOSE were nonzero
        n_reciprocal = np.count_nonzero(back_adj[fwd_inds[::-1]])
        p_reciprocal = n_reciprocal / len(fwd_inds[0])
        null_p_reciprocal = p_fwd * p_back * n_possible / len(fwd_inds[0])
        rows.append(
            {
                "forward": edge_types[i],
                "backward": edge_types[j],
                "n_reciprocal": n_reciprocal,
                "p_reciprocal": p_reciprocal,
                "null_p_reciprocal": null_p_reciprocal,
            }
        )
results = pd.DataFrame(rows)
p_reciprocal = results.pivot(index="forward", columns="backward", values="p_reciprocal")
p_reciprocal = p_reciprocal.reindex(index=edge_types, columns=edge_types)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(
    data=p_reciprocal, annot=True, square=True, cbar=False, cmap="RdBu_r", center=0
)
ax.set_title("Edge reciprocity", fontsize="large", pad=10)
plt.setp(ax.get_yticklabels(), rotation=0)
ax.set(ylabel="Forward", xlabel="Backward")
stashfig("edge-reciprocity")

null_p_reciprocal = results.pivot(
    index="forward", columns="backward", values="null_p_reciprocal"
)
null_p_reciprocal = null_p_reciprocal.reindex(index=edge_types, columns=edge_types)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(
    data=null_p_reciprocal, annot=True, square=True, cbar=False, cmap="RdBu_r", center=0
)
ax.set_title("Null edge reciprocity", fontsize="large", pad=10)
plt.setp(ax.get_yticklabels(), rotation=0)
ax.set(ylabel="Forward", xlabel="Backward")
stashfig("null-edge-reciprocity")


#%%
# extracting neurons that have >1 either 3- or 4- multiplexed edges
adj_multiplexity = adj_tensor.sum(axis=0).astype(int)

from pathlib import Path

out_path = Path("maggot_models/experiments/multiplex_edges/outs")


def extract_plexity_edges(adj_multiplexity, plexity):
    threeplex_mask = adj_multiplexity == plexity
    sources, targets = np.nonzero(threeplex_mask)
    nodes = mg.nodes
    index = nodes.index
    source_ids = index[sources].values.reshape((-1, 1))
    target_ids = index[targets].values.reshape((-1, 1))

    edgelist_arr = np.concatenate((source_ids, target_ids), axis=1)
    edgelist_df = pd.DataFrame(data=edgelist_arr, columns=["source", "target"])

    return edgelist_df


edgelist_df = extract_plexity_edges(adj_multiplexity, 3)
edgelist_df.to_csv(out_path / "threeplex-edges.csv", index=False)

source_nodes = edgelist_df["source"].unique()
pd.Series(source_nodes).to_csv(
    out_path / "threeplex-source-nodes.csv", index=False, header=False
)

target_nodes = edgelist_df["target"].unique()
pd.Series(target_nodes).to_csv(
    out_path / "threeplex-target-nodes.csv", index=False, header=False
)

edgelist_df = extract_plexity_edges(adj_multiplexity, 4)
edgelist_df.to_csv(out_path / "fourplex-edges.csv", index=False)

source_nodes = edgelist_df["source"].unique()
pd.Series(source_nodes).to_csv(
    out_path / "fourplex-source-nodes.csv", index=False, header=False
)

target_nodes = edgelist_df["target"].unique()
pd.Series(target_nodes).to_csv(
    out_path / "fourplex-target-nodes.csv", index=False, header=False
)

#%%
edge_type = "ad"
for edge_type in edge_types:
    conditional_combo_data = combo_data.xs(True, level=edge_type)
    singleton_count = conditional_combo_data.loc[(False, False, False)]["count"]
    total_count = conditional_combo_data["count"].sum()
    singleton_proportion = singleton_count / total_count
    print(edge_type)
    print(f"{singleton_proportion:.2f}")
    print()