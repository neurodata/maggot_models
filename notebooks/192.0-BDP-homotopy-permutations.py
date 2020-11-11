#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from graspologic.utils import binarize
from src.data import load_metagraph
from src.io import savefig
from src.utils import get_paired_inds
from src.visualization import set_theme

FNAME = os.path.basename(__file__)[:-3]
set_theme()


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


exp_dir = Path("maggot_models/experiments/matched_subgraph_omni_cluster/outs/")

# load full metadata with cluster labels
meta_loc = exp_dir / "meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv"
meta = pd.read_csv(meta_loc, index_col=0)

meta["inds"] = range(len(meta))
lp_inds, rp_inds = get_paired_inds(meta, check_in=True)
meta = pd.concat((meta.iloc[lp_inds.values], meta.iloc[rp_inds.values]), axis=0)
meta["inds"] = range(len(meta))
lp_inds, rp_inds = get_paired_inds(meta, check_in=True)

graph_types = ["G", "Gad", "Gaa", "Gdd", "Gda"]
graph_names = dict(
    zip(graph_types, [r"Sum", r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"])
)
adjs = []
adj_dict = {}
mg_dict = {}
for g in graph_types:
    temp_mg = load_metagraph(g)
    temp_mg.reindex(meta.index, use_ids=True, inplace=True)
    temp_mg.meta = meta
    mg_dict[g] = temp_mg
    adj_dict[g] = temp_mg.adj

colors = sns.color_palette("deep", n_colors=len(graph_types))
graph_type_colors = dict(zip(graph_types[1:], colors))
graph_type_colors[graph_types[0]] = colors[-1]


# %%


def _shuffle(mat, row, col):
    subgraph = mat[row][:, col].copy()
    shuffled_subgraph = subgraph.ravel()
    np.random.shuffle(shuffled_subgraph)
    shuffled_subgraph = shuffled_subgraph.reshape(subgraph.shape)
    mat[np.ix_(row, col)] = shuffled_subgraph


def shuffle_subgraph(adj, inds_to_shuffle):
    inds_to_shuffle = np.asarray(inds_to_shuffle)
    adj = adj.copy()
    inds = np.arange(len(adj), dtype=int)
    inds_not_shuffle = np.setdiff1d(inds, inds_to_shuffle)
    _shuffle(adj, inds_not_shuffle, inds_to_shuffle)
    _shuffle(adj, inds_to_shuffle, inds_not_shuffle)
    _shuffle(adj, inds_to_shuffle, inds_to_shuffle)
    return adj


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

shuffle_subgraph(a, [2, 3])


#%%


def diag_indices(length, k=0):
    neg = False
    if k < 0:
        neg = True
    k = np.abs(k)
    inds = (np.arange(length - k), np.arange(k, length))
    if neg:
        return (inds[1], inds[0])
    else:
        return inds


adj = np.zeros((6, 6))
n_pairs = 3
adj[0, 3] = 1
adj[1, 4] = 2
adj[2, 5] = 3
adj[3, 0] = -1
adj[4, 1] = -2
adj[5, 2] = -3
print(adj)

upper_diag_inds = diag_indices(len(adj), k=n_pairs)
print(adj[upper_diag_inds])
lower_diag_inds = diag_indices(len(adj), k=-n_pairs)
print(adj[lower_diag_inds])

#%%

from graspologic.simulations import er_np
from graspologic.plot import heatmap

from graspologic.simulations import sbm

B = np.array([[0.15, 0.05], [0.05, 0.15]])
n_pairs = 25
adj, labels = sbm([n_pairs, n_pairs], B, directed=True, return_labels=True)
upper_diag_inds = diag_indices(len(adj), k=n_pairs)
lower_diag_inds = diag_indices(len(adj), k=-n_pairs)
adj[upper_diag_inds] = 1
adj[lower_diag_inds] = 1

annot = np.array([["Ipsi", "Contra"], ["Contra", "Ipsi"]])
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
ax = axs[0]
sns.heatmap(
    B,
    annot=annot,
    cmap="RdBu_r",
    center=0,
    cbar=False,
    fmt="",
    annot_kws=dict(fontsize="medium"),
    xticklabels=False,
    yticklabels=["Left", "Right"],
    ax=ax,
    square=True,
)
ax.set_title("Block structure")
plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

heatmap_kws = dict(
    cbar=False,
)  # linewidth=0.05, linestyle="-", color="white")
ax = axs[1]
ax.set_title("Observed network")
heatmap(adj, ax=ax, **heatmap_kws)
ax.axvline(n_pairs, linewidth=1, linestyle="--", color="darkgrey")
ax.axhline(n_pairs, linewidth=1, linestyle="--", color="darkgrey")

ax = axs[2]
ax.set_title("Non-homotypic connections")
mask = np.ones_like(adj)
mask[:n_pairs, :n_pairs] = 0
mask[n_pairs:, n_pairs:] = 0
mask[upper_diag_inds] = 0
mask[lower_diag_inds] = 0
heatmap(mask, ax=ax, cmap="RdBu", **heatmap_kws)

ax = axs[3]
ax.set_title("Homotypic connections")
mask = np.zeros_like(adj)
mask[upper_diag_inds] = 1
mask[lower_diag_inds] = 1
heatmap(mask, ax=ax, cmap="RdBu", **heatmap_kws)
stashfig("homotypic-explanation")

#%%


n_pairs = len(lp_inds)


def calc_typic_ps(adj, lp_inds=lp_inds, rp_inds=rp_inds, n_pairs=n_pairs):
    upper_diag_inds = diag_indices(len(adj), k=n_pairs)
    p_upper_diag = adj[upper_diag_inds].mean()
    lower_diag_inds = diag_indices(len(adj), k=-n_pairs)
    p_lower_diag = adj[lower_diag_inds].mean()
    p_diag = (p_upper_diag + p_lower_diag) / 2

    upper_diag_block = adj[lp_inds][:, rp_inds]
    p_upper_upper = upper_diag_block[np.triu_indices_from(upper_diag_block, k=1)].mean()
    p_upper_lower = upper_diag_block[np.tril_indices_from(upper_diag_block, k=1)].mean()

    lower_diag_block = adj[rp_inds][:, lp_inds]
    p_lower_upper = lower_diag_block[np.triu_indices_from(lower_diag_block, k=1)].mean()
    p_lower_lower = lower_diag_block[np.tril_indices_from(lower_diag_block, k=1)].mean()
    p_off_diag = (p_upper_upper + p_upper_lower + p_lower_upper + p_lower_lower) / 4
    return p_diag, p_off_diag


rows = []
n_shuffles = 20

for graph in graph_types:
    adj = adj_dict[graph]
    adj = binarize(adj)
    p_diag, p_off_diag = calc_typic_ps(adj)
    rows.append(
        {
            "p_diag": p_diag,
            "p_off_diag": p_off_diag,
            "version": "True",
            "graph": graph,
        }
    )

    for i in range(n_shuffles):
        shuffle_adj = shuffle_subgraph(adj, rp_inds)
        p_diag, p_off_diag = calc_typic_ps(shuffle_adj)
        rows.append(
            {
                "p_diag": p_diag,
                "p_off_diag": p_off_diag,
                "version": "Shuffled",
                "graph": graph,
            }
        )

results = pd.DataFrame(rows)
results

#%%


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

stripplot_kws = dict(
    x="graph",
    hue="graph",
    hue_order=graph_types,
    palette=graph_type_colors,
)

ax = axs[0]
sns.stripplot(
    y="p_off_diag",
    data=results[results["version"] != "Shuffled"],
    ax=ax,
    marker="+",
    s=10,
    linewidth=2,
    **stripplot_kws,
)
sns.stripplot(
    y="p_off_diag",
    data=results[results["version"] == "Shuffled"],
    ax=ax,
    alpha=0.5,
    jitter=0.35,
    **stripplot_kws,
)

ax = axs[1]
sns.stripplot(
    y="p_diag",
    data=results[results["version"] != "Shuffled"],
    ax=ax,
    marker="+",
    s=10,
    linewidth=2,
    **stripplot_kws,
)
sns.stripplot(
    y="p_diag",
    data=results[results["version"] == "Shuffled"],
    ax=ax,
    alpha=0.5,
    jitter=0.35,
    **stripplot_kws,
)


ax = axs[2]
results["ratio"] = results["p_diag"] / results["p_off_diag"]
sns.stripplot(
    y="ratio",
    data=results[results["version"] != "Shuffled"],
    ax=ax,
    marker="+",
    s=10,
    linewidth=2,
    **stripplot_kws,
)
sns.stripplot(
    y="ratio",
    data=results[results["version"] == "Shuffled"],
    ax=ax,
    alpha=0.5,
    jitter=0.35,
    **stripplot_kws,
)
ax.axhline(1, color="black", linestyle=":", linewidth=2)

new_ticklabels = list(map(graph_names.get, graph_types))
for ax in axs:
    ax.set_xticklabels(new_ticklabels, rotation=45)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(graph_type_colors[graph_types[i]])
    ax.get_legend().remove()
    ax.set(xlabel="")
    # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
axs[1].set_xlabel("Graph")
axs[0].set_ylabel(r"Non-homotypic probability ($p_{NH}$)")
axs[1].set_ylabel(r"Homotypic probability ($p_H$)")
axs[2].set_ylabel(r"Homotypy ratio $(p_H / p_{NH})$")
yticks = list(results[results["version"] == "True"]["ratio"].unique())
yticks.append(1)
axs[2].set_yticks(yticks)
axs[2].yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}")


true_line = Line2D([0], [0], markeredgewidth=2, marker="+", linewidth=0, color="black")
shuffle_line = Line2D(
    [0], [0], markeredgewidth=0.05, marker="o", linewidth=0, color="black", markersize=5
)

axs[0].legend([true_line, shuffle_line], ["True", "Shuffled"])
plt.tight_layout()

stashfig("homotypy-by-color")


#%%

rows = []
n_shuffles = 1
level = 7
label_groups = meta[f"lvl{level}_labels_side"].unique()
for graph in graph_types:
    adj = adj_dict[graph]
    adj = binarize(adj)
    p_diag, p_off_diag = calc_typic_ps(adj)
    rows.append(
        {
            "p_diag": p_diag,
            "p_off_diag": p_off_diag,
            "version": "True",
            "graph": graph,
        }
    )

    for _ in range(n_shuffles):
        shuffle_adj = adj.copy()
        for label_group in label_groups:
            inds = meta[meta[f"lvl{level}_labels_side"] == label_group]["inds"].values
            shuffle_adj = shuffle_subgraph(shuffle_adj, rp_inds)
        p_diag, p_off_diag = calc_typic_ps(shuffle_adj)
        rows.append(
            {
                "p_diag": p_diag,
                "p_off_diag": p_off_diag,
                "version": "Shuffled",
                "graph": graph,
            }
        )

results = pd.DataFrame(rows)
results

#%%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

stripplot_kws = dict(
    x="graph",
    hue="graph",
    hue_order=graph_types,
    palette=graph_type_colors,
)

ax = axs[0]
sns.stripplot(
    y="p_off_diag",
    data=results[results["version"] != "Shuffled"],
    ax=ax,
    marker="+",
    s=10,
    linewidth=2,
    **stripplot_kws,
)
sns.stripplot(
    y="p_off_diag",
    data=results[results["version"] == "Shuffled"],
    ax=ax,
    alpha=0.5,
    jitter=0.35,
    **stripplot_kws,
)

ax = axs[1]
sns.stripplot(
    y="p_diag",
    data=results[results["version"] != "Shuffled"],
    ax=ax,
    marker="+",
    s=10,
    linewidth=2,
    **stripplot_kws,
)
sns.stripplot(
    y="p_diag",
    data=results[results["version"] == "Shuffled"],
    ax=ax,
    alpha=0.5,
    jitter=0.35,
    **stripplot_kws,
)


ax = axs[2]
results["ratio"] = results["p_diag"] / results["p_off_diag"]
sns.stripplot(
    y="ratio",
    data=results[results["version"] != "Shuffled"],
    ax=ax,
    marker="+",
    s=10,
    linewidth=2,
    **stripplot_kws,
)
sns.stripplot(
    y="ratio",
    data=results[results["version"] == "Shuffled"],
    ax=ax,
    alpha=0.5,
    jitter=0.35,
    **stripplot_kws,
)
ax.axhline(1, color="black", linestyle=":", linewidth=2)

new_ticklabels = list(map(graph_names.get, graph_types))
for ax in axs:
    ax.set_xticklabels(new_ticklabels, rotation=45)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(graph_type_colors[graph_types[i]])
    ax.get_legend().remove()
    ax.set(xlabel="")
    # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
axs[1].set_xlabel("Graph")
axs[0].set_ylabel(r"Non-homotypic probability ($p_{NH}$)")
axs[1].set_ylabel(r"Homotypic probability ($p_H$)")
axs[2].set_ylabel(r"Homotypy ratio $(p_H / p_{NH})$")
yticks = list(results[results["version"] == "True"]["ratio"].unique())
yticks.append(1)
axs[2].set_yticks(yticks)
axs[2].yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}")


true_line = Line2D([0], [0], markeredgewidth=2, marker="+", linewidth=0, color="black")
shuffle_line = Line2D(
    [0], [0], markeredgewidth=0.05, marker="o", linewidth=0, color="black", markersize=5
)

axs[0].legend([true_line, shuffle_line], ["True", "Shuffled"])
plt.tight_layout()

stashfig("homotypy-by-color-classed")
# TODO this result is surprising to me
# see if the clusters are more likely than not to be connected to corresponding
# cluster on the other side.

#%%


from graspologic.models import SBMEstimator

level = 7
uni_labels = np.unique(meta[f"lvl{level}_labels"])
uni_labels_left = [ul + "L" for ul in uni_labels]
uni_labels_right = [ul + "R" for ul in uni_labels]

uni_labels_full = np.concatenate((uni_labels_left, uni_labels_right))
uni_labels_map = dict(zip(uni_labels_full, np.arange(len(uni_labels_full))))

labels = meta[f"lvl{level}_labels_side"]
int_labels = np.vectorize(uni_labels_map.get)(labels)

adj = adj_dict["G"]

sbme = SBMEstimator(directed=True, loops=True)
sbme.fit(binarize(adj), y=int_labels)

#%%

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
sns.heatmap(
    sbme.block_p_,
    cmap="RdBu_r",
    center=0,
    ax=ax,
    xticklabels=False,
    yticklabels=False,
    square=True,
    cbar_kws=dict(shrink=0.7),
)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
sns.heatmap(
    np.log10(sbme.block_p_ + 1),
    cmap="RdBu_r",
    center=0,
    ax=ax,
    xticklabels=False,
    yticklabels=False,
    square=True,
    cbar_kws=dict(shrink=0.7),
)
#%%

# def calc_typic_ps(adj, lp_inds=lp_inds, rp_inds=rp_inds, n_pairs=n_pairs):
#     upper_diag_inds = diag_indices(len(adj), k=n_pairs)
#     p_upper_diag = adj[upper_diag_inds].mean()
#     lower_diag_inds = diag_indices(len(adj), k=-n_pairs)
#     p_lower_diag = adj[lower_diag_inds].mean()
#     p_diag = (p_upper_diag + p_lower_diag) / 2

#     upper_diag_block = adj[lp_inds][:, rp_inds]
#     p_upper_upper = upper_diag_block[np.triu_indices_from(upper_diag_block, k=1)].mean()
#     p_upper_lower = upper_diag_block[np.tril_indices_from(upper_diag_block, k=1)].mean()

#     lower_diag_block = adj[rp_inds][:, lp_inds]
#     p_lower_upper = lower_diag_block[np.triu_indices_from(lower_diag_block, k=1)].mean()
#     p_lower_lower = lower_diag_block[np.tril_indices_from(lower_diag_block, k=1)].mean()
#     p_off_diag = (p_upper_upper + p_upper_lower + p_lower_upper + p_lower_lower) / 4
#     return p_diag, p_off_diag


B = sbme.block_p_
n_pairs = len(uni_labels_left)
upper_diag_inds = diag_indices(len(B), k=n_pairs)

lp_inds = np.arange(n_pairs)
rp_inds = np.arange(n_pairs) + n_pairs

upper_diag_block = B[lp_inds][:, rp_inds]
upper_upper_vals = upper_diag_block[np.triu_indices_from(upper_diag_block, k=1)]
upper_lower_vals = upper_diag_block[np.tril_indices_from(upper_diag_block, k=1)]
non_homotypic_vals = np.concatenate((upper_upper_vals, upper_lower_vals))
homotypic_vals = B[upper_diag_inds]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

sns.distplot(
    np.log10(non_homotypic_vals + 0.00001), label="Non-homotypic", rug=True, ax=ax
)
sns.distplot(np.log10(homotypic_vals + 0.00001), label="Homotypic", rug=True, ax=ax)
ax.legend()
