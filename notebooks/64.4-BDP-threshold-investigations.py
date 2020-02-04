# %% [markdown]
# # Imports
import os
import random
from operator import itemgetter
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.io import savefig, saveobj, saveskels
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True
BRAIN_VERSION = "2020-01-29"

sns.set_context("talk")

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


def threshold_sweep(edgelist_df, start=0, stop=0.3, steps=30):
    threshs = np.linspace(start, stop, steps)
    rows = []
    for threshold in threshs:
        thresh_df = edgelist_df[edgelist_df["max_norm_weight"] > threshold].copy()
        n_left = thresh_df["edge pair ID"].nunique()
        if n_left > 0:
            p_sym = (
                thresh_df[thresh_df["edge pair counts"] == 2]["edge pair ID"].nunique()
                / n_left
            )
        else:
            p_sym = np.nan
        p_edges_left = len(thresh_df) / len(edgelist_df)
        p_syns_left = thresh_df["syn_weight"].sum() / edgelist_df["syn_weight"].sum()
        row = {
            "threshold": threshold,
            "Prop. paired edges symmetric": p_sym,
            "Prop. edges left": p_edges_left,
            "Prop. synapses left": p_syns_left,
        }
        rows.append(row)
    return pd.DataFrame(rows)


data = np.array(
    [
        [0.1, 1, 2, 3],
        [0.1, 1, 2, 2],
        [0.01, 2, 2, 3],
        [0.01, 2, 2, 4],
        [0.01, 3, 1, 4],
        [0.02, 4, 1, 5],
    ]
)
fake_edgelist = pd.DataFrame(
    columns=["max_norm_weight", "edge pair ID", "edge pair counts", "syn_weight"],
    data=data,
)
out_df = threshold_sweep(fake_edgelist)

#%%
graph_type = "Gad"
remove_pdiff = True
input_thresh = 0

mg = load_metagraph(graph_type, BRAIN_VERSION)

use_subset = True
if use_subset:
    meta = mg.meta
    meta["Original index"] = range(len(meta))
    subset_classes = [
        # "KC",
        "MBIN",
        "mPN",
        "tPN",
        "APL",
        "MBON",
        "uPN",
        "vPN",
        "mPN; FFN",
        "bLN",
        "cLN",
        "pLN",
        # "sens",
    ]
    subset_inds = meta[meta["Class 1"].isin(subset_classes)]["Original index"]
    mg.reindex(subset_inds)

print(f"{len(mg.to_edgelist())} original edges")


def preprocess(mg, remove_pdiff=False):
    n_original_verts = len(mg)

    if remove_pdiff:
        keep_inds = np.where(~mg["is_pdiff"])[0]
        print(sum(keep_inds))
        mg = mg.reindex(keep_inds)
        print(f"Removed {n_original_verts - len(mg.meta)} partially differentiated")

    mg.verify(n_checks=100000, version=BRAIN_VERSION, graph_type=graph_type)

    edgelist_df = mg.to_edgelist(remove_unpaired=True)
    edgelist_df.rename(columns={"weight": "syn_weight"}, inplace=True)
    edgelist_df["norm_weight"] = (
        edgelist_df["syn_weight"] / edgelist_df["target dendrite_input"]
    )

    max_pair_edges = edgelist_df.groupby("edge pair ID", sort=False)["syn_weight"].max()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    edgelist_df["max_syn_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
        edge_max_weight_map
    )
    # print(np.unique(edgelist_df["edge pair ID"], return_counts=True))
    # temp_df = edgelist_df[edgelist_df["edge pair ID"] == 0]
    # edgelist_df.loc[temp_df.index, "max_syn_weight"] = temp_df["syn_weight"]

    max_pair_edges = edgelist_df.groupby("edge pair ID", sort=False)[
        "norm_weight"
    ].max()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    edgelist_df["max_norm_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
        edge_max_weight_map
    )
    # temp_df = edgelist_df[edgelist_df["edge pair ID"] == 0]
    # edgelist_df.loc[temp_df.index, "max_norm_weight"] = temp_df["norm_weight"]
    return edgelist_df


edgelist_df = preprocess(mg, remove_pdiff=remove_pdiff)
n_paired_edges = len(edgelist_df)

meta = mg.meta
remove_inds = meta[meta["dendrite_input"] < input_thresh].index
edgelist_df = edgelist_df[~edgelist_df["target"].isin(remove_inds)]
n_left_edges = len(edgelist_df)

print(f"{n_left_edges} edges after preprocessing")

thresh_result_df = threshold_sweep(edgelist_df)

plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.lineplot(
    data=thresh_result_df, x="threshold", y="Prop. paired edges symmetric", ax=ax
)
remove_spines(ax)
ax_right = plt.twinx(ax)
sns.lineplot(
    data=thresh_result_df,
    x="threshold",
    y="Prop. edges left",
    ax=ax_right,
    color="orange",
    label="Edges",
)
remove_spines(ax_right)
sns.lineplot(
    data=thresh_result_df,
    x="threshold",
    y="Prop. synapses left",
    ax=ax_right,
    color="green",
    label="Synapses",
)
ax_right.set_ylabel("Prop. left")
ax.set_title(
    f"Min dendridic input = {input_thresh}"
    + f" (removed {n_paired_edges - n_left_edges} edges) "
    + f"subset = {use_subset} "
    + f"remove pdiff = {remove_pdiff} "
)
pad = 0.03
ax.set_ylim((0 - pad, 1 + pad))
ax_right.set_ylim((0 - pad, 1 + pad))
plt.legend(bbox_to_anchor=(1.08, 1), loc=2, borderaxespad=0.0)
savename = (
    f"threshold-sweep-{graph_type}-min-dend{input_thresh}"
    + f"-rem-pdiff{remove_pdiff}-subset{use_subset}"
)
stashfig(savename)
# %% [markdown]
# # get number of inputs to kenyon cells
# # just list the number of connections onto each kenyon cell, by claw number
plt.style.use("seaborn-whitegrid")
sns.set_context("talk")
graph_type = "Gad"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
edgelist_df = mg.to_edgelist()
adj = mg.adj
class_labels = mg.meta["Class 1"].fillna("")
subclass_labels = mg.meta["Class 2"].fillna("")
kc_inds = np.where(class_labels == "KC")[0]
for i in range(1, 7):
    name = f"{i}claw"
    sub_edgelist_df = edgelist_df[edgelist_df["target Class 2"] == name]
    ids = sub_edgelist_df["target"].unique()
    # fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    fig = plt.figure(figsize=(20, 7))
    ax = plt.subplot2grid((1, 5), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((1, 5), (0, 4), colspan=1)
    sns.stripplot(data=sub_edgelist_df, y="weight", x="target", ax=ax, order=ids)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_title(name + " input weights")
    # remove_spines(ax, keep_corner=True)
    mins = []
    ticks = ax.get_xticks()
    color = sns.color_palette("deep", desat=1, n_colors=2)[1]
    for j, cell_id in enumerate(ids):
        cell_df = sub_edgelist_df[sub_edgelist_df["target"] == cell_id]
        cell_df = cell_df.sort_values("weight", ascending=False)
        cell_df = cell_df.iloc[:i, :]
        min_max_weight = cell_df["weight"].min()
        ax.text(
            j,
            min_max_weight,
            min_max_weight,
            fontsize="small",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        mins.append(min_max_weight)
    sns.violinplot(
        mins, ax=ax2, orient="v", inner="quart", color=color, alpha=0.8, saturation=1
    )
    median = np.median(mins)
    ax2.text(
        0,
        median,
        f"{median:.0f}",
        horizontalalignment="center",
        verticalalignment="center",
        backgroundcolor=color,
        alpha=0.8,
    )
    # ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.set_ylim(ax.get_ylim())
    ax2.yaxis.set_ticks([])
    ax2.set_title("")
    stashfig(name + "-input-weights")

name = "all KC"
kc_edgelist_df = edgelist_df[edgelist_df["target Class 1"] == "KC"]
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
sns.stripplot(
    data=kc_edgelist_df,
    y="weight",
    x="target Class 2",
    ax=ax,
    order=[f"{i}claw" for i in range(1, 7)],
    jitter=0.45,
)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.set_title(name + " input weights")
# remove_spines(ax, keep_corner=True)
stashfig("all-kc-input-weights")

# %% [markdown]
# # plot the distribution of # of dendritic / axonic inputs

graph_type = "Gad"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
meta = mg.meta
meta.loc[input_df.index, "dendrite_input"] = input_df[" dendrite_inputs"]
meta.loc[input_df.index, "axon_input"] = input_df[" axon_inputs"]


def filter(string):
    string = string.replace("akira", "")
    string = string.replace("Lineage", "")
    string = string.replace("*", "")
    string = string.strip("_")
    return string


lineages = meta["lineage"]
lineages = np.vectorize(filter)(lineages)
meta["lineage"] = lineages


n_rows = 6
fig, axs = plt.subplots(n_rows, 1, figsize=(15, 30), sharey=True)
uni_lineages = np.unique(meta["lineage"])
n_lineages = len(uni_lineages)
n_per_row = n_lineages // n_rows
for i in range(n_rows):
    ax = axs[i]
    temp_lineages = uni_lineages[i * n_per_row : (i + 1) * n_per_row]
    temp_df = meta[meta["lineage"].isin(temp_lineages)]
    sns.stripplot(
        data=temp_df, x="lineage", y="dendrite_input", ax=ax, palette="deep", jitter=0.4
    )
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel("")
    remove_spines(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_tick_params(length=0)
plt.tight_layout()
stashfig("all-lineage-dendrite-input")

# %% [markdown]
# # Plot this but by cell class
n_rows = 3
uni_lineages = np.unique(meta["Merge Class"])
n_lineages = len(uni_lineages)
n_per_row = n_lineages // n_rows
fig, axs = plt.subplots(n_rows + 1, 1, figsize=(15, 30), sharey=True)
for i in range(n_rows):
    ax = axs[i]
    temp_lineages = uni_lineages[i * n_per_row : (i + 1) * n_per_row]
    temp_df = meta[meta["Merge Class"].isin(temp_lineages)]
    sns.stripplot(
        data=temp_df,
        x="Merge Class",
        y="dendrite_input",
        ax=ax,
        palette="deep",
        jitter=0.4,
    )
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel("")
    remove_spines(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_tick_params(length=0)
xlim = ax.get_xlim()
ax = axs[-1]
temp_lineages = uni_lineages[(i + 1) * n_per_row :]
temp_df = meta[meta["Merge Class"].isin(temp_lineages)]
sns.stripplot(
    data=temp_df, x="Merge Class", y="dendrite_input", ax=ax, palette="deep", jitter=0.4
)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.set_xlabel("")
remove_spines(ax)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_tick_params(length=0)
ax.set_xlim(xlim)
plt.tight_layout()
stashfig("all-merge-class-dendrite-input")

# %% [markdown]
# # plot some kind of asymmetry score by lineage
# # - proportion of edges onto a lineage which are asymmetric after thresholding
# # - IOU score?
# # - something else?

graph_type = "Gadn"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
g = mg.g
meta = mg.meta

lineages = meta["lineage"]
lineages = np.vectorize(filter)(lineages)
meta["lineage"] = lineages

edgelist_df = mg.to_edgelist(remove_unpaired=True)
edgelist_df["source"] = edgelist_df["source"].astype("int64")
edgelist_df["target"] = edgelist_df["target"].astype("int64")

n_paired_edges = len(edgelist_df)
# get rid of edges where the target is a low dendritic input node
edgelist_df = edgelist_df[~edgelist_df["target"].isin(remove_inds)]
n_left_edges = len(edgelist_df)

max_pair_edge_df = edgelist_df.groupby("edge pair ID").max()
edge_max_weight_map = dict(
    zip(max_pair_edge_df.index.values, max_pair_edge_df["weight"])
)
edgelist_df["max_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
    edge_max_weight_map
)

threshold = 0.0
thresh_df = max_pair_edge_df[max_pair_edge_df["weight"] > threshold]

source_pair_ids = np.unique(max_pair_edge_df["source Pair ID"])
target_pair_ids = np.unique(max_pair_edge_df["target Pair ID"])
pair_ids = np.union1d(source_pair_ids, target_pair_ids)

rows = []
for pid in pair_ids:
    temp_df = thresh_df[
        (thresh_df["source Pair ID"] == pid) | (thresh_df["target Pair ID"] == pid)
    ]

    if len(temp_df) > 0:
        iou = len(temp_df[temp_df["edge pair counts"] == 2]) / len(temp_df)
    else:
        iou = 0

    temp_meta = meta[meta["Pair ID"] == pid]
    lineage = temp_meta["lineage"].values[0]
    row = {"IOU": iou, "lineage": lineage}
    rows.append(row)

lineage_iou_df = pd.DataFrame(rows)

n_rows = 6
fig, axs = plt.subplots(n_rows, 1, figsize=(15, 30), sharey=True)
uni_lineages = np.unique(lineage_iou_df["lineage"])
n_lineages = len(uni_lineages)
n_per_row = n_lineages // n_rows
for i in range(n_rows):
    ax = axs[i]
    temp_lineages = uni_lineages[i * n_per_row : (i + 1) * n_per_row]
    temp_df = lineage_iou_df[lineage_iou_df["lineage"].isin(temp_lineages)]
    sns.stripplot(data=temp_df, x="lineage", y="IOU", ax=ax, palette="deep", jitter=0.4)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel("")
    remove_spines(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_tick_params(length=0)
plt.suptitle(f"IOU after threshold = {threshold}", y=1.02)
plt.tight_layout()
stashfig(f"all-lineage-iou-{threshold}")

# %% [markdown]
# # Do the same by cell class

rows = []
for pid in pair_ids:
    temp_df = thresh_df[
        (thresh_df["source Pair ID"] == pid) | (thresh_df["target Pair ID"] == pid)
    ]

    if len(temp_df) > 0:
        iou = len(temp_df[temp_df["edge pair counts"] == 2]) / len(temp_df)
    else:
        iou = 0

    temp_meta = meta[meta["Pair ID"] == pid]
    lineage = temp_meta["Merge Class"].values[0]
    row = {"IOU": iou, "Merge Class": lineage}
    rows.append(row)

lineage_iou_df = pd.DataFrame(rows)

n_rows = 3
fig, axs = plt.subplots(n_rows, 1, figsize=(15, 30), sharey=True)
uni_lineages = np.unique(lineage_iou_df["Merge Class"])
n_lineages = len(uni_lineages)
n_per_row = n_lineages // n_rows
for i in range(n_rows):
    ax = axs[i]
    temp_lineages = uni_lineages[i * n_per_row : (i + 1) * n_per_row]
    temp_df = lineage_iou_df[lineage_iou_df["Merge Class"].isin(temp_lineages)]
    sns.stripplot(
        data=temp_df, x="Merge Class", y="IOU", ax=ax, palette="deep", jitter=0.4
    )
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel("")
    remove_spines(ax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_tick_params(length=0)
plt.suptitle(f"IOU after threshold = {threshold}", y=1.02)
plt.tight_layout()
stashfig(f"all-class-iou-{threshold}")
