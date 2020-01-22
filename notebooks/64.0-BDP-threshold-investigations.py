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
BRAIN_VERSION = "2020-01-21"

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


def threshold_sweep(edgelist_df, max_pair_edgelist_df, start=0, stop=0.3, steps=20):
    threshs = np.linspace(start, stop, steps)
    rows = []
    for threshold in threshs:
        thresh_df = max_pair_edge_df[max_pair_edge_df["weight"] > threshold]
        p_sym = len(thresh_df[thresh_df["edge pair counts"] == 2]) / len(thresh_df)
        p_edges_left = (
            thresh_df["edge pair counts"].sum()
            / max_pair_edge_df["edge pair counts"].sum()
        )
        temp_df = edgelist_df[edgelist_df["max_weight"] > threshold]
        p_syns_left = temp_df["weight"].sum() / edgelist_df["weight"].sum()
        row = {
            "threshold": threshold,
            "Prop. paired edges symmetric": p_sym,
            "Prop. edges left": p_edges_left,
            "Prop. synapses left": p_syns_left,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# %% [markdown]
# # throw out all edges to or from any cell with...
# # threshold curves for cells w > 100 dendritic inputs
# # threshold curves for cells w > 50 dendritic inputs
# # do the thresholding based on percent input

base_path = Path(
    "maggot_models/data/raw/Maggot-Brain-Connectome/4-color-matrices_Brain/"
)
sub_path = Path("2020-01-21/input_counts.csv")
input_path = base_path / sub_path
input_df = pd.read_csv(input_path)
input_df = input_df.set_index("skeleton_id")
input_thresh = 100
remove_inds = input_df[input_df[" dendrite_inputs"] < input_thresh].index

graph_type = "Gadn"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
g = mg.g
meta = mg.meta
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

thresh_result_df = threshold_sweep(edgelist_df, max_pair_edge_df)

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
    f"Min dendridic input = {input_thresh} (removed {n_paired_edges - n_left_edges} edges)"
)
pad = 0.02
ax.set_ylim((0 - pad, 1 + pad))
ax_right.set_ylim((0 - pad, 1 + pad))
plt.legend(bbox_to_anchor=(1.08, 1), loc=2, borderaxespad=0.0)
stashfig(f"min-dend-{input_thresh}-threshold-sweep-{graph_type}")
# %% [markdown]
# # get number of inputs to kenyon cells
# # just list the number of connections onto each kenyon cell, by claw number

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
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.stripplot(data=sub_edgelist_df, y="weight", x="target", ax=ax)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_title(name + " input weights")
    remove_spines(ax)
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
    tick.set_rotation(45)
ax.set_title(name + " input weights")
remove_spines(ax)
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
