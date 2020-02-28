# %% [markdown]
# #
import os
import urllib.request
from operator import itemgetter
from pathlib import Path

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graph_tool import load_graph
from graph_tool.inference import minimize_blockmodel_dl
from joblib import Parallel, delayed
from random_word import RandomWords
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import ParameterGrid

from graspy.utils import cartprod
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig
from src.utils import get_blockmodel_df
from src.visualization import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    barplot_text,
    probplot,
    remove_spines,
    stacked_barplot,
)

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
BRAIN_VERSION = "2020-02-26"


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def add_max_weight(df):
    max_pair_edges = df.groupby("edge pair ID", sort=False)["weight"].max()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    df["max_weight"] = itemgetter(*df["edge pair ID"])(edge_max_weight_map)
    asym_inds = df[df["edge pair ID"] == -1].index
    df.loc[asym_inds, "max_weight"] = df.loc[asym_inds, "weight"]
    return df


def edgelist_to_mg(edgelist, meta):
    g = nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=nx.DiGraph)
    nx.set_node_attributes(g, meta.to_dict(orient="index"))
    mg = MetaGraph(g)
    return mg


run_dir = Path("81.1-BDP-community")
base_dir = Path("./maggot_models/notebooks/outs")
block_file = base_dir / run_dir / "csvs" / "block-labels.csv"
block_df = pd.read_csv(block_file, index_col=0)

run_names = block_df.columns.values
n_runs = len(block_df.columns)
block_pairs = cartprod(range(n_runs), range(n_runs))

param_file = base_dir / run_dir / "csvs" / "parameters.csv"
param_df = pd.read_csv(param_file, index_col=0)
param_df.set_index("param_key", inplace=True)
param_groupby = param_df.groupby(["graph_type", "threshold", "res", "binarize"])
param_df["Parameters"] = -1
for i, (key, val) in enumerate(param_groupby.indices.items()):
    param_df.iloc[val, param_df.columns.get_loc("Parameters")] = i


# %% [markdown]
# # Look at modularity over all of the parameters
sns.set_context("talk", font_scale=1)

mean_modularities = param_df.groupby("Parameters")["modularity"].mean()
order = mean_modularities.sort_values(ascending=False).index
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=param_df, x="Parameters", y="modularity", ax=ax, order=order, jitter=0.4
)
ax.set_xlabel("Parameter set")
stashfig("mod-by-parameters")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=param_df, x="threshold", y="modularity", ax=ax, jitter=0.4)
stashfig("mod-by-threshold")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=param_df, x="binarize", y="modularity", ax=ax, jitter=0.4)
stashfig("mod-by-binarize")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=param_df, x="res", y="modularity", ax=ax, jitter=0.4)
stashfig("mod-by-res")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=param_df,
    x="threshold",
    y="modularity",
    ax=ax,
    hue="Parameters",
    palette=cc.glasbey_light,
    jitter=0.45,
)
ax.legend([])
stashfig("mod-by-threshold-colored")

# %% [markdown]
# #
max_inds = param_df.groupby("Parameters")["modularity"].idxmax().values
best_param_df = param_df.loc[max_inds]
best_block_df = block_df[max_inds]
n_runs = len(max_inds)
block_pairs = cartprod(range(n_runs), range(n_runs))
ari_mat = np.empty((n_runs, n_runs))
for bp in block_pairs:
    from_block_labels = best_block_df.iloc[:, bp[0]].values
    to_block_labels = best_block_df.iloc[:, bp[1]].values
    mask = np.logical_and(~np.isnan(from_block_labels), ~np.isnan(to_block_labels))
    from_block_labels = from_block_labels[mask]
    to_block_labels = to_block_labels[mask]
    ari = adjusted_rand_score(from_block_labels, to_block_labels)
    ari_mat[bp[0], bp[1]] = ari
ari_df = pd.DataFrame(data=ari_mat, index=max_inds, columns=max_inds)


# %% [markdown]
# #


sns.set_context("talk", font_scale=1)

lut = dict(zip(param_df["Parameters"].unique(), cc.glasbey_light))
row_colors = param_df["Parameters"].map(lut)
clustergrid = sns.clustermap(
    ari_df,
    cmap="RdBu_r",
    center=0,
    method="single",
    vmin=None,
    figsize=(20, 20),
    row_colors=row_colors,
    col_colors=row_colors,
    dendrogram_ratio=0.2,
)
clustergrid.fig.suptitle("ARI", y=1.02)
stashfig("ari-clustermap")


# %% [markdown]
# #
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=best_param_df, x="Parameters", y="modularity", ax=ax, order=order, jitter=0.4
)
ax.set_xlabel("Parameter set")
stashfig("mod-by-parameters")

# %% [markdown]
# #
mb_classes = ["APL", "MBON", "MBIN", "KC"]

# for idx in best_param_df.index[:2]:


def compute_ari(idx):
    preprocess_params = dict(best_param_df.loc[idx, ["binarize", "threshold"]])
    graph_type = best_param_df.loc[idx, "graph_type"]
    mg = load_metagraph(graph_type, version=BRAIN_VERSION)
    mg = preprocess(mg, sym_threshold=True, remove_pdiff=True, **preprocess_params)
    left_mb_indicator = mg.meta["Class 1"].isin(mb_classes) & (
        mg.meta["Hemisphere"] == "L"
    )
    right_mb_indicator = mg.meta["Class 1"].isin(mb_classes) & (
        mg.meta["Hemisphere"] == "R"
    )
    labels = np.zeros(len(mg.meta))
    labels[left_mb_indicator.values] = 1
    labels[right_mb_indicator.values] = 2
    pred_labels = best_block_df[idx]
    pred_labels = pred_labels[pred_labels.index.isin(mg.meta.index)]
    assert np.array_equal(pred_labels.index, mg.meta.index)
    ari = adjusted_rand_score(labels, pred_labels)
    return ari


aris = Parallel(n_jobs=-2, verbose=15)(
    delayed(compute_ari)(i) for i in best_param_df.index
)

# %% [markdown]
# #
mg = load_metagraph("G", version=BRAIN_VERSION)


def compute_pairedness(partition, meta, rand_adjust=False, normalize=False, plot=False):
    partition = partition.copy()
    meta = meta.copy()
    uni_labels, inv = np.unique(partition, return_inverse=True)
    int_mat = np.zeros((len(uni_labels), len(uni_labels)))
    meta = meta.loc[partition.index]

    for i, ul in enumerate(uni_labels):
        c1_mask = inv == i
        c1_pairs = meta.loc[c1_mask, "Pair"]
        c1_pairs.drop(
            c1_pairs[c1_pairs == -1].index
        )  # HACK must be a better pandas sol
        for j, ul in enumerate(uni_labels):
            c2_mask = inv == j
            c2_inds = meta.loc[c2_mask].index
            n_pairs_in_other = np.sum(c1_pairs.isin(c2_inds))
            if normalize:
                n_pairs_in_other /= len(c1_pairs)
            int_mat[i, j] = n_pairs_in_other
    row_ind, col_ind = linear_sum_assignment(int_mat, maximize=True)
    pairedness = np.trace(int_mat[np.ix_(row_ind, col_ind)]) / np.sum(int_mat)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(
            int_mat, square=True, ax=axs[0], cbar=False, cmap="RdBu_r", center=0
        )
        int_df = pd.DataFrame(data=int_mat, index=uni_labels, columns=uni_labels)
        int_df = int_df.reindex(index=uni_labels[row_ind])
        int_df = int_df.reindex(columns=uni_labels[col_ind])
        sns.heatmap(int_df, square=True, ax=axs[1], cbar=False, cmap="RdBu_r", center=0)

    # TODO what is the null model here???

    if rand_adjust:
        part_vals = partition.values
        np.random.shuffle(part_vals)
        partition = pd.Series(data=part_vals, index=partition.index)
        rand_pairedness = compute_pairedness(
            partition, meta, rand_adjust=False, normalize=normalize, plot=False
        )
        pairedness = pairedness - rand_pairedness
    return pairedness


partitions = [best_block_df[idx] for idx in best_param_df.index]

pairedness = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_pairedness)(i, mg.meta) for i in partitions
)
# %% [markdown]
# #
#%%
argsort_inds = np.argsort(pairedness)[::-1]
sort_index = best_param_df.index[argsort_inds]


# %% [markdown]
# #
# get out some metadata

# class_label_dict = nx.get_node_attributes(g_sym, "Merge Class")
# class_labels = np.array(itemgetter(*skeleton_labels)(class_label_dict))

idx = sort_index[1]
preprocess_params = dict(best_param_df.loc[idx, ["binarize", "threshold"]])
graph_type = best_param_df.loc[idx, "graph_type"]
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
mg = preprocess(mg, sym_threshold=True, remove_pdiff=True, **preprocess_params)
left_mb_indicator = mg.meta["Class 1"].isin(mb_classes) & (mg.meta["Hemisphere"] == "L")
right_mb_indicator = mg.meta["Class 1"].isin(mb_classes) & (
    mg.meta["Hemisphere"] == "R"
)
labels = np.zeros(len(mg.meta))
labels[left_mb_indicator.values] = 1
labels[right_mb_indicator.values] = 2
pred_labels = best_block_df[idx]
pred_labels = pred_labels[pred_labels.index.isin(mg.meta.index)]
partition = pred_labels
title = idx
class_labels = mg["Merge Class"]
lineage_labels = mg["lineage"]
basename = idx


def augment_classes(class_labels, lineage_labels, fill_unk=True):
    if fill_unk:
        classlin_labels = class_labels.copy()
        fill_inds = np.where(class_labels == "unk")[0]
        classlin_labels[fill_inds] = lineage_labels[fill_inds]
        used_inds = np.array(list(CLASS_IND_DICT.values()))
        unused_inds = np.setdiff1d(range(len(cc.glasbey_light)), used_inds)
        lineage_color_dict = dict(
            zip(np.unique(lineage_labels), np.array(cc.glasbey_light)[unused_inds])
        )
        color_dict = {**CLASS_COLOR_DICT, **lineage_color_dict}
        hatch_dict = {}
        for key, val in color_dict.items():
            if key[0] == "~":
                hatch_dict[key] = "//"
            else:
                hatch_dict[key] = ""
    else:
        color_dict = "class"
        hatch_dict = None
    return classlin_labels, color_dict, hatch_dict


lineage_labels = np.vectorize(lambda x: "~" + x)(lineage_labels)
classlin_labels, color_dict, hatch_dict = augment_classes(class_labels, lineage_labels)

# TODO then sort all of them by proportion of sensory/motor
# barplot by merge class and lineage
_, _, order = barplot_text(
    partition,
    classlin_labels,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
    figsize=(24, 18),
    title=title,
    hatch_dict=hatch_dict,
    return_order=True,
)
stashfig(basename + "barplot-mergeclasslin-props")
category_order = np.unique(partition)[order]

fig, axs = barplot_text(
    partition,
    class_labels,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
    figsize=(24, 18),
    title=title,
    hatch_dict=None,
    category_order=category_order,
)
stashfig(basename + "barplot-mergeclass-props")
fig, axs = barplot_text(
    partition,
    class_labels,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=False,
    figsize=(24, 18),
    title=title,
    hatch_dict=None,
    category_order=category_order,
)
stashfig(basename + "barplot-mergeclass-counts")

# TODO add gridmap

counts = False
weights = False
prob_df = get_blockmodel_df(
    mg.adj, partition, return_counts=counts, use_weights=weights
)
prob_df = prob_df.reindex(category_order, axis=0)
prob_df = prob_df.reindex(category_order, axis=1)
probplot(100 * prob_df, fmt="2.0f", figsize=(20, 20), title=title, font_scale=0.7)
stashfig(basename + f"probplot-counts{counts}-weights{weights}")


# %% [markdown]
# # Need to figure out L/R community matching
# Cells can either be in a

# for each cluster
#     find the other cluster which contains the most pairs that look like my pairs
#     (how to settle ties?)
#

# make a contigency table.

# %% [markdown]
# # Look at the relationship between ARI MB and Pairdness

best_param_df["ARI-MB"] = aris
best_param_df["Pairedness"] = pairedness
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(data=best_param_df, x="ARI-MB", y="Pairedness", ax=ax)
stashfig("pair-vs-ari")
# %% [markdown]
# #

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=best_param_df, x="Parameters", y="Pairedness", ax=ax, order=order, jitter=0.4
)
ax.set_xlabel("Parameter set")
stashfig("pair-by-parameters")


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=best_param_df, x="threshold", y="Pairedness", ax=ax, jitter=0.4)
stashfig("pair-by-threshold")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=best_param_df, x="binarize", y="Pairedness", ax=ax, jitter=0.4)
stashfig("pair-by-binarize")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=best_param_df, x="res", y="Pairedness", ax=ax, jitter=0.4)
stashfig("pair-by-res")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=best_param_df,
    x="threshold",
    y="Pairedness",
    ax=ax,
    hue="Parameters",
    palette=cc.glasbey_light,
    jitter=0.45,
)
ax.legend([])
stashfig("pair-by-threshold-colored")

# %% [markdown]
# #

adjusted_pairedness = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_pairedness)(i, mg.meta, rand_adjust=True) for i in partitions
)


# %%
print(compute_pairedness(partitions[i], mg.meta, rand_adjust=True))
print(compute_pairedness(partitions[i], mg.meta, rand_adjust=True))
print(compute_pairedness(partitions[i], mg.meta, rand_adjust=True))
print(compute_pairedness(partitions[i], mg.meta, rand_adjust=True))
print(compute_pairedness(partitions[i], mg.meta, rand_adjust=True))
#%%
print(compute_pairedness(partitions[i], mg.meta, rand_adjust=False))
w
