# %% [markdown]
# #
import os
from pathlib import Path

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
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
BRAIN_VERSION = "2020-03-02"


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


# %% [markdown]
# # Load the runs

run_dir = Path("81.3-BDP-community")
base_dir = Path("./maggot_models/notebooks/outs")
block_file = base_dir / run_dir / "csvs" / "block-labels.csv"
block_df = pd.read_csv(block_file, index_col=0)

run_names = block_df.columns.values
n_runs = len(block_df.columns)
block_pairs = cartprod(range(n_runs), range(n_runs))

opt_dir = Path("94.1-BDP-community-selection")
param_file = base_dir / opt_dir / "csvs" / "best_params.csv"
param_df = pd.read_csv(param_file, index_col=0)

best_block_df = block_df[param_df.index]
# %% [markdown]
# #

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="MB-ARI", y="MB-cls", ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="MB-cls", y="AL-cls", ax=ax)


# %% [markdown]
# # plot results

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(data=param_df, x="MB-ARI", y="train_pairedness", ax=axs[0])
sns.scatterplot(data=param_df, x="MB-ARI", y="test_pairedness", ax=axs[1])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="test_pairedness", y="train_pairedness", ax=ax)

# %% [markdown]
# #
rank_df = param_df.rank(axis=0, ascending=False)
# param_df.loc[rank_df.index, "rank_train_pairedness"] = rank_df["train_pairedness"]
# param_df.loc[rank_df.index, "rank_test_pairedness"] = rank_df["test_pairedness"]
param_df.loc[rank_df.index, "rank_MB-ARI"] = rank_df["MB-ARI"]
param_df.loc[rank_df.index, "rank_MB-cls"] = rank_df["MB-cls"]
param_df.loc[rank_df.index, "rank_AL-ARI"] = rank_df["AL-ARI"]
param_df.loc[rank_df.index, "rank_AL-cls"] = rank_df["AL-cls"]
param_df.sort_values("MB-ARI", ascending=False)

#%%
param_df.sort_values("AL-ARI", ascending=False)

# %% [markdown]
# # Plot a candidate

# idx = sort_index[2]
idx = "GoodmanDeimos"
preprocess_params = dict(param_df.loc[idx, ["binarize", "threshold"]])
graph_type = param_df.loc[idx, "graph_type"]
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
mg = preprocess(mg, sym_threshold=True, remove_pdiff=True, **preprocess_params)

labels = np.zeros(len(mg.meta))

pred_labels = best_block_df[idx]
pred_labels = pred_labels[pred_labels.index.isin(mg.meta.index)]
partition = pred_labels.astype(int)
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


# %%
