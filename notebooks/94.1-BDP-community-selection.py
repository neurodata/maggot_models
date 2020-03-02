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
BRAIN_VERSION = "2020-03-01"

mb_classes = ["APL", "MBON", "MBIN", "KC"]
al_classes = [
    "bLN-Duet",
    "bLN-Trio",
    "cLN",
    "keystone",
    "mPN-multi",
    "mPN-olfac",
    "mPN;FFN-multi",
    "pLN",
    "uPN",
    "sens-ORN",
]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def compute_ari(idx, param_df, classes, class_type="Class 1", remove_non_mb=False):
    preprocess_params = dict(param_df.loc[idx, ["binarize", "threshold"]])
    graph_type = param_df.loc[idx, "graph_type"]
    mg = load_metagraph(graph_type, version=BRAIN_VERSION)
    mg = preprocess(mg, sym_threshold=True, remove_pdiff=True, **preprocess_params)
    left_mb_indicator = mg.meta[class_type].isin(classes) & (
        mg.meta["Hemisphere"] == "L"
    )
    right_mb_indicator = mg.meta[class_type].isin(classes) & (
        mg.meta["Hemisphere"] == "R"
    )
    labels = np.zeros(len(mg.meta))
    labels[left_mb_indicator.values] = 1
    labels[right_mb_indicator.values] = 2
    pred_labels = best_block_df[idx]
    pred_labels = pred_labels[pred_labels.index.isin(mg.meta.index)]  # FIXME
    assert np.array_equal(pred_labels.index, mg.meta.index), print(idx)

    if remove_non_mb:  # only consider ARI for clusters with some MB mass
        uni_pred = np.unique(pred_labels)
        keep_mask = np.ones(len(labels), dtype=bool)
        for p in uni_pred:
            if np.sum(labels[pred_labels == p]) == 0:
                keep_mask[pred_labels == p] = False
        labels = labels[keep_mask]
        pred_labels = pred_labels[keep_mask]

    ari = adjusted_rand_score(labels, pred_labels)
    return ari


def compute_good_ari(
    partition, meta, classes, class_type="Class 1", remove_other=False, mask_other=False
):
    partition = partition.copy()
    partition = partition[~partition.isna()]
    meta = meta.copy()
    meta = meta.loc[partition.index]
    assert np.array_equal(partition.index, meta.index)
    left_indicator = meta[class_type].isin(classes) & (meta["Hemisphere"] == "L")
    right_indicator = meta[class_type].isin(classes) & (meta["Hemisphere"] == "R")
    true_labels = np.zeros(len(meta))
    true_labels[left_indicator.values] = 1
    true_labels[right_indicator.values] = 2

    if remove_other or mask_other:  # only consider ARI for clusters with some MB mass
        uni_pred = np.unique(partition)
        keep_mask = np.ones(len(true_labels), dtype=bool)
        for p in uni_pred:
            if np.sum(true_labels[partition == p]) == 0:
                keep_mask[partition == p] = False
        if remove_other:
            true_labels = true_labels[keep_mask]
            partition = partition[keep_mask]
        else:
            # make everything one class
            partition[~keep_mask] = -9999
    ari = adjusted_rand_score(true_labels, partition)
    return ari


def compute_classness(partition, meta, classes, class_type="Class 1"):
    partition = partition.copy()
    partition = partition[~partition.isna()]
    meta = meta.copy()
    meta = meta.loc[partition.index]

    left_indicator = meta[class_type].isin(classes) & (meta["Hemisphere"] == "L")
    right_indicator = meta[class_type].isin(classes) & (meta["Hemisphere"] == "R")
    n_left = left_indicator.sum()
    n_right = right_indicator.sum()

    uni_labels, inv = np.unique(partition, return_inverse=True)
    left_score = 0
    right_score = 0
    for i, ul in enumerate(uni_labels):
        pred_mask = inv == i
        left_intersection = left_indicator[pred_mask]
        left_true_in_pred = left_intersection.sum()
        left_score += left_true_in_pred ** 2 / (n_left * pred_mask.sum())
        right_intersection = right_indicator[pred_mask]
        right_true_in_pred = right_intersection.sum()
        right_score += right_true_in_pred ** 2 / (n_right * pred_mask.sum())
    return (left_score + right_score) / 2


# FIXME
def compute_pairedness(partition, meta, holdout=None, rand_adjust=False, plot=False):
    partition = partition.copy()
    meta = meta.copy()

    if holdout is not None:
        keep_inds = meta[~meta["Pair ID"].isin(holdout)].index
        test_inds = meta[meta["Pair ID"].isin(holdout)].index
        # partition = partition.loc[keep_inds]

    uni_labels, inv = np.unique(partition, return_inverse=True)

    train_int_mat = np.zeros((len(uni_labels), len(uni_labels)))
    test_int_mat = np.zeros((len(uni_labels), len(uni_labels)))
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
            train_pairs_in_other = np.sum(
                c1_pairs.isin(c2_inds) & c1_pairs.index.isin(keep_inds)
            )
            test_pairs_in_other = np.sum(
                c1_pairs.isin(c2_inds) & c1_pairs.index.isin(test_inds)
            )
            train_int_mat[i, j] = train_pairs_in_other
            test_int_mat[i, j] = test_pairs_in_other

    row_ind, col_ind = linear_sum_assignment(train_int_mat, maximize=True)
    train_pairedness = np.trace(train_int_mat[np.ix_(row_ind, col_ind)]) / np.sum(
        train_int_mat
    )
    test_pairedness = np.trace(test_int_mat[np.ix_(row_ind, col_ind)]) / np.sum(
        test_int_mat
    )

    if plot:
        # FIXME broken
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(
            int_mat, square=True, ax=axs[0], cbar=False, cmap="RdBu_r", center=0
        )
        int_df = pd.DataFrame(data=int_mat, index=uni_labels, columns=uni_labels)
        int_df = int_df.reindex(index=uni_labels[row_ind])
        int_df = int_df.reindex(columns=uni_labels[col_ind])
        sns.heatmap(int_df, square=True, ax=axs[1], cbar=False, cmap="RdBu_r", center=0)

    if rand_adjust:
        # attempt to correct for difference in matchings as result of random chance
        # TODO this could be analytic somehow
        part_vals = partition.values
        np.random.shuffle(part_vals)
        partition = pd.Series(data=part_vals, index=partition.index)
        rand_train_pairedness, rand_test_pairedness = compute_pairedness(
            partition, meta, rand_adjust=False, plot=False, holdout=holdout
        )
        test_pairedness -= rand_test_pairedness
        train_pairedness -= rand_train_pairedness
        # pairedness = pairedness - rand_pairedness
    return train_pairedness, test_pairedness


# %% [markdown]
# # Load the runs

run_dir = Path("81.3-BDP-community")
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
# # Get the best modularities by parameter set

max_inds = param_df.groupby("Parameters")["modularity"].idxmax().values
best_param_df = param_df.loc[max_inds]
best_block_df = block_df[max_inds]
n_runs = len(max_inds)

mg = load_metagraph("G", version=BRAIN_VERSION)

meta = mg.meta
partitions = [best_block_df[idx] for idx in best_param_df.index]

# %% [markdown]
# # Compute ARI relative to MB

aris = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_good_ari)(i, meta, mb_classes, "Class 1") for i in partitions
)
best_param_df["MB-ARI"] = aris

aris = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_good_ari)(i, meta, mb_classes, "Class 1", remove_other=True)
    for i in partitions
)
best_param_df["MB-roARI"] = aris

aris = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_good_ari)(i, meta, mb_classes, "Class 1", mask_other=True)
    for i in partitions
)
best_param_df["MB-moARI"] = aris

# %% [markdown]
# # ARI relative to AL

aris = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_good_ari)(i, meta, al_classes, "Class 1") for i in partitions
)
best_param_df["AL-ARI"] = aris

aris = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_good_ari)(i, meta, al_classes, "Class 1", remove_other=True)
    for i in partitions
)
best_param_df["AL-roARI"] = aris

aris = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_good_ari)(i, meta, al_classes, "Class 1", mask_other=True)
    for i in partitions
)
best_param_df["AL-moARI"] = aris


# %% [markdown]
# # Compute pairedness
# TODO get the held-out pairs

outs = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_classness)(i, meta, mb_classes, "Class 1") for i in partitions
)
best_param_df["MB-cls"] = outs
#%%
outs = Parallel(n_jobs=-2, verbose=10)(
    delayed(compute_classness)(i, meta, al_classes, "Merge Class") for i in partitions
)
best_param_df["AL-cls"] = outs

#%%

mb_pairs = meta[meta["Class 1"].isin(mb_classes) & (meta["Pair"] != -1)][
    "Pair ID"
].unique()
pairs = meta[meta["Pair"] != -1]["Pair ID"].unique()
pairs_left = np.setdiff1d(pairs, mb_pairs)
holdout_pairs = np.random.choice(pairs_left, size=(len(pairs_left) // 2), replace=False)
test_pairs = np.setdiff1d(pairs_left, holdout_pairs)

# %% [markdown]
# #


# outs = Parallel(n_jobs=-2, verbose=10)(
#     delayed(compute_pairedness)(i, mg.meta, rand_adjust=True, holdout=holdout_pairs)
#     for i in partitions
# )

# outs = list(zip(*outs))
# train_pairedness = outs[0]
# test_pairedness = outs[1]
# best_param_df["train_pairedness"] = train_pairedness
# best_param_df["test_pairedness"] = test_pairedness

#%%

stashcsv(best_param_df, "best_params")
