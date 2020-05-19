# %% [markdown]
# ##
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import tplquad
from scipy.stats import gaussian_kde

import pymaid
from graspy.utils import pass_to_ranks
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    get_mid_map,
    gridmap,
    matrixplot,
    remove_axis,
    remove_spines,
    set_axes_equal,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# params

level = 7
class_key = f"lvl{level}_labels"

metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "color_iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.2-BDP-omni-clust"


# load data
pair_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
pair_meta["lvl0_labels"] = pair_meta["lvl0_labels"].astype(str)
pair_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
pair_adj = pair_adj.values
mg = MetaGraph(pair_adj, pair_meta)
meta = mg.meta

level_names = [f"lvl{i}_labels" for i in range(level + 1)]


def sort_mg(mg, level_names):
    meta = mg.meta
    sort_class = level_names + ["merge_class"]
    class_order = ["sf"]
    total_sort_by = []
    for sc in sort_class:
        for co in class_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    mg = mg.sort_values(total_sort_by, ascending=False)
    return mg


# def calc_ego_connectivity(adj, meta, label, axis=0):
#     this_inds = meta[meta[class_key] == label]["inds"].values
#     uni_cat = meta[key].unique()
#     connect_mat = []
#     for other_label in uni_cat:
#         other_inds = meta[meta[key] == other_label]["inds"].values
#         if axis == 0:
#             sum_vec = adj[np.ix_(other_inds, this_inds)].sum(axis=axis)
#         elif axis == 1:
#             sum_vec = adj[np.ix_(this_inds, other_inds)].sum(axis=axis)
#         connect_mat.append(sum_vec)
#     return np.array(connect_mat)


mg = sort_mg(mg, level_names)
meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj


skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)


# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)


# %% [markdown]
# ##

# plot params
scale = 5
n_col = 10
n_row = 3
margin = 0.01
gap = 0.02

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "grey",
    "ytick.color": "dimgrey",
    "xtick.color": "dimgrey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
}
for k, val in rc_dict.items():
    mpl.rcParams[k] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

# compare dendrite inputs

compartment = "dendrite"
direction = "postsynaptic"


def filter_connectors(connectors, ids, direction, compartment):
    label_connectors = connectors[connectors[f"{direction}_to"].isin(ids)]
    label_connectors = label_connectors[
        label_connectors[f"{direction}_type"] == compartment
    ]
    label_connectors = label_connectors[
        ~label_connectors["connector_id"].duplicated(keep="first")
    ]
    return label_connectors


from hyppo.ksample import KSample


def run_dcorr(data1, data2):
    ksamp = KSample("Dcorr")
    stat, pval = ksamp.test(data1, data2, auto=True, workers=-1)
    return stat, pval


def spatial_dcorr(data1, data2, method="full", max_samples=1000, n_subsamples=10):
    if (len(data1) == 0) or (len(data2) == 0):
        return np.nan, np.nan

    if method == "full":
        stat, p_val = run_dcorr(data1, data2)
    elif method == "subsample":
        stats = np.empty(n_subsamples)
        p_vals = np.empty(n_subsamples)
        for i in range(n_subsamples):
            subsampled_data = []
            for data in [data1, data2]:
                n_subsamples = min(len(data), max_samples)
                inds = np.random.choice(n_subsamples, size=n_subsamples, replace=False)
                subsampled_data.append(data[inds])
            stat, p_val = run_dcorr(*subsampled_data)
            stats[i] = stat
            p_vals[i] = p_val
        stat = np.median(stats)
        p_val = np.median(p_vals)
    elif method == "max-d":
        max_dim_stat = -np.inf
        best_p_val = np.nan
        for dim in range(data1.shape[1]):
            dim_stat, dim_p_val = run_dcorr(data1[:, dim], data2[:, dim])
            if dim_stat > max_dim_stat:
                max_dim_stat = dim_stat
                best_p_val = dim_p_val
        stat = max_dim_stat
        p_val = best_p_val
    else:
        raise ValueError()

    return stat, p_val


# %% [markdown]
# ##
import time

currtime = time.time()

n_reps = 5
labels = ["uPN", "mPN", "KC"]
class_keys = 3 * ["class1"]

rows = []

for _ in range(n_reps):
    class_ids = []
    class_names = []
    for label, class_key in zip(labels, class_keys):
        # split the class in half
        all_ids = meta[meta[class_key] == label].index.values
        label1_ids = np.random.choice(all_ids, size=len(all_ids) // 2, replace=False)
        label2_ids = np.setdiff1d(all_ids, label1_ids)
        class_ids.append(label1_ids)
        class_ids.append(label2_ids)
        class_names.append(label + "_1")
        class_names.append(label + "_2")

    for i, (label1_ids, label1) in enumerate(zip(class_ids, class_names)):
        for j, (label2_ids, label2) in enumerate(zip(class_ids, class_names)):
            if i < j:
                label1_connectors = filter_connectors(
                    connectors, label1_ids, direction, compartment
                )
                label2_connectors = filter_connectors(
                    connectors, label2_ids, direction, compartment
                )
                data1 = label1_connectors[["x", "y", "z"]].values
                data2 = label2_connectors[["x", "y", "z"]].values
                print(len(data1))
                print(len(data2))
                stat, p_val = spatial_dcorr(data1, data2, method="full")
                same = label1[:-2] == label2[:-2]
                row = {
                    "stat": stat,
                    "p_val": p_val,
                    "label": f"{label1} vs {label2}",
                    "same": same,
                }
                rows.append(row)

print(f"{time.time() - currtime} elapsed")


res_df = pd.DataFrame(rows)
res_df["-log10_p_val"] = -np.log10(res_df["p_val"])
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.stripplot(data=res_df, x="label", y="stat", hue="same")
plt.xticks(rotation=90)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Same class")
ax.set_title("Axon outputs")
stashfig("spatial-dcorr-dendrite-inputs-max-d")

# print(stats)
# print(p_vals)

# plot_p_vals = -np.log10(p_vals)
# adjplot(
#     plot_p_vals,
#     meta=cluster_meta,
#     center=0,
#     vmax=np.nanmax(plot_p_vals[~np.isinf(plot_p_vals)]),
#     cbar_kws=dict(shrink=0.7),
# )
# %% [markdown]
# ##
first = 3
class_labels = meta[class_key].unique()[::-1][10:15]
p_vals = np.zeros((len(class_labels), len(class_labels)))
stats = np.zeros_like(p_vals)
cluster_meta = pd.DataFrame(index=class_labels)

for i, label1 in enumerate(class_labels):
    label1_meta = meta[meta[class_key] == label1]
    label1_ids = label1_meta.index.values
    label1_connectors = filter_connectors(
        connectors, label1_ids, direction, compartment
    )
    cluster_meta.loc[label1, "n_samples"] = len(label1_connectors)
    for j, label2 in enumerate(class_labels):
        if i < j:
            label2_meta = meta[meta[class_key] == label2]
            label2_ids = label2_meta.index.values
            label2_connectors = filter_connectors(
                connectors, label2_ids, direction, compartment
            )
            data1 = label1_connectors[["x", "y", "z"]].values
            data2 = label2_connectors[["x", "y", "z"]].values
            stat, p_val = spatial_dcorr(data1, data2, method="full")
            stats[i, j] = stat
            p_vals[i, j] = p_val


print(stats)
print(p_vals)

plot_p_vals = -np.log10(p_vals)
adjplot(
    plot_p_vals,
    meta=cluster_meta,
    center=0,
    vmax=np.nanmax(plot_p_vals[~np.isinf(plot_p_vals)]),
    cbar_kws=dict(shrink=0.7),
)

