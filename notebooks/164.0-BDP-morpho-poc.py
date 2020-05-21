# %% [markdown]
# ##
import os
import time

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
from hyppo.ksample import KSample
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
from src.spatial import spatial_dcorr, run_dcorr

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


mg = sort_mg(mg, level_names)
meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj


# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)


# %% [markdown]
# ##


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


def filter_connectors(connectors, ids, direction, compartment):
    label_connectors = connectors[connectors[f"{direction}_to"].isin(ids)]
    label_connectors = label_connectors[
        label_connectors[f"{direction}_type"] == compartment
    ]
    label_connectors = label_connectors[
        ~label_connectors["connector_id"].duplicated(keep="first")
    ]
    return label_connectors


from src.visualization import plot_3view

start_instance()

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)

# %% [markdown]
# ##
# compare dendrite inputs

compartment = "dendrite"
direction = "postsynaptic"
if direction == "presynaptic":
    direction_label = "output"
elif direction == "postsynaptic":
    direction_label = "input"

n_reps = 10
labels = ["uPN", "mPN", "KC"]
class_keys = 3 * ["class1"]
n_subsamples = 12
max_samples = 500
method = "full"

currtime = time.time()

rows = []

for rep in range(n_reps):
    print(rep)
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
                stat, p_val = spatial_dcorr(
                    data1,
                    data2,
                    method=method,
                    max_samples=max_samples,
                    n_subsamples=n_subsamples,
                )
                same = label1[:-2] == label2[:-2]
                row = {
                    "stat": stat,
                    "p_val": p_val,
                    "label": f"{label1} vs {label2}",
                    "same": same,
                }
                rows.append(row)
                if rep == 0:
                    fig = plt.figure(figsize=(15, 10))
                    axs = np.empty((2, 3), dtype="O")
                    gs = plt.GridSpec(2, 3, figure=fig, wspace=0, hspace=0)
                    for i in range(2):
                        for j in range(3):
                            ax = fig.add_subplot(gs[i, j], projection="3d")
                            axs[i, j] = ax
                            ax.axis("off")
                    plot_3view(
                        label1_connectors,
                        axs[0, :],
                        palette=skeleton_color_dict,
                        label_by=f"{direction}_to",
                        alpha=0.6,
                        s=2,
                        row_title=label1,
                    )
                    plot_3view(
                        label2_connectors,
                        axs[1, :],
                        palette=skeleton_color_dict,
                        label_by=f"{direction}_to",
                        alpha=0.6,
                        s=2,
                        row_title=label2,
                    )
                    savename = f"morpho-{label1}-{label2}-compartment={compartment}-direction={direction}"
                    fig.suptitle(
                        f"{compartment} {direction_label}, stat = {stat:.3f}", y=0.9
                    )
                    stashfig(savename)


print(f"{time.time() - currtime} elapsed")


# %% [markdown]
# ##
res_df = pd.DataFrame(rows)
res_df["-log10_p_val"] = -np.log10(res_df["p_val"])

y = "stat"

basename = f"-compartment={compartment}-direction={direction}-method={method}"
if method == "subsample":
    basename += f"-n_sub={n_subsamples}-max_samp={max_samples}"

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.stripplot(data=res_df, x="label", y=y, hue="same")
plt.xticks(rotation=90)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Same class")
ax.set_title(f"Dcorr compartment = {compartment}, direction = {direction}")
stashfig(f"dcorr-poc-{y}" + basename)

