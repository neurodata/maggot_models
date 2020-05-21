# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.visualization import remove_shared_ax

from scipy.cluster.hierarchy import linkage
from scipy.integrate import tplquad
from scipy.spatial.distance import squareform
from scipy.special import comb
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances

import pymaid
from graspy.utils import is_symmetric, pass_to_ranks, symmetrize
from hyppo.ksample import KSample
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    draw_leaf_dendrogram,
    get_mid_map,
    gridmap,
    matrixplot,
    plot_single_dendrogram,
    remove_axis,
    remove_spines,
    set_axes_equal,
    stacked_barplot,
)


np.random.seed(8888)

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


level = 4
lowest_level = level
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
mg = sort_mg(mg, level_names)


fig, axs = plt.subplots(
    1,
    2,
    figsize=(20, 15),
    sharey=True,
    gridspec_kw=dict(width_ratios=[0.25, 0.75], wspace=0),
)
mid_map = draw_leaf_dendrogram(
    mg.meta, axs[0], lowest_level=lowest_level, draw_labels=False
)
key_order = list(mid_map.keys())


compartment = "dendrite"
direction = "postsynaptic"
foldername = "160.1-BDP-morpho-dcorr"
filename = f"test-statslvl={level}-compartment={compartment}-direction={direction }-method=subsample-n_sub=96-max_samp=500"
stat_df = readcsv(filename, foldername=foldername, index_col=0)
sym_vals = symmetrize(stat_df.values, method="triu")
stat_df = pd.DataFrame(data=sym_vals, index=stat_df.index, columns=stat_df.index)
ordered_stat_df = stat_df.loc[key_order, key_order]
sns.set_context("talk")
sns.heatmap(ordered_stat_df, ax=axs[1], cbar=False, cmap="RdBu_r", center=0)
axs[1].invert_yaxis()
axs[1].invert_xaxis()
axs[1].set_xticklabels([])

remove_shared_ax(axs[0])
remove_shared_ax(axs[1])
axs[1].set_yticks(np.arange(len(key_order)) + 0.5)
axs[1].set_yticklabels(key_order)
axs[1].yaxis.tick_right()
plt.tick_params(labelright="on", rotation=0, color="grey", labelsize=8)
axs[1].set_xticks([])
axs[1].set_title(
    f"DCorr test statistic, compartment = {compartment}, direction = {direction}"
)
# plt.tight_layout()
basename = f"-test-stats-lvl={level}-compartment={compartment}-direction={direction}"
stashfig("dcorr-heatmap-bar-dendrogram" + basename)


# sym_vals[~np.isfinite(sym_vals)] = 1
# pdist = squareform(sym_vals)
# Z = linkage(pdist, method="average", metric="euclidean")

# sns.clustermap(
#     sym_vals, row_linkage=Z, col_linkage=Z, xticklabels=False, yticklabels=False
# )
# stashfig("test-stat-clustered")
