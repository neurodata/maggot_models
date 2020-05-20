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
from scipy.special import comb
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances

import pymaid
from graspy.utils import pass_to_ranks
from hyppo.ksample import KSample
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
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
from joblib import Parallel, delayed
from graspy.utils import symmetrize

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


lowest_level = 7
level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
mg = sort_mg(mg, level_names)

from src.visualization import plot_single_dendrogram, draw_leaf_dendrogram
from graspy.utils import is_symmetric

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

foldername = "160.1-BDP-morpho-dcorr"
filename = "test-stats-method=subsample-n_subsample=48-max_samples=500"
stat_df = readcsv(filename, foldername=foldername, index_col=0)
sym_vals = symmetrize(stat_df.values, method="triu")
stat_df = pd.DataFrame(data=sym_vals, index=stat_df.index, columns=stat_df.index)
ordered_stat_df = stat_df.loc[key_order, key_order]
sns.heatmap(ordered_stat_df, ax=axs[1], cbar=False)
axs[1].invert_yaxis()
axs[1].invert_xaxis()
axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
plt.tight_layout()
stashfig("dcorr-heatmap-dendrogram-test-stats")

# %% [markdown]
# ##
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

sym_vals[~np.isfinite(sym_vals)] = 1
pdist = squareform(sym_vals)
Z = linkage(pdist, method="average", metric="euclidean")

sns.clustermap(
    sym_vals, row_linkage=Z, col_linkage=Z, xticklabels=False, yticklabels=False
)
stashfig("test-stat-clustered")


# %%
