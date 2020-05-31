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

np.random.seed(8888)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# load data
mg = load_metagraph("G")


# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)


# compare dendrite inputs

compartment = "axon"
direction = "presynaptic"
max_samples = 500
n_subsamples = 48 * 2
method = "subsample"


def filter_connectors(connectors, ids, direction, compartment):
    label_connectors = connectors[connectors[f"{direction}_to"].isin(ids)]
    label_connectors = label_connectors[
        label_connectors[f"{direction}_type"] == compartment
    ]
    label_connectors = label_connectors[
        ~label_connectors["connector_id"].duplicated(keep="first")
    ]
    return label_connectors


def euclidean(x):
    """Default euclidean distance function calculation"""
    return pairwise_distances(X=x, metric="euclidean", n_jobs=-1)


def run_dcorr(data1, data2):
    ksamp = KSample("Dcorr", compute_distance=euclidean)
    stat, pval = ksamp.test(data1, data2, auto=True, workers=-1)
    return stat, pval


def spatial_dcorr(data1, data2, method="full", max_samples=1000, n_subsamples=5):
    if (len(data1) == 0) or (len(data2) == 0):
        return np.nan, np.nan

    if method == "subsample":
        if max(len(data1), len(data2)) < max_samples:
            method = "full"
        else:
            stats = np.empty(n_subsamples)
            p_vals = np.empty(n_subsamples)
            all_shuffles = []
            for i in range(n_subsamples):
                subsampled_data = []
                for data in [data1, data2]:
                    n_subsamples = min(len(data), max_samples)
                    inds = np.random.choice(
                        n_subsamples, size=n_subsamples, replace=False
                    )
                    subsampled_data.append(data[inds])
                all_shuffles.append(subsampled_data)
            outs = Parallel(n_jobs=-1)(delayed(run_dcorr)(*s) for s in all_shuffles)
            outs = list(zip(*outs))
            stats = outs[0]
            p_vals = outs[1]
            stat = np.median(stats)
            p_val = np.median(p_vals)
    if method == "max-d":
        max_dim_stat = -np.inf
        best_p_val = np.nan
        for dim in range(data1.shape[1]):
            dim_stat, dim_p_val = run_dcorr(data1[:, dim], data2[:, dim])
            if dim_stat > max_dim_stat:
                max_dim_stat = dim_stat
                best_p_val = dim_p_val
        stat = max_dim_stat
        p_val = best_p_val
    if method == "full":
        stat, p_val = run_dcorr(data1, data2)
    return stat, p_val


# %% [markdown]
# ##
meta = mg.meta
ids = meta.index[210:250]
p_vals = np.zeros((len(ids), len(ids)))
stats = np.zeros_like(p_vals)
cluster_meta = pd.DataFrame(index=ids)

total = comb(len(ids), k=2, exact=True)
count = 0
currtime = time.time()


for i, id1 in enumerate(ids):
    label1_connectors = filter_connectors(connectors, [id1], direction, compartment)
    for j, id2 in enumerate(ids):
        if i < j:
            print(f"Progress: {count / total:.2f}")

            label2_connectors = filter_connectors(
                connectors, [id2], direction, compartment
            )
            data1 = label1_connectors[["x", "y", "z"]].values
            print(len(data1))
            data2 = label2_connectors[["x", "y", "z"]].values
            print(len(data2))
            stat, p_val = spatial_dcorr(
                data1,
                data2,
                method=method,
                max_samples=max_samples,
                n_subsamples=n_subsamples,
            )
            stats[i, j] = stat
            p_vals[i, j] = p_val
            count += 1

print(f"\n{time.time() - currtime} elapsed\n")

from graspy.utils import symmetrize

p_vals = symmetrize(p_vals, method="triu")
stats = symmetrize(stats, method="triu")

basename = f"-compartment={compartment}-direction={direction}-method={method}"
if method == "subsample":
    basename += f"-n_sub={n_subsamples}-max_samp={max_samples}"

p_val_df = pd.DataFrame(data=p_vals, index=ids, columns=ids)
stashcsv(p_val_df, "p-vals" + basename)

stats_df = pd.DataFrame(data=stats, index=ids, columns=ids)
stashcsv(stats_df, "test-stats" + basename)

plot_p_vals = -np.log10(p_vals)
plt.figure()
adjplot(
    plot_p_vals,
    # meta=cluster_meta,
    vmax=np.nanmax(plot_p_vals[~np.isinf(plot_p_vals)]),
    cbar_kws=dict(shrink=0.7),
    cbar=True,
    cmap="Reds",
)
stashfig("p-val-plot" + basename)

plt.figure(figsize=(10, 10))
sns.heatmap(
    stats,
    cmap="Reds",
    cbar_kws=dict(shrink=0.7),
    square=True,
    xticklabels=False,
    yticklabels=False,
)
stashfig("stats-plot" + basename)

