# %% [markdown]
# ##
import os
import warnings
from itertools import chain

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import poisson
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm
from umap import UMAP

from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import pairplot
from graspy.simulations import sbm
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplace,
)
from src.align import Procrustes
from src.cluster import BinaryCluster, MaggotCluster, get_paired_inds
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.pymaid import start_instance
from src.traverse import Cascade, RandomWalk, to_markov_matrix, to_transmission_matrix
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    remove_spines,
    screeplot,
    set_axes_equal,
    stacked_barplot,
)

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


ks = np.arange(1, 100)
n = 2310

base_df = pd.DataFrame()
base_df["K"] = ks

# def k_squared(k, *args):

funcs = [lambda k, n: k ** 2, lambda k, n: k ** 2 + n, lambda k, n: k ** 2 + 2 * n]
names = [r"$K^2$", r"$K^2 + N$", r"$K^2 + 2N$"]

dfs = []
for func, name in zip(funcs, names):
    func_df = base_df.copy()
    func_df["n_params"] = func(ks, n)
    func_df["func"] = name
    dfs.append(func_df)

plot_df = pd.concat(dfs, axis=0)

sns.lineplot(data=plot_df, x="K", y="n_params", hue="func")

# %%
metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}-G"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.0-BDP-omni-clust"

# load data
meta = readcsv("meta" + basename, foldername=exp, index_col=0)
meta["lvl0_labels"] = meta["lvl0_labels"].astype(str)
adj = readcsv("adj" + basename, foldername=exp, index_col=0)
mg = MetaGraph(adj.values, meta)

meta = mg.meta
adj = mg.adj
lp_inds, rp_inds = get_paired_inds(meta)


# parameters
lowest_level = 7
n_levels = 10


# %% [markdown]
# ##
rows = []


class DDCSBMEstimator(DCSBMEstimator):
    def __init__(self, **kwargs):
        super().__init__(degree_directed=True, **kwargs)


for l in range(n_levels + 1):
    labels = meta[f"lvl{l}_labels"].values
    left_adj = binarize(adj[np.ix_(lp_inds, lp_inds)])
    left_adj = remove_loops(left_adj)
    right_adj = binarize(adj[np.ix_(rp_inds, rp_inds)])
    right_adj = remove_loops(right_adj)
    for model, name in zip(
        [DDCSBMEstimator, DCSBMEstimator, SBMEstimator], ["DDCSBM", "DCSBM", "SBM"]
    ):
        # train on left
        estimator = model(directed=True, loops=False)
        uni_labels, inv = np.unique(labels, return_inverse=True)
        estimator.fit(left_adj, inv[lp_inds])
        train_left_p = estimator.p_mat_
        train_left_p[train_left_p == 0] = 1 / train_left_p.size

        n_params = estimator._n_parameters() + len(uni_labels)

        # test on left
        score = poisson.logpmf(left_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Same",
                test_side="Left",
                score=score,
                level=l,
                model=name,
                n_params=n_params,
                norm_score=score / left_adj.sum(),
                K=len(uni_labels),
            )
        )
        # test on right
        score = poisson.logpmf(right_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Opposite",
                test_side="Right",
                score=score,
                level=l,
                model=name,
                n_params=n_params,
                norm_score=score / right_adj.sum(),
                K=len(uni_labels),
            )
        )

        # train on right
        estimator = model(directed=True, loops=False)
        estimator.fit(right_adj, inv[rp_inds])
        train_right_p = estimator.p_mat_
        train_right_p[train_right_p == 0] = 1 / train_right_p.size

        n_params = estimator._n_parameters() + len(uni_labels)
        # test on left
        score = poisson.logpmf(left_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Opposite",
                test_side="Left",
                score=score,
                level=l,
                model=name,
                n_params=n_params,
                norm_score=score / left_adj.sum(),
                K=len(uni_labels),
            )
        )
        # test on right
        score = poisson.logpmf(right_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Same",
                test_side="Right",
                score=score,
                level=l,
                model=name,
                n_params=n_params,
                norm_score=score / right_adj.sum(),
                K=len(uni_labels),
            )
        )

plot_df = pd.DataFrame(rows)

sns.set_palette(sns.color_palette("deep"))
model_name = "DCSBM"
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
ax = axs[0]
sns.lineplot(
    data=plot_df[plot_df["model"] == model_name],
    hue="test",
    x="level",
    y="norm_score",
    style="train_side",
    ax=ax,
    markers=True,
)
ax.get_legend().remove()
ax.set_ylabel("Normalized log likelihood")

ax = axs[1]
sns.lineplot(
    data=plot_df[plot_df["model"] == model_name],
    hue="test",
    x="K",
    y="norm_score",
    style="train_side",
    ax=ax,
    markers=True,
)
ax.get_legend().remove()


ax = axs[2]
sns.lineplot(
    data=plot_df[plot_df["model"] == model_name],
    hue="test",
    x="n_params",
    y="norm_score",
    style="train_side",
    ax=ax,
    markers=True,
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.set_ylabel("Normalized log likelihood")
n_edges = np.mean((left_adj.sum(), right_adj.sum()))
ax.axvline(n_edges, linestyle="--", color="red")
stashfig("dcsbm-log-likelihood")
# ax.axis("off")

# ax = axs[1, 1]

# sns.lineplot(
#     data=plot_df[plot_df["model"] == model_name],
#     hue="test",
#     x="K",
#     y="norm_score",
#     style="train_side",
#     ax=ax,
#     markers=True,
# )
# ax.get_legend().remove()

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    data=plot_df[
        (plot_df["test"] == "Opposite")
    ],  # & (plot_df["train_side"] == "Left")],
    y="norm_score",
    x="n_params",
    hue="model",
    style="train_side",
    markers=True,
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.axvline(n_edges, linestyle="--", color="red")
stashfig("param-comparison")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    data=plot_df[
        (plot_df["test"] == "Opposite")
    ],  # & (plot_df["train_side"] == "Left")],
    y="norm_score",
    x="level",
    hue="model",
    style="train_side",
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
# ax.axvline(n_edges, linestyle="--", color="red")


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    data=plot_df[
        (plot_df["test"] == "Opposite")
    ],  # & (plot_df["train_side"] == "Left")],
    y="n_params",
    x="level",
    hue="model",
    style="train_side",
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.axhline(n_edges, linestyle="--", color="red")
stashfig("n_params")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    data=plot_df[
        (plot_df["test"] == "Opposite")
    ],  # & (plot_df["train_side"] == "Left")],
    y="K",
    x="norm_score",
    hue="model",
    style="train_side",
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

