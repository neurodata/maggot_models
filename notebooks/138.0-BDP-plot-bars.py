# %% [markdown]
# ##
import os
import warnings

import matplotlib as mpl
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
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance

# %%
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    plot_neurons,
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


# %% [markdown]
# ##

n_levels = 10  # max # of splits
metric = "bic"
bic_ratio = 1
d = 10  # embedding dimension
method = "aniso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.0-BDP-omni-clust"

from src.io import readcsv
from src.graph import MetaGraph

meta = readcsv("meta" + basename, foldername=exp, index_col=0)
adj = readcsv("adj" + basename, foldername=exp, index_col=0)
mg = MetaGraph(adj.values, meta)

# %% [markdown] 
# ## 




# %% [markdown]
# ## Define what we want to look at
n_pairs = len(mg) // 2
new_lp_inds = np.arange(n_pairs)
new_rp_inds = np.arange(n_pairs) + n_pairs
names = ["iso", "aniso"]


# %% [markdown]
# ## Look at the best one! (ish)

# new_meta = meta.iloc[np.concatenate((lp_inds, rp_inds), axis=0)].copy()
# labels = new_meta["merge_class"].values
# plot_pairs(
#     ase_flat_embed[:, :8],
#     labels,
#     left_pair_inds=new_lp_inds,
#     right_pair_inds=new_rp_inds,
# )
# stashfig("ase-flat-pairs")

# quick_embed_viewer(
#     ase_flat_embed[:, :8], labels=labels, lp_inds=new_lp_inds, rp_inds=new_rp_inds
# )
# stashfig("ase-flat-manifold")

# %% [markdown]
# ## Cluster


n_levels = 10  # max # of splits
metric = "bic"
bic_ratio = 1
d = 10  # embedding dimension
method = "aniso"
if method == "aniso":
    X = svd_aniso_embed
elif method == "iso":
    X = svd_iso_embed
X = X[:, :d]
basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

np.random.seed(8888)
mc = BinaryCluster(
    "0",
    adj=adj,
    n_init=25,
    meta=new_meta,
    stashfig=stashfig,
    X=X,
    bic_ratio=bic_ratio,
    reembed=False,
    min_split=4,
)

mc.fit(n_levels=n_levels, metric=metric)

n_levels = mc.height

fig, axs = plt.subplots(1, n_levels, figsize=(8 * n_levels, 30))
for i in range(n_levels):
    ax = axs[i]
    stacked_barplot(
        mc.meta[f"lvl{i}_labels_side"],
        mc.meta["merge_class"],
        category_order=np.unique(mc.meta[f"lvl{i}_labels_side"].values),
        color_dict=CLASS_COLOR_DICT,
        norm_bar_width=False,
        ax=ax,
    )
    ax.set_yticks([])
    ax.get_legend().remove()
    ax.set_title(title)

plt.tight_layout()

stashfig(f"count-barplot-lvl{i}" + basename)
plt.close()


inds = np.concatenate((lp_inds, rp_inds))
new_adj = adj[np.ix_(inds, inds)]
new_meta = mc.meta
new_meta["sf"] = -signal_flow(new_adj)

for l in range(n_levels):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    sort_class = [f"lvl{i}_labels" for i in range(l)]
    sort_class += [f"lvl{l}_labels_side"]
    _, _, top, _ = adjplot(
        new_adj,
        meta=new_meta,
        sort_class=sort_class,
        item_order="merge_class",
        plot_type="scattermap",
        class_order="sf",
        sizes=(0.5, 1),
        ticks=False,
        colors="merge_class",
        ax=ax,
        palette=CLASS_COLOR_DICT,
        gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
    )
    top.set_title(title + f" level={l}")
    stashfig(f"adj-lvl{l}" + basename)
    plt.close()

stashcsv(new_meta, "meta" + basename)
adj_df = pd.DataFrame(new_adj, index=new_meta.index, columns=new_meta.columns)
stashcsv(adj_df, "adj" + basename)

# %% [markdown]
# ##
pairs = np.unique(new_meta["Pair ID"])
p_same_clusters = []
p_same_chance = []
rows = []
n_shuffles = 10
for l in range(n_levels):
    n_same = 0
    pred_labels = new_meta[f"lvl{l}_labels"].values.copy()
    left_labels = pred_labels[new_lp_inds]
    right_labels = pred_labels[new_rp_inds]
    n_same = (left_labels == right_labels).sum()
    p_same = n_same / len(pairs)
    rows.append(dict(p_same_cluster=p_same, labels="True", level=l))

    # look at random chance
    for i in range(n_shuffles):
        np.random.shuffle(pred_labels)
        left_labels = pred_labels[new_lp_inds]
        right_labels = pred_labels[new_rp_inds]
        n_same = (left_labels == right_labels).sum()
        p_same = n_same / len(pairs)
        rows.append(dict(p_same_cluster=p_same, labels="Shuffled", level=l))

plot_df = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(data=plot_df, x="level", y="p_same_cluster", ax=ax, hue="labels")
ax.set_ylabel("P same cluster")
ax.set_xlabel("Level")
ax.set_title(title)
stashfig("p_in_same_cluster" + basename)

n_clusters = []
for l in range(n_levels):
    n_clusters.append(new_meta[f"lvl{l}_labels"].nunique())

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(x=range(n_levels), y=n_clusters, ax=ax)
sns.scatterplot(x=range(n_levels), y=n_clusters, ax=ax)
ax.set_ylabel("Clusters per side")
ax.set_xlabel("Level")
ax.set_title(title)
stashfig("n_cluster" + basename)

size_dfs = []
for l in range(n_levels):
    sizes = new_meta.groupby(f"lvl{l}_labels_side").size().values
    sizes = pd.DataFrame(data=sizes, columns=["Size"])
    sizes["Level"] = l
    size_dfs.append(sizes)

size_df = pd.concat(size_dfs)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=size_df, x="Level", y="Size", ax=ax, jitter=0.45, alpha=0.5)
ax.set_yscale("log")
ax.set_title(title)
stashfig("log-sizes" + basename)

# %% [markdown]
# ## Fit models and compare L/R

rows = []

for l in range(n_levels):
    labels = new_meta[f"lvl{l}_labels"].values
    left_adj = binarize(new_adj[np.ix_(new_lp_inds, new_lp_inds)])
    left_adj = remove_loops(left_adj)
    right_adj = binarize(new_adj[np.ix_(new_rp_inds, new_rp_inds)])
    right_adj = remove_loops(right_adj)
    for model, name in zip([DCSBMEstimator, SBMEstimator], ["DCSBM", "SBM"]):
        estimator = model(directed=True, loops=False)
        uni_labels, inv = np.unique(labels, return_inverse=True)
        estimator.fit(left_adj, inv[new_lp_inds])
        train_left_p = estimator.p_mat_
        train_left_p[train_left_p == 0] = 1 / train_left_p.size

        score = poisson.logpmf(left_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="left",
                test="same",
                test_side="left",
                score=score,
                level=l,
                model=name,
            )
        )
        score = poisson.logpmf(right_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="left",
                test="opposite",
                test_side="right",
                score=score,
                level=l,
                model=name,
            )
        )

        estimator = model(directed=True, loops=False)
        estimator.fit(right_adj, inv[new_rp_inds])
        train_right_p = estimator.p_mat_
        train_right_p[train_right_p == 0] = 1 / train_right_p.size

        score = poisson.logpmf(left_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="right",
                test="opposite",
                test_side="left",
                score=score,
                level=l,
                model=name,
            )
        )
        score = poisson.logpmf(right_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="right",
                test="same",
                test_side="right",
                score=score,
                level=l,
                model=name,
            )
        )


# %% [markdown]
# ## Plot model results

plot_df = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
model_name = "SBM"
sns.lineplot(
    data=plot_df[plot_df["model"] == model_name],
    hue="test",
    x="level",
    y="score",
    style="train_side",
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.set_title(title)
ax.set_ylabel(f"{model_name} log lik.")
stashfig("sbm-lik-curves" + basename)

model_name = "DCSBM"
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    data=plot_df[plot_df["model"] == model_name],
    hue="test",
    x="level",
    y="score",
    style="train_side",
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.set_title(title)
ax.set_ylabel(f"{model_name} log lik.")
stashfig("dcsbm-lik-curves" + basename)


# %% [markdown]
# ## Plot neurons
lvl = 4
show_neurons = False

if show_neurons:
    uni_labels = np.unique(new_meta[f"lvl{lvl}_labels"])
    start_instance()

    for label in uni_labels:
        plot_neurons(new_meta, f"lvl{lvl}_labels", label=label, barplot=True)
        stashfig(f"label{label}_lvl{lvl}" + basename)
