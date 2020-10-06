# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.embed import ClassicalMDS, LaplacianSpectralEmbed, OmnibusEmbed, selectSVD
from graspy.match import GraphMatch
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplace,
)
from matplotlib.collections import LineCollection
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from src.cluster import BinaryCluster
from src.data import load_metagraph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.utils import get_paired_inds
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
    set_theme,
)

meta_loc = "maggot_models/data/processed/2020-06-10/meta_data_w_order.csv"

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


# %%
order_meta = pd.read_csv(meta_loc, index_col=0)

mg = load_metagraph("G")


order_meta = order_meta[~order_meta["median_node_visits"].isna()]

mg = mg.reindex(order_meta.index, use_ids=True)

# %% [markdown]
# ##


adj = mg.adj

estimator = LaplacianSpectralEmbed(form="R-DAD", n_components=4)
adj = pass_to_ranks(adj)
embedding = estimator.fit_transform(adj)
embedding = np.concatenate(embedding, axis=1)

embed_dists = pairwise_distances(embedding, metric="cosine")

cmds = ClassicalMDS(n_components=1, dissimilarity="precomputed")
cmds_embed = cmds.fit_transform(symmetrize(embed_dists))

# %% [markdown]
# ## the original plot I was working with
set_theme()
sort_meta = order_meta
# sns.set_context("talk")
fig, ax = plt.subplots(
    1,
    1,
    figsize=(20, 10),
)
sort_meta["rand"] = np.random.uniform(size=len(sort_meta))
sort_meta["median_node_visits_jitter"] = sort_meta[
    "median_node_visits"
] + np.random.normal(0, 0.01, size=len(sort_meta))
sort_meta["cmds-1"] = (cmds_embed - cmds_embed.max()).copy()

right_inds = sort_meta[sort_meta["hemisphere"] == "R"].index
left_inds = sort_meta[sort_meta["hemisphere"] == "L"].index
sort_meta.loc[right_inds, "cmds-1"] *= -1  #
sort_meta.loc[right_inds, "cmds-1"] += 0.2
sort_meta.loc[left_inds, "cmds-1"] -= 0.2
ax.axvline(0, linewidth=2, color="grey", linestyle="--")
sns.scatterplot(
    data=sort_meta,  # [sort_meta["hemisphere"] == "L"],
    y="median_node_visits_jitter",
    x="cmds-1",
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    s=15,
    linewidth=0,
    alpha=0.7,
    legend=False,
    ax=ax,
)
ax.invert_yaxis()
ax.set(
    xticks=[],
    yticks=[],
    ylabel=r"$\leftarrow$ Motor $\quad \quad$ Sensory $\rightarrow$",
    xlabel="",
)


xs = []
ys = []
segments = []
x_key = "cmds-1"
y_key = "median_node_visits_jitter"
for pre, post in mg.g.edges():
    pre_x = sort_meta.loc[pre, x_key]
    pre_y = sort_meta.loc[pre, y_key]
    post_x = sort_meta.loc[post, x_key]
    post_y = sort_meta.loc[post, y_key]
    segments.append([(pre_x, pre_y), (post_x, post_y)])
lc = LineCollection(segments, colors="lightgrey", alpha=0.02, linewidths=0.1)
ax.add_collection(lc)
#
stashfig("sm-cmds-layout")


#%%

time_pos = order_meta["median_node_visits"].values

adj = mg.adj.copy()
adj = symmetrize(adj)

alpha = 1
estimator = LaplacianSpectralEmbed(form="R-DAD", n_components=4)
adj = pass_to_ranks(adj)
embedding = estimator.fit_transform(adj)
embedding = np.concatenate((embedding, alpha * time_pos.reshape(-1, 1)), axis=-1)

embed_dists = pairwise_distances(embedding, metric="cosine")

cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")

cmds_embed = cmds.fit_transform(symmetrize(embed_dists))
sort_meta = order_meta
# sns.set_context("talk")
fig, ax = plt.subplots(
    1,
    1,
    figsize=(20, 10),
)
sort_meta["rand"] = np.random.uniform(size=len(sort_meta))
sort_meta["median_node_visits_jitter"] = sort_meta[
    "median_node_visits"
] + np.random.normal(0, 0.01, size=len(sort_meta))
sort_meta["cmds-1"] = (cmds_embed[:, 0] - cmds_embed[:, 0].max()).copy()
sort_meta["cmds-2"] = (cmds_embed[:, 1] - cmds_embed[:, 1].max()).copy()

right_inds = sort_meta[sort_meta["hemisphere"] == "R"].index
left_inds = sort_meta[sort_meta["hemisphere"] == "L"].index
sort_meta.loc[right_inds, "cmds-2"] *= -1  #
sort_meta.loc[right_inds, "cmds-2"] += 0.2
sort_meta.loc[left_inds, "cmds-2"] -= 0.2
ax.axvline(0, linewidth=2, color="grey", linestyle="--")
sns.scatterplot(
    data=sort_meta,  # [sort_meta["hemisphere"] == "L"],
    y="median_node_visits_jitter",
    x="cmds-2",
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    s=15,
    linewidth=0,
    alpha=0.7,
    legend=False,
    ax=ax,
)
ax.invert_yaxis()
ax.set(
    xticks=[],
    yticks=[],
    ylabel=r"$\leftarrow$ Motor $\quad \quad$ Sensory $\rightarrow$",
    xlabel="",
)
stashfig("sm-cmds-layout-experimental")


#%%

time_pos = order_meta["median_node_visits"].values.copy()

adj = mg.adj.copy()
# adj = symmetrize(adj)

estimator = LaplacianSpectralEmbed(form="R-DAD", n_components=4)
adj = pass_to_ranks(adj)
embedding = estimator.fit_transform(adj)
embedding = np.concatenate(embedding, axis=-1)

embed_dists = pairwise_distances(embedding, metric="cosine")

time_dists = pairwise_distances(time_pos.reshape(-1, 1), metric="manhattan")
alpha = 10
dists = np.multiply(embed_dists, alpha * time_dists)

cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")

cmds_embed = cmds.fit_transform(symmetrize(dists))
sort_meta = order_meta
# sns.set_context("talk")
fig, ax = plt.subplots(
    1,
    1,
    figsize=(20, 10),
)
sort_meta["rand"] = np.random.uniform(size=len(sort_meta))
sort_meta["median_node_visits_jitter"] = sort_meta[
    "median_node_visits"
] + np.random.normal(0, 0.01, size=len(sort_meta))
sort_meta["cmds-1"] = (cmds_embed[:, 0] - cmds_embed[:, 0].max()).copy()
sort_meta["cmds-2"] = (cmds_embed[:, 1] - cmds_embed[:, 1].max()).copy()

right_inds = sort_meta[sort_meta["hemisphere"] == "R"].index
left_inds = sort_meta[sort_meta["hemisphere"] == "L"].index
xdim = "cmds-2"
sort_meta.loc[right_inds, xdim] *= -1  #
sort_meta.loc[right_inds, xdim] += 0.2
sort_meta.loc[left_inds, xdim] -= 0.2
ax.axvline(0, linewidth=2, color="grey", linestyle="--")
sns.scatterplot(
    data=sort_meta,  # [sort_meta["hemisphere"] == "L"],
    y="median_node_visits_jitter",
    # y="cmds-2",
    x=xdim,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    s=15,
    linewidth=0,
    alpha=0.7,
    legend=False,
    ax=ax,
)
ax.invert_yaxis()
ax.set(
    xticks=[],
    yticks=[],
    ylabel=r"$\leftarrow$ Motor $\quad \quad$ Sensory $\rightarrow$",
    xlabel="",
)
stashfig("sm-cmds-layout-experimental-2")

#%%

cluster_meta_loc = "maggot_models/notebooks/outs/137.8-BDP-omni-clust/csvs/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv"
