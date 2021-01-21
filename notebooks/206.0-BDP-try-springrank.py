#%%
import logging
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats.mstats import gmean
from sklearn.preprocessing import QuantileTransformer
from umap import UMAP
import navis
import pymaid
from giskard.plot import (
    dissimilarity_clustermap,
    screeplot,
    simple_scatterplot,
    simple_umap_scatterplot,
    stacked_barplot,
)
from giskard.stats import calc_discriminability_statistic
from giskard.utils import careys_rule
from graspologic.cluster import DivisiveCluster
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, selectSVD
from graspologic.plot import pairplot
from graspologic.utils import pass_to_ranks, remap_labels, symmetrize
from src.data import load_metagraph
from src.embed import JointEmbed, unscale
from src.io import savefig
from src.metrics import calc_model_liks, plot_pairedness
from src.pymaid import start_instance
from src.visualization import CLASS_COLOR_DICT, adjplot, set_theme, simple_plot_neurons

import SpringRank as sr

mg = load_metagraph("G")
mg.make_lcc()

adj = mg.adj

ranks = sr.get_ranks(adj)

meta = mg.meta

meta["sr_ranks"] = ranks

#%%

meta["random"] = np.random.uniform(size=len(meta))


#%%


set_theme()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=meta,
    x="random",
    y="sr_ranks",
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    ax=ax,
    s=10,
)
ax.get_legend().remove()


# %%
from src.hierarchy import signal_flow

meta["sf_ranks"] = signal_flow(adj)


#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=meta,
    x="random",
    y="sf_ranks",
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    ax=ax,
    s=10,
)
ax.get_legend().remove()

#%%

from SpringRank.SpringRank import SpringRank

meta["sr_ranks"] = SpringRank(pass_to_ranks(adj), alpha=2)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=meta,
    x="sf_ranks",
    y="sr_ranks",
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    ax=ax,
    s=10,
)
ax.get_legend().remove()

# %%
from src.visualization import adjplot

adjplot(np.maximum(adj, adj.T), plot_type="scattermap")
#%%

winner_adj = np.zeros_like(adj)
for i in range(len(adj)):
    for j in range(len(adj)):
        if adj[i, j] > adj[j, i]:
            winner_adj[i, j] = 1

#%%
meta["sr_ranks"] = SpringRank(winner_adj, alpha=2)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=meta,
    x="sf_ranks",
    y="sr_ranks",
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    ax=ax,
    s=10,
)
ax.get_legend().remove()