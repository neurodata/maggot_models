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

meta = mg.meta


#%%

graph_types = ["Gaa", "Gad", "Gda", "Gdd"]

graphs = {}

for graph_type in graph_types:
    temp_mg = load_metagraph(graph_type)
    temp_mg = temp_mg.reindex(meta.index, use_ids=True)
    assert (temp_mg.meta.index.values == meta.index.values).all()
    graphs[graph_type] = temp_mg.adj

#%%
from graspologic.utils import get_lcc
from scipy.stats import rankdata

for graph_type in graph_types:
    adj = graphs[graph_type]
    adj_lcc, inds = get_lcc(adj, return_inds=True)
    ranks = sr.get_ranks(adj_lcc)
    meta[f"{graph_type}_sr_score"] = np.nan
    meta[f"{graph_type}_sr_rank"] = np.nan
    meta.loc[meta.index[inds], f"{graph_type}_sr_score"] = ranks
    spring_rank = rankdata(ranks)
    meta.loc[meta.index[inds], f"{graph_type}_sr_rank"] = spring_rank

#%%

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


#%%
from scipy.stats import spearmanr, pearsonr

set_theme()
hue_key = "simple_class"
var = "sr_score"
n_graphs = 4

fig, axs = plt.subplots(n_graphs, n_graphs, figsize=(16, 16))
for i, row_graph in enumerate(graph_types):
    for j, col_graph in enumerate(graph_types):

        x_var = f"{col_graph}_{var}"
        y_var = f"{row_graph}_{var}"

        spearman_corr, _ = spearmanr(meta[x_var], meta[y_var], nan_policy="omit")

        ax = axs[i, j]
        if i > j:
            sns.scatterplot(
                data=meta,
                x=x_var,
                y=y_var,
                hue=hue_key,
                palette=CLASS_COLOR_DICT,
                ax=ax,
                s=5,
                alpha=0.5,
                linewidth=0,
                legend=False,
            )
            text = ax.text(
                0.98,
                0.03,
                r"$\rho = $" + f"{spearman_corr:0.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="black",
            )
            text.set_bbox(dict(facecolor="white", alpha=0.6, edgecolor="w"))
        elif i == j:
            sns.histplot(
                data=meta,
                x=x_var,
                ax=ax,
                bins=50,
                element="step",
                # color="grey",
                hue=hue_key,
                palette=CLASS_COLOR_DICT,
                legend=False,
                stat="density",
                common_norm=True,
            )
        else:
            ax.axis("off")
        ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
        if i == n_graphs - 1:
            ax.set(xlabel=f"{col_graph}")
        if j == 0:
            ax.set(ylabel=f"{row_graph}")
    stashfig(f"{var}-pairwise")
# %%

for graph_type in graph_types:
    adj = graphs[graph_type]
    adj_lcc, inds = get_lcc(adj, return_inds=True)
    ranks = sr.get_ranks(adj_lcc)
    beta = sr.get_inverse_temperature(adj_lcc, ranks)
    print(beta)

#%%
A = adj_lcc.copy()

ranks = sr.get_ranks(A)
beta = sr.get_inverse_temperature(A, ranks)


def estimate_spring_rank_P(A, ranks, beta):
    H = ranks[:, None] - ranks[None, :] - 1
    H = np.multiply(H, H)
    H *= 0.5
    P = np.exp(-beta * H)
    P *= np.mean(A) / np.mean(P)
    return P


#%%
from graspologic.plot import heatmap
from src.visualization import adjplot


#%%


for graph_type in graph_types:
    adj = graphs[graph_type]
    A, inds = get_lcc(adj, return_inds=True)
    ranks = sr.get_ranks(A)
    beta = sr.get_inverse_temperature(A, ranks)
    P = estimate_spring_rank_P(A, ranks, beta)
    sort_inds = np.argsort(-ranks)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    adjplot(P[np.ix_(sort_inds, sort_inds)], ax=axs[0], cbar=False, title=r"$\hat{P}$")
    adjplot(
        A[np.ix_(sort_inds, sort_inds)],
        plot_type="scattermap",
        ax=axs[1],
        sizes=(1, 1),
        title=r"$A$",
    )
    stashfig(f"{graph_type}-sr-prob-model")

#%%
sort_inds = np.argsort(-ranks)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
adjplot(P[np.ix_(sort_inds, sort_inds)], ax=axs[0], cbar=False)
adjplot(A[np.ix_(sort_inds, sort_inds)], plot_type="scattermap", ax=axs[1])
