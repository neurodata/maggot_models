# %% [markdown]
# #
import os
import urllib.request
from pathlib import Path

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graph_tool import load_graph
from graph_tool.inference import minimize_blockmodel_dl
from joblib import Parallel, delayed
from random_word import RandomWords
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from operator import itemgetter
from graspy.utils import cartprod
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.utils import get_blockmodel_df
from src.visualization import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    barplot_text,
    probplot,
    remove_spines,
    stacked_barplot,
)
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
BRAIN_VERSION = "2020-01-29"


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def add_max_weight(df):
    max_pair_edges = df.groupby("edge pair ID", sort=False)["weight"].max()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    df["max_weight"] = itemgetter(*df["edge pair ID"])(edge_max_weight_map)
    asym_inds = df[df["edge pair ID"] == -1].index
    df.loc[asym_inds, "max_weight"] = df.loc[asym_inds, "weight"]
    return df


def edgelist_to_mg(edgelist, meta):
    g = nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=nx.DiGraph)
    nx.set_node_attributes(g, meta.to_dict(orient="index"))
    mg = MetaGraph(g)
    return mg


run_dir = Path("81.1-BDP-community")
base_dir = Path("./maggot_models/notebooks/outs")
block_file = base_dir / run_dir / "csvs" / "block-labels.csv"
block_df = pd.read_csv(block_file, index_col=0)

run_names = block_df.columns.values
n_runs = len(block_df.columns)
block_pairs = cartprod(range(n_runs), range(n_runs))

param_file = base_dir / run_dir / "csvs" / "parameters.csv"
param_df = pd.read_csv(param_file, index_col=0)
param_df.set_index("param_key", inplace=True)
param_groupby = param_df.groupby(["graph_type", "threshold", "res", "binarize"])
param_df["Parameters"] = -1
for i, (key, val) in enumerate(param_groupby.indices.items()):
    param_df.iloc[val, param_df.columns.get_loc("Parameters")] = i

# %% [markdown]
# # Look at modularity over all of the parameters
sns.set_context("talk", font_scale=1)

mean_modularities = param_df.groupby("Parameters")["modularity"].mean()
order = mean_modularities.sort_values(ascending=False).index
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=param_df, x="Parameters", y="modularity", ax=ax, order=order, jitter=0.4
)
ax.set_xlabel("Parameter set")
stashfig("mod-by-parameters")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=param_df, x="threshold", y="modularity", ax=ax, jitter=0.4)
stashfig("mod-by-threshold")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=param_df, x="binarize", y="modularity", ax=ax, jitter=0.4)
stashfig("mod-by-binarize")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=param_df, x="res", y="modularity", ax=ax, jitter=0.4)
stashfig("mod-by-res")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=param_df,
    x="threshold",
    y="modularity",
    ax=ax,
    hue="Parameters",
    palette=cc.glasbey_light,
    jitter=0.45,
)
ax.legend([])
stashfig("mod-by-threshold-colored")

# %% [markdown]
# #
max_inds = param_df.groupby("Parameters")["modularity"].idxmax().values
best_param_df = param_df.loc[max_inds]
best_block_df = block_df[max_inds]
n_runs = len(max_inds)
block_pairs = cartprod(range(n_runs), range(n_runs))
ari_mat = np.empty((n_runs, n_runs))
for bp in block_pairs:
    from_block_labels = best_block_df.iloc[:, bp[0]].values
    to_block_labels = best_block_df.iloc[:, bp[1]].values
    mask = np.logical_and(~np.isnan(from_block_labels), ~np.isnan(to_block_labels))
    from_block_labels = from_block_labels[mask]
    to_block_labels = to_block_labels[mask]
    ari = adjusted_rand_score(from_block_labels, to_block_labels)
    ari_mat[bp[0], bp[1]] = ari
ari_df = pd.DataFrame(data=ari_mat, index=max_inds, columns=max_inds)

sns.set_context("talk", font_scale=0.8)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.heatmap(
    data=ari_df,
    cmap="Reds",
    annot=False,
    square=True,
    ax=ax,
    cbar_kws=dict(shrink=0.7),
    vmin=0,
)
stashfig("ari-heatmap")


# %% [markdown]
# #


sns.set_context("talk", font_scale=1)

lut = dict(zip(param_df["Parameters"].unique(), cc.glasbey_light))
row_colors = param_df["Parameters"].map(lut)
clustergrid = sns.clustermap(
    ari_df,
    cmap="RdBu_r",
    center=0,
    method="single",
    vmin=None,
    figsize=(20, 20),
    row_colors=row_colors,
    col_colors=row_colors,
    dendrogram_ratio=0.2,
)
clustergrid.fig.suptitle("ARI", y=1.02)
stashfig("ari-clustermap")


# %% [markdown]
# #
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(
    data=best_param_df, x="Parameters", y="modularity", ax=ax, order=order, jitter=0.4
)
ax.set_xlabel("Parameter set")
stashfig("mod-by-parameters")

