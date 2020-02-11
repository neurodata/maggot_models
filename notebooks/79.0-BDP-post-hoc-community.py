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

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
BRAIN_VERSION = "2020-01-29"

print(sns.__version__)


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


run_dir = Path("81.0-BDP-community")
base_dir = Path("./maggot_models/notebooks/outs")
block_file = base_dir / run_dir / "csvs" / "block-labels.csv"
block_df = pd.read_csv(block_file, index_col=0)

run_names = block_df.columns.values
n_runs = len(block_df.columns)
block_pairs = cartprod(range(n_runs), range(n_runs))

ari_mat = np.empty((n_runs, n_runs))
for bp in block_pairs:
    from_block_labels = block_df.iloc[:, bp[0]].values
    to_block_labels = block_df.iloc[:, bp[1]].values
    mask = np.logical_and(~np.isnan(from_block_labels), ~np.isnan(to_block_labels))
    from_block_labels = from_block_labels[mask]
    to_block_labels = to_block_labels[mask]
    ari = adjusted_rand_score(from_block_labels, to_block_labels)
    ari_mat[bp[0], bp[1]] = ari
ari_df = pd.DataFrame(data=ari_mat, index=run_names, columns=run_names)

sns.set_context("talk", font_scale=0.8)
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.heatmap(
    data=ari_df,
    cmap="Reds",
    annot=True,
    square=True,
    ax=ax,
    cbar_kws=dict(shrink=0.7),
    vmin=0,
)
stashfig("ari-heatmap")


# %% [markdown]
# #
param_file = base_dir / run_dir / "csvs" / "parameters.csv"
param_df = pd.read_csv(param_file, index_col=0)
param_df.set_index("param_key", inplace=True)
param_groupby = param_df.groupby(["graph_type", "threshold"])
param_df["Parameters"] = -1
for i, (key, val) in enumerate(param_groupby.indices.items()):
    param_df.iloc[val, param_df.columns.get_loc("Parameters")] = i


sns.set_context("talk", font_scale=1)

lut = dict(zip(param_df["Parameters"].unique(), sns.color_palette("deep")))
row_colors = param_df["Parameters"].map(lut)
clustergrid = sns.clustermap(
    ari_df,
    cmap="Reds",
    method="single",
    annot=True,
    vmin=None,
    figsize=(20, 20),
    row_colors=row_colors,
    col_colors=row_colors,
    dendrogram_ratio=0.1,
    cbar_pos=None,
)
clustergrid.fig.suptitle("ARI", y=1.02)
stashfig("ari-clustermap")

# %% [markdown]
# # evaluate modularity


from community import modularity

for c in block_df.columns:
    partition = block_df[c]
    params = param_df.loc[c]
    mg = load_metagraph(params["graph_type"], version=BRAIN_VERSION)
    edgelist = mg.to_edgelist()
    edgelist = add_max_weight(edgelist)
    edgelist = edgelist[edgelist["max_weight"] > params["threshold"]]
    mg = edgelist_to_mg(edgelist, mg.meta)
    mg = mg.make_lcc()
    mg = mg.remove_pdiff()
    g_sym = nx.to_undirected(mg.g)
    modularity(partition, g)
