# %% [markdown]
# #

from src.graph import MetaGraph, preprocess
from src.data import load_metagraph
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


def get_edges(adj):
    edgemat = np.empty((len(adj), 2 * len(adj)))
    for i in range(len(adj)):
        edgemat[i, : len(adj)] = adj[i, :]
        edgemat[i, len(adj) :] = adj[:, i]
    return edgemat


rl_nblast_loc = "maggot_models/data/external/Brain-NBLAST_R-to-L.csv"
lr_nblast_loc = "maggot_models/data/external/Brain-NBLAST_L-to-R.csv"

rl_nblast_df = pd.read_csv(rl_nblast_loc, index_col=0)
lr_nblast_df = pd.read_csv(lr_nblast_loc, index_col=0)
lr_nblast_df.columns = lr_nblast_df.columns.astype(int)
rl_nblast_df.columns = rl_nblast_df.columns.astype(int)
sns.distplot(rl_nblast_df.values)
sns.distplot(lr_nblast_df.values)

# %% [markdown]
# #
VERSION = "2020-03-09"
graph_types = ["Gad", "Gaa", "Gda", "Gdd"]
weight = "weight"
mg = load_metagraph("G", VERSION)
last_index = mg.meta.index
meta = mg.meta.copy()

adjs = []
for graph_type in graph_types:
    mg = load_metagraph(graph_type, VERSION)
    assert np.array_equal(last_index, mg.meta.index)
    last_index = mg.meta.index
    adj = mg.adj
    adjs.append(adj)

edgemats = []
for adj in adjs:
    edgemat = get_edges(adj)
    edgemats.append(edgemat)
full_edgemat = np.concatenate(edgemats, axis=-1)


# %% [markdown]
# #

from sklearn.metrics import pairwise_distances

pdists = pairwise_distances(full_edgemat, metric="cosine")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(pdists, cmap="RdBu_r", center=0, square=True, ax=ax)

#%%
adj_index = last_index

# %% [markdown]
# #
def get_hemisphere_indices(mg):
    meta = mg.meta.copy()
    meta["Original index"] = range(len(meta))
    left_meta = meta[(meta["Hemisphere"] == "L") & (meta["Pair"] != -1)].copy()
    right_meta = meta[(meta["Hemisphere"] == "R") & (meta["Pair"] != -1)].copy()
    left_meta.sort_values("Pair ID", inplace=True)
    right_meta.sort_values("Pair ID", inplace=True)
    assert np.array_equal(left_meta["Pair ID"].values, right_meta["Pair ID"].values)

    left_inds = left_meta["Original index"].values
    right_inds = right_meta["Original index"].values
    return left_inds, right_inds


left_inds, right_inds = get_hemisphere_indices(mg)
left_left_adj = adj[np.ix_(left_inds, left_inds)]
right_right_adj = adj[np.ix_(right_inds, right_inds)]
left_edges = get_edges(left_left_adj)
right_edges = get_edges(right_right_adj)


#%%
lr_cos_dists = pairwise_distances(left_edges, right_edges, metric="cosine")

left_ids = meta.iloc[left_inds].index
right_ids = meta.iloc[right_inds].index

# lr_ids = meta.iloc[lr_index].index
# %%
lr_nblast_df = lr_nblast_df.reindex(left_ids)
lr_nblast_df = lr_nblast_df.reindex(columns=right_ids)
lr_nblast_dists = lr_nblast_df.values

# %%
sns.scatterplot(lr_nblast_dists.ravel(), lr_cos_dists.ravel())
sns.jointplot(lr_nblast_dists.ravel(), lr_cos_dists.ravel(), kind="hex")


# %% [markdown]
# #
sns.distplot(lr_cos_dists.ravel(), kde=None)

sns.distplot(lr_cos_dists.ravel()[lr_cos_dists.ravel() != 1], kde=None)

# %% [markdown]
# #
cos_dists = lr_cos_dists.ravel()[lr_cos_dists.ravel() != 1]
nblast_dists = lr_nblast_dists.ravel()[lr_cos_dists.ravel() != 1]
isna = np.isnan(nblast_dists)
cos_dists = cos_dists[~isna]
nblast_dists = nblast_dists[~isna]
nblast_dists -= nblast_dists.min()
nblast_dists /= nblast_dists.max()
nblast_dists *= -1
nblast_dists += 1
sns.jointplot(nblast_dists, cos_dists, kind="hex")
