# %% [markdown]
# # Imports
import os
import random
from operator import itemgetter
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.io import savefig, saveobj, saveskels
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True
BRAIN_VERSION = "2020-01-21"

sns.set_context("talk")

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


graph_type = "Gadn"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
edgelist_df = mg.to_edgelist()
edgelist_df["source"] = edgelist_df["source"].astype("int64")
edgelist_df["target"] = edgelist_df["target"].astype("int64")
max_pair_edge_df = edgelist_df.groupby("edge pair ID").max()
edge_max_weight_map = dict(
    zip(max_pair_edge_df.index.values, max_pair_edge_df["weight"])
)
edgelist_df["max_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
    edge_max_weight_map
)
temp_df = edgelist_df[edgelist_df["edge pair ID"] == 0]
edgelist_df.loc[temp_df.index, "max_weight"] = temp_df["weight"]
# %% [markdown]
# #
nodelist = list(mg.g.nodes())
nodelist = [int(i) for i in nodelist]
og_adj = mg.adj
g = nx.from_pandas_edgelist(edgelist_df, edge_attr=True, create_using=nx.DiGraph)
adj = nx.to_numpy_array(g, nodelist=nodelist)
heatmap(adj, transform="binarize")
heatmap(og_adj, transform="binarize")
(adj == og_adj).all()

# %% [markdown]
# #
# graph_type = "Gadn"
# mg = load_metagraph(graph_type, version=BRAIN_VERSION)
# # edgelist_df = mg.to_edgelist()
meta = mg.meta
nodelist = meta.index.values
g = nx.from_pandas_edgelist(edgelist_df, edge_attr=True, create_using=nx.DiGraph)
nx.set_node_attributes(g, meta.to_dict(orient="index"))
adj = nx.to_numpy_array(g, nodelist=nodelist, weight="max_weight")
left_paired_df = meta[(meta["Hemisphere"] == "L") & (meta["Pair"] != -1)]
left_nodes = left_paired_df.index.values
right_nodes = left_paired_df["Pair"].values
pairs = list(zip(left_nodes, right_nodes))
for (left, right) in pairs:
    if left in g and right in g:
        g = nx.contracted_nodes(g, left, right)

sym_mg = MetaGraph(g)
sym_mg.adj
#%%
is_cont = ~sym_mg.meta["contraction"].isna()
sym_mg.meta.loc[is_cont, "Hemisphere"] = "P"
#%%
heatmap(
    sym_mg.adj,
    transform="binarize",
    inner_hier_labels=sym_mg["Hemisphere"],
    cbar=False,
    figsize=(20, 20),
)

#%%
latent = lse(sym_mg.adj, n_components=None)
n_components = latent.shape[1]
class_labels = sym_mg["Class 1"]
n_unique = len(np.unique(class_labels))
sym_mg.meta["Original index"] = range(len(sym_mg.meta))
latent_cols = [f"dim {i}" for i in range(latent.shape[1])]
latent_df = pd.DataFrame(data=latent, index=sym_mg.meta.index, columns=latent_cols)
latent_df = pd.concat((sym_mg.meta, latent_df), axis=1)
latent_df.index.name = "Skeleton ID"
out_file = f"maggot_models/notebooks/outs/{FNAME}/latent.csv"
latent_df.to_csv(out_file)

#%%
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

norm_latent = latent / np.linalg.norm(latent, axis=1)[:, np.newaxis]
cos_dists = pairwise_distances(norm_latent, metric="cosine")
mds = MDS(n_components=norm_latent.shape[1] - 1, dissimilarity="precomputed")
mds_latents = mds.fit_transform(cos_dists)

