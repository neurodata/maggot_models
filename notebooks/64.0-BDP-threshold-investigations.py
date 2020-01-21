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
BRAIN_VERSION = "2020-01-16"

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


# %% [markdown]
# # Play with getting edge df representatino
graph_type = "Gadn"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)

g = mg.g
meta = mg.meta
edgelist_df = mg.to_edgelist()

max_pair_edge_df = edgelist_df.groupby("edge pair ID").max()
edge_max_weight_map = dict(
    zip(max_pair_edge_df.index.values, max_pair_edge_df["weight"])
)
edgelist_df["max_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
    edge_max_weight_map
)

# %% [markdown]
# # Try thresholding in this new format
props = []
prop_edges = []
prop_syns = []
threshs = np.linspace(0, 0.3, 20)
for threshold in threshs:
    thresh_df = max_pair_edge_df[max_pair_edge_df["weight"] > threshold]
    prop = len(thresh_df[thresh_df["edge pair counts"] == 2]) / len(thresh_df)
    props.append(prop)
    prop_edges_left = (
        thresh_df["edge pair counts"].sum() / max_pair_edge_df["edge pair counts"].sum()
    )
    prop_edges.append(prop_edges_left)
    temp_df = edgelist_df[edgelist_df["max_weight"] > threshold]
    p_syns = temp_df["weight"].sum() / edgelist_df["weight"].sum()
    prop_syns.append(p_syns)


# %% [markdown]
# # threshold curves for cells w > 100 dendritic inputs
# # threshold curves for cells w > 50 dendritic inputs


# %% [markdown]
# # plot the distribution of # of dendritic / axonic inputs

# %% [markdown]
# # (if time) plot the distribution of # of inputs by lineage

# %% [markdown]
# # plot some kind of asymmetry score by lineage
# # - proportion of edges onto a lineage which are asymmetric after thresholding
# # - IOU score?
# # - something else?

# %% [markdown]
# # get number of inputs to kenyon cells
# # just list the number of connections onto each kenyon cell, by claw number
