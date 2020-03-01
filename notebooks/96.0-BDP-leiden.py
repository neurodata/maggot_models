# %% [markdown]
# #

import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import community as cm
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from sklearn.model_selection import ParameterGrid

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.data import load_everything, load_metagraph, load_networkx
from src.embed import lse, preprocess_graph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels, savecsv
from src.utils import get_blockmodel_df, get_sbm_prob
from src.visualization import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    barplot_text,
    bartreeplot,
    draw_networkx_nice,
    get_color_dict,
    get_colors,
    palplot,
    probplot,
    sankey,
    screeplot,
    stacked_barplot,
    random_names,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2020-03-01"
BLIND = True
SAVEFIGS = False
SAVESKELS = False
SAVEOBJS = True

np.random.seed(9812343)
sns.set_context("talk")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)
    plt.close()


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


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


def stashobj(obj, name, **kws):
    saveobj(obj, name, foldername=FNAME, save_on=SAVEOBJS, **kws)


graph_type = "G"
threshold = 0
binarize = True

# load and preprocess the data
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
mg = preprocess(
    mg, threshold=threshold, sym_threshold=True, remove_pdiff=True, binarize=binarize
)

from src.block import run_leiden, run_leiden_igraph

partition = run_leiden(mg)


# %% [markdown]
# #


out = run_leiden_igraph(mg)

# adj = mg.adj
# adj = symmetrize(adj, method="avg")
# mg = MetaGraph(adj, mg.meta)
# g_sym = mg.g
# # g_sym = nx.to_undirected(mg.g)
# skeleton_labels = np.array(list(g_sym.nodes()))
# temp_loc = None


# if temp_loc is None:
#     temp_loc = f"maggot_models/data/interim/temp-{np.random.randint(1e8)}.graphml"
#     # save to temp
#     nx.write_graphml(mg.g, temp_loc)

# # %% [markdown]
# # #

# import igraph as ig
# import leidenalg as la


# g = ig.Graph.Read_GraphML(temp_loc)
# nodes = [int(v["id"]) for v in g.vs]
# vert_part = la.find_partition(g, la.ModularityVertexPartition)
# labels = vert_part.membership
# partition = pd.Series(data=labels, index=nodes)


# %%


# %%
