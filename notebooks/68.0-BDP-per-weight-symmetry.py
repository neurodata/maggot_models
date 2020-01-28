# %% [markdown]
# # Imports
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.embed import file_html
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, FactorRange, Legend, Span, PreText, Circle
from bokeh.palettes import Spectral4, all_palettes
from bokeh.plotting import curdoc, figure, output_file, show
from bokeh.resources import CDN
from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph_shortest_path import graph_shortest_path
from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.utils import pass_to_ranks, get_lcc
from graspy.plot import degreeplot, edgeplot, gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph, load_networkx
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import (
    get_blockmodel_df,
    get_sbm_prob,
    invert_permutation,
    meta_to_array,
    savefig,
)
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

from bokeh.models import Select
from bokeh.palettes import Spectral5
from bokeh.plotting import curdoc, figure
from scipy.linalg import orthogonal_procrustes

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


# %% [markdown]
# #
graph_type = "Gad"
mg = load_metagraph(graph_type, BRAIN_VERSION)
# only consider the edges for which we have a mirror edges
edgelist = mg.to_edgelist(remove_unpaired=True)

max_edge = edgelist["weight"].max()

rows = []
for edgeweight in range(1, max_edge + 1):
    temp_edgelist = edgelist[edgelist["weight"] == edgeweight]
    n_edges = len(temp_edgelist)
    n_edges_mirrored = (temp_edgelist["edge pair counts"] == 2).sum()
    row = {
        "weight": edgeweight,
        "n_edges": n_edges,
        "n_edges_mirrored": n_edges_mirrored,
        "p_edge_mirrored": n_edges_mirrored / n_edges,
        "n_edges_unmirrored": n_edges - n_edges_mirrored,
    }
    rows.append(row)

result_df = pd.DataFrame(rows)
result_df = result_df[result_df["n_edges"] != 0]

# %% [markdown]
# #
fig, axs = plt.subplots(2, 1, figsize=(10, 15), sharex=True)
sns.scatterplot(x="weight", y="p_edge_mirrored", data=result_df, ax=axs[0])
sns.scatterplot(x="weight", y="n_edges", data=result_df, ax=axs[1])
sns.scatterplot(x="weight", y="n_edges_unmirrored", data=result_df, ax=axs[1])


fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
sns.scatterplot(x="weight", y="p_edge_mirrored", data=result_df, ax=axs[0])
axs[0].set_yscale("log")
sns.scatterplot(x="weight", y="n_edges", data=result_df, ax=axs[1])
axs[1].set_yscale("log")
sns.scatterplot(x="weight", y="n_edges_unmirrored", data=result_df, ax=axs[2])
axs[2].set_yscale("log")

