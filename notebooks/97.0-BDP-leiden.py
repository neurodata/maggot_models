#%%
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
from src.visualization import random_names
from src.block import run_leiden

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


def run_experiment(
    graph_type=None, threshold=None, res=None, binarize=None, seed=None, param_key=None
):
    np.random.seed(seed)

    # load and preprocess the data
    mg = load_metagraph(graph_type, version=BRAIN_VERSION)
    mg = preprocess(
        mg,
        threshold=threshold,
        sym_threshold=True,
        remove_pdiff=True,
        binarize=binarize,
    )
    adj = mg.adj
    adj = symmetrize(adj, method="avg")
    mg = MetaGraph(adj, mg.meta)
    g_sym = mg.g
    skeleton_labels = np.array(list(g_sym.nodes()))
    partition, modularity = run_leiden(g_sym, res, skeleton_labels)

    partition_series = pd.Series(partition, index=skeleton_labels)
    partition_series.name = param_key

    return partition_series, modularity


# %% [markdown]
# #
np.random.seed(889888)
n_replicates = 5
param_grid = {
    "graph_type": ["G"],
    "threshold": [0, 1, 2, 3],
    "res": np.geomspace(0.0005, 0.05, 10),
    "binarize": [True, False],
    "objective_finction": ["CPM", "modularity"],
    "n_iterations": [5],
}
params = list(ParameterGrid(param_grid))
seeds = np.random.randint(1e8, size=n_replicates * len(params))
param_keys = random_names(len(seeds))

rep_params = []
for i, seed in enumerate(seeds):
    p = params[i % len(params)].copy()
    p["seed"] = seed
    p["param_key"] = param_keys[i]
    rep_params.append(p)

# %% [markdown]
# #
print("\n\n\n\n")
print(f"Running {len(rep_params)} jobs in total")
print("\n\n\n\n")
outs = Parallel(n_jobs=-2, verbose=10)(delayed(run_experiment)(**p) for p in rep_params)
partitions, modularities = list(zip(*outs))
# %% [markdown]
# #
block_df = pd.concat(partitions, axis=1, ignore_index=False)
stashcsv(block_df, "block-labels")
param_df = pd.DataFrame(rep_params)
param_df["modularity"] = modularities
stashcsv(param_df, "parameters")

