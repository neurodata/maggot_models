# %% [markdown]
# #
import os
import urllib.request
from operator import itemgetter
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
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.model_selection import ParameterGrid

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

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
BRAIN_VERSION = "2020-01-29"


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def read_outs(name):
    run_dir = Path(name)
    base_dir = Path("./maggot_models/notebooks/outs")
    block_file = base_dir / run_dir / "csvs" / "block-labels.csv"
    block_df = pd.read_csv(block_file, index_col=0)
    param_file = base_dir / run_dir / "csvs" / "parameters.csv"
    param_df = pd.read_csv(param_file, index_col=0)
    param_df.set_index("param_key", inplace=True)
    return block_df, param_df


# %% [markdown]
# # Load data

comm_df, comm_param_df = read_outs("81.0-BDP-community")

cell_df, cell_param_df = read_outs("77.0-BDP-try-graph-tool")

# %% [markdown]
# # Pick some pairs of cell type - community, look at confusion matrix

com_ind = 3
cell_ind = 10
comm_labels = comm_df.iloc[:, com_ind]
comm_name = comm_df.columns[com_ind]
cell_labels = cell_df.iloc[:, cell_ind]
cell_name = comm_df.columns[cell_ind]

shared_inds = np.intersect1d(comm_labels.index, cell_labels.index)
comm_labels = comm_labels[shared_inds]
cell_labels = cell_labels[shared_inds]
mask = np.logical_and(~np.isnan(comm_labels), ~np.isnan(cell_labels))
comm_labels = comm_labels[mask]
cell_labels = cell_labels[mask]
conf_mat = confusion_matrix(comm_labels, cell_labels)
row_sums = conf_mat.sum(axis=1)
conf_mat = conf_mat[row_sums != 0, :]
col_sums = conf_mat.sum(axis=0)
conf_mat = conf_mat[:, col_sums != 0]
fig, ax = plt.subplots(1, 1, figsize=(25, 25))
mask = conf_mat == 0
sns.heatmap(
    conf_mat,
    cmap="Reds",
    annot=True,
    ax=ax,
    cbar=False,
    square=True,
    xticklabels=[],
    yticklabels=[],
    mask=mask,
)
ax.set_title(f"{comm_name} vs. {cell_name}", fontsize=40)
ax.set_ylabel("Community", fontsize=30)
ax.set_xlabel("Cell type", fontsize=30)
stashfig(f"conf-mat-{comm_name}-{cell_name}")
