# %% [markdown]
# #
import os

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import urllib.request
from graph_tool import load_graph
from graph_tool.inference import minimize_blockmodel_dl
from joblib import Parallel, delayed
from random_word import RandomWords
from sklearn.model_selection import ParameterGrid

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
from pathlib import Path

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)
    plt.close()


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


run_dir = Path("77.0-BDP-try-graph-tool")

base_dir = Path("./maggot_models/notebooks/outs")

block_file = base_dir / run_dir / "csvs" / "block-labels.csv"

block_df = pd.read_csv(block_file)

n_runs = len(block_df.columns) - 1

from graspy.utils import cartprod
from sklearn.metrics import adjusted_rand_score

block_pairs = cartprod(range(n_runs), range(n_runs))

ari_mat = np.empty((n_runs, n_runs))
for bp in block_pairs:
    from_block_labels = block_df.iloc[:, bp[0] + 1].values
    to_block_labels = block_df.iloc[:, bp[1] + 1].values
    mask = np.logical_and(~np.isnan(from_block_labels), ~np.isnan(to_block_labels))
    from_block_labels = from_block_labels[mask]
    to_block_labels = to_block_labels[mask]
    ari = adjusted_rand_score(from_block_labels, to_block_labels)
    ari_mat[bp[0], bp[1]] = ari

# %% [markdown]
# #
run_names = block_df.columns[1:].values
ari_df = pd.DataFrame(data=ari_mat, index=run_names, columns=run_names)

import seaborn as sns

sns.set_context("talk", font_scale=0.8)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    data=ari_df,
    cmap="Reds",
    annot=True,
    square=True,
    ax=ax,
    cbar_kws=dict(shrink=0.7),
    vmin=0,
)
