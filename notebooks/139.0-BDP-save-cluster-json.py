# %% [markdown]
# ##
import os
import warnings
import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import poisson
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm
from umap import UMAP

from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import pairplot
from graspy.simulations import sbm
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplace,
)
from src.align import Procrustes
from src.cluster import BinaryCluster, MaggotCluster, get_paired_inds
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig, saveskels
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    plot_neurons,
    screeplot,
    set_axes_equal,
    stacked_barplot,
)

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


def stashskels(name, ids, labels, **kws):
    saveskels(name, ids, labels, foldername=FNAME, **kws)


# %% [markdown]
# ##


metric = "bic"
bic_ratio = 1
d = 10  # embedding dimension
method = "aniso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.0-BDP-omni-clust"


full_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
full_meta["lvl0_labels"] = full_meta["lvl0_labels"].astype(str)
# full_meta["sf"] = -full_meta["sf"]
full_adj = readcsv("adj" + basename, foldername=exp, index_col=0)
full_mg = MetaGraph(full_adj.values, full_meta)

full_meta = full_mg.meta

# %% [markdown]
# ##
for i in range(1, 8):
    labels = full_meta[f"lvl{i}_labels"].values
    uni_labels, inv_labels = np.unique(labels, return_inverse=True)
    cmap = dict(zip(np.unique(inv_labels), cc.glasbey))
    colors = np.vectorize(cmap.get)(inv_labels)
    stashskels(
        f"lvl{i}",
        ids=full_meta.index.values.astype(int),
        labels=labels,
        colors=colors,
        palette=None,
        multiout=True,
        postfix=f"-{exp}{basename}",
    )


# %%
