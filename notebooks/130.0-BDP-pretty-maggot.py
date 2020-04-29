# %% [markdown]
# # THE MIND OF A MAGGOT

# %% [markdown]
# ## Imports
import os
import time
import warnings

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, NodeMixin
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.testing import ignore_warnings
from tqdm import tqdm

import pymaid
from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, selectSVD
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import rdpg
from graspy.utils import augment_diagonal, binarize, pass_to_ranks
from src.cluster import (
    MaggotCluster,
    add_connections,
    compute_pairedness_bipartite,
    crossval_cluster,
    fit_and_score,
    get_paired_inds,
    make_ellipses,
    plot_cluster_pairs,
    plot_metrics,
    predict,
)
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
    set_axes_equal,
    stacked_barplot,
    plot_neurons,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name)


mg = load_metagraph("G", version="2020-04-01")

meta = mg.meta

n_verts = len(meta)

uni_class = np.unique(meta["merge_class"])

subsample_ratio = 0.3

sub_metas = []
for uc in uni_class:
    class_meta = meta[meta["merge_class"] == uc]
    n_class = len(class_meta)
    n_show = int(np.ceil(n_class * subsample_ratio / 2))
    print(uc)
    print(n_show)
    print()
    uni_pairs = np.unique(class_meta["Pair ID"])
    if len(uni_pairs) > 1:
        uni_pairs = uni_pairs[1:]

    n_show = min(n_show, len(uni_pairs))
    pairs = np.random.choice(uni_pairs, size=n_show, replace=False)

    show_meta = class_meta[class_meta["Pair ID"].isin(pairs)]
    sub_metas.append(show_meta.copy())


start_instance()
plot_meta = pd.concat(sub_metas, axis=0, ignore_index=False)

print()
print("Total neurons being plotted:")
print(len(plot_meta))
print()

plot_neurons(plot_meta)
stashfig("pretty-maggot")
