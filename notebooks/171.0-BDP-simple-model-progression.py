# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import tplquad
from scipy.special import comb
from scipy.stats import gaussian_kde
from sklearn.metrics import pairwise_distances

import pymaid
from graspy.utils import pass_to_ranks
from hyppo.ksample import KSample
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    get_mid_map,
    gridmap,
    matrixplot,
    remove_axis,
    remove_spines,
    set_axes_equal,
    stacked_barplot,
)


# plotting settings
rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "lightgrey",
    "ytick.color": "grey",
    "xtick.color": "grey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)


np.random.seed(8888)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="png", dpi=200, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# load data
mg = load_metagraph("G")
mg = mg.reindex(mg.meta[~mg.meta["super"]].index, use_ids=True)

# %%
from src.visualization import remove_shared_ax

log_scale = True
distplot_kws = dict(kde=False)
ylabel = "Out degree (weighted)"
xlabel = "In degree (weighted)"
degrees = mg.calculate_degrees()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

median_degrees = np.median(degrees, axis=0)
median_in_degree = median_degrees[0]
median_out_degree = median_degrees[1]

if log_scale:
    degrees += 1
    ax.set_yscale("log")
    ax.set_xscale("log")
    # degrees = np.log10(degrees + 1)
    # ylabel = "Log10 " + ylabel
    # xlabel = "Log10 " + xlabel


sns.scatterplot(
    data=degrees, x="In edgesum", y="Out edgesum", s=5, alpha=0.2, linewidth=0, ax=ax
)
ax.scatter(
    median_in_degree, median_out_degree, s=200, marker="+", color="black", linewidth=2
)

ax.set_ylabel(ylabel)
ax.set_xlabel(xlabel)

divider = make_axes_locatable(ax)
top_ax = divider.append_axes("top", size="20%", sharex=ax)
sns.distplot(degrees["In edgesum"], ax=top_ax, **distplot_kws)
top_ax.xaxis.set_visible(False)
top_ax.yaxis.set_visible(False)

right_ax = divider.append_axes("right", size="20%", sharey=ax)
sns.distplot(degrees["Out edgesum"], ax=right_ax, vertical=True, **distplot_kws)
right_ax.yaxis.set_visible(False)
right_ax.xaxis.set_visible(False)
ax.axis("square")
stashfig(f"neuron-weighted-dergee-log_scale={log_scale}")


# %%
