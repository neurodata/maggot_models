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
    set_theme,
)


# plotting settings
rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    # "axes.edgecolor": "lightgrey",
    # "ytick.color": "grey",
    # "xtick.color": "grey",
    # "axes.labelcolor": "dimgrey",
    # "text.color": "dimgrey",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}

set_theme(rc_dict=rc_dict, font_scale=1.25)


np.random.seed(8888)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, format="png", dpi=200, **kws)
    savefig(name, foldername=FNAME, save_on=True, format="pdf", dpi=200, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# load data
mg = load_metagraph("G")
# mg = mg.reindex(mg.meta[~mg.meta["super"]].index, use_ids=True)


graph_types = ["Gad", "Gaa", "Gdd", "Gda"]  # "Gs"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g)
    # this line is important, to make the graphs aligned
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)


# %%

fig, ax = plt.subplots(2, 2, figsize=(20, 20))


# %% [markdown]
# ##

# %% [markdown]
# ## Load the 4-color graphs

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g)
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

# %% [markdown]
# ## Combine them into the 2N graph...
n_verts = len(adjs[0])
axon_inds = np.arange(n_verts)
dend_inds = axon_inds.copy() + n_verts
double_adj = np.empty((2 * n_verts, 2 * n_verts))
double_adj[np.ix_(axon_inds, axon_inds)] = adjs[1]  # Gaa
double_adj[np.ix_(axon_inds, dend_inds)] = adjs[0]  # Gad
double_adj[np.ix_(dend_inds, axon_inds)] = adjs[3]  # Gda
double_adj[np.ix_(dend_inds, dend_inds)] = adjs[2]  # Gdd
# double_adj[axon_inds, dend_inds] = 1000  # make internal edges, make em big
# double_adj[dend_inds, axon_inds] = 1000

axon_meta = mg.meta.rename(index=lambda x: str(x) + "_axon")
axon_meta["compartment"] = "Axon"
dend_meta = mg.meta.rename(index=lambda x: str(x) + "_dend")
dend_meta["compartment"] = "Dendrite"


double_meta = pd.concat((axon_meta, dend_meta), axis=0)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    double_adj,
    plot_type="scattermap",
    sizes=(1, 1),
    ax=ax,
    meta=double_meta,
    sort_class=["compartment"],
    item_order=["merge_class", "pair_id"],
    colors=["merge_class"],
    palette=CLASS_COLOR_DICT,
)
stashfig("double-adj")


#%%
fig, axs = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw=dict(hspace=0, wspace=0))
matrixplot_kws = dict(
    row_meta=mg.meta,
    col_meta=mg.meta,
    row_item_order=[
        "merge_class",
        "pair_id",
    ],  # TODO maybe pick whatever we do in next figure
    col_item_order=["merge_class", "pair_id"],
    # colors=["merge_class"],
    palette=CLASS_COLOR_DICT,
    sizes=(1, 1),
    plot_type="scattermap",
)

edge_type_palette = dict(zip(graph_types, sns.color_palette("deep")))

ax = axs[0, 0]
matrixplot(adjs[1], ax=ax, color=edge_type_palette["Gaa"], **matrixplot_kws)
ax.set(ylabel="Axon", title="Axon")

ax = axs[0, 1]
matrixplot(adjs[0], ax=ax, color=edge_type_palette["Gad"], **matrixplot_kws)
ax.set(title="Dendrite")

ax = axs[1, 0]
matrixplot(adjs[3], ax=ax, color=edge_type_palette["Gda"], **matrixplot_kws)
ax.set(ylabel="Dendrite")

ax = axs[1, 1]
matrixplot(adjs[2], ax=ax, color=edge_type_palette["Gdd"], **matrixplot_kws)
stashfig("4-color-adjplot")