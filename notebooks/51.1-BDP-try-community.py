# %% [markdown]
# # Imports
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
from matplotlib.cm import ScalarMappable

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph, load_networkx
from src.embed import lse, preprocess_graph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    probplot,
    sankey,
    screeplot,
    stacked_barplot,
)

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2020-01-21"

SAVEFIGS = True
SAVESKELS = False
SAVEOBJS = True

np.random.seed(23409857)


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


def stashobj(obj, name, **kws):
    saveobj(obj, name, foldername=FNAME, save_on=SAVEOBJS, **kws)


graph_type = "G"
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
print(len(mg.meta))
mg = mg.make_lcc()
print(len(mg.meta))
not_pdiff = np.where(~mg["is_pdiff"])[0]
mg = mg.reindex(not_pdiff)
print(len(mg.meta))
g_sym = nx.to_undirected(mg.g)


# %% [markdown]
# #
skeleton_labels = np.array(list(g_sym.nodes()))
scales = [1]
r = 0.3
out_dict = cm.best_partition(g_sym, resolution=r)
partition = np.array(itemgetter(*skeleton_labels)(out_dict))
adj = nx.to_numpy_array(g_sym, nodelist=skeleton_labels)

part_unique, part_count = np.unique(partition, return_counts=True)
for uni, count in zip(part_unique, part_count):
    if count < 3:
        inds = np.where(partition == uni)[0]
        partition[inds] = -1

basename = f"louvain-res{r}-{graph_type}-"

# barplot by class label
class_label_dict = nx.get_node_attributes(g_sym, "Class 1")
class_labels = np.array(itemgetter(*skeleton_labels)(class_label_dict))
part_color_dict = dict(zip(np.unique(partition), cc.glasbey_warm))
true_color_dict = dict(zip(np.unique(class_labels), cc.glasbey_light))
color_dict = {**part_color_dict, **true_color_dict}
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
stacked_barplot(
    partition,
    class_labels,
    ax=ax,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
)
stashfig(basename + "barplot-class1")

# barplot by merge class label (more detail)
class_label_dict = nx.get_node_attributes(g_sym, "Merge Class")
class_labels = np.array(itemgetter(*skeleton_labels)(class_label_dict))
part_color_dict = dict(zip(np.unique(partition), cc.glasbey_warm))
true_color_dict = dict(zip(np.unique(class_labels), cc.glasbey_light))
color_dict = {**part_color_dict, **true_color_dict}
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
stacked_barplot(
    partition,
    class_labels,
    ax=ax,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
)
stashfig(basename + "barplot-mergeclass")

# sorted heatmap
heatmap(
    mg.adj,
    transform="simple-nonzero",
    figsize=(20, 20),
    inner_hier_labels=partition,
    hier_label_fontsize=10,
)
stashfig(basename + "heatmap")

# block probabilities
counts = False
weights = False
prob_df = get_blockmodel_df(
    mg.adj, partition, return_counts=counts, use_weights=weights
)
probplot(
    100 * prob_df,
    fmt="2.0f",
    figsize=(20, 20),
    title=f"Louvain, res = {r}, counts = {counts}, weights = {weights}",
)
stashfig(basename + f"probplot-counts{counts}-weights{weights}")

weights = True
prob_df = get_blockmodel_df(
    mg.adj, partition, return_counts=counts, use_weights=weights
)

probplot(
    100 * prob_df,
    fmt="2.0f",
    figsize=(20, 20),
    title=f"Louvain, res = {r}, counts = {counts}, weights = {weights}",
)
stashfig(basename + f"probplot-counts{counts}-weights{weights}")

# %% [markdown]
# # Play with colormapping
from src.visualization import palplot

fig, axs = plt.subplots(1, 4, figsize=(5, 10))
n_per_col = 40
for i, ax in enumerate(axs):
    pal = cc.glasbey_light[i * n_per_col : (i + 1) * n_per_col]
    palplot(n_per_col, pal, figsize=(1, 10), ax=ax, start=i * n_per_col)
stashfig("glasbey-colors")

# %% [markdown]
# #
print(np.unique(class_labels))

manual_cmap = {
    "KC": 0,
    "KC-1claw": 28,
    "KC-2claw": 32,
    "KC-3claw": 92,
    "KC-4claw": 91,
    "KC-5claw": 78,
    "KC-6claw": 61,
    "APL": 24,
    "MBIN": 121,
    "MBIN-DAN": 58,
    "MBIN-OAN": 5,
    "MBON": 11,
    "sens-AN": 1,
    "sens-MN": 12,
    "sens-ORN": 51,
    "sens-PaN": 76,
    "sens-photoRh5": 84,
    "sens-photoRh6": 106,
    "sens-thermo;AN": 110,
    "sens-vtd": 145,
    "mPN-multi": 3,
    "mPN-olfac": 16,
    "mPN;FFN-multi": 3,
    "tPN": 33,
    "uPN": 36,
    "vPN": 62,
    "pLN": 30,
    "bLN-Duet": 159,
    "bLN-Trio": 81,
    "cLN": 99,
    "FAN": 2,
    "FB2N": 21,
    "FBN": 50,
    "FFN": 52,
    "O_IPC": 42,
    "O_ITP": 7,
    "O_ITP;O_dSEZ": 7,
    "O_dSEZ": 26,
    "O_dSEZ;FB2N": 21,
    "O_dSEZ;FFN": 52,
    "O_dSEZ;O_CA-LP": 124,
    "O_dVNC": 113,
    "Unk": 44,
}

names = []
color_inds = []

for key, val in manual_cmap.items():
    names.append(key)
    color_inds.append(val)

fig, ax = plt.subplots(1, 1, figsize=(3, 15))
colors = np.array(cc.glasbey_light)[color_inds]
palplot(len(colors), colors, ax=ax)
ax.yaxis.set_major_formatter(plt.FixedFormatter(names))
stashfig("named-cmap")
# %% [markdown]
# #

class_label_dict = nx.get_node_attributes(g_sym, "Merge Class")
class_labels = np.array(itemgetter(*skeleton_labels)(class_label_dict))
part_color_dict = dict(zip(np.unique(partition), cc.glasbey_warm))
true_color_dict = dict(zip(names, colors))
color_dict = {**part_color_dict, **true_color_dict}
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
stacked_barplot(
    partition,
    class_labels,
    ax=ax,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
)
stashfig(basename + "barplot-mergeclass")

