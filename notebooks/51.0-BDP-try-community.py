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
from src.data import load_everything, load_networkx
from src.embed import lse, preprocess_graph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
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
BRAIN_VERSION = "2020-01-14"

SAVEFIGS = True
SAVESKELS = False
SAVEOBJS = True

PTR = True
if PTR:
    ptr_type = "PTR"
else:
    ptr_type = "Raw"

ONLY_RIGHT = False
if ONLY_RIGHT:
    brain_type = "Right Hemisphere"
    brain_type_short = "righthemi"
else:
    brain_type = "Full Brain"
    brain_type_short = "fullbrain"

GRAPH_TYPE = "Gad"
if GRAPH_TYPE == "Gad":
    graph_type = r"A $\to$ D"

N_INIT = 200

CLUSTER_METHOD = "graspy-gmm"
if CLUSTER_METHOD == "graspy-gmm":
    cluster_type = "GraspyGMM"
elif CLUSTER_METHOD == "auto-gmm":
    cluster_type = "AutoGMM"

EMBED = "LSE"
if EMBED == "LSE":
    embed_type = "LSE"

N_COMPONENTS = None


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


# adj, class_labels, side_labels, pair_labels, skeleton_labels, = load_everything(
#     "Gadn",
#     version=BRAIN_VERSION,
#     return_keys=["Merge Class", "Hemisphere", "Pair"],
#     return_ids=True,
# )


g = load_networkx("Gn", version=BRAIN_VERSION)

g_sym = nx.to_undirected(g)
skeleton_labels = np.array(list(g_sym.nodes()))
scales = [1]
r = 1
out_dict = cm.best_partition(g_sym, resolution=r)
partition = np.array(itemgetter(*skeleton_labels.astype(str))(out_dict))
adj = nx.to_numpy_array(g_sym, nodelist=skeleton_labels)

part_unique, part_count = np.unique(partition, return_counts=True)
for uni, count in zip(part_unique, part_count):
    if count < 3:
        inds = np.where(partition == uni)[0]
    partition[inds] = -1

class_label_dict = nx.get_node_attributes(g_sym, "Class 1")
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
stashfig("louvain-barplot")

