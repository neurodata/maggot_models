# %% [markdown]
# #

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
from src.visualization import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    barplot_text,
    bartreeplot,
    draw_networkx_nice,
    get_color_dict,
    get_colors,
    palplot,
    probplot,
    sankey,
    screeplot,
    stacked_barplot,
    random_names,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2020-03-02"
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


graph_type = "G"
threshold = 3
binarize = False

# load and preprocess the data
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
mg = preprocess(
    mg, threshold=threshold, sym_threshold=True, remove_pdiff=True, binarize=binarize
)

#%%

import leidenalg as la
import igraph as ig


def _process_metagraph(mg, temp_loc):
    adj = mg.adj
    adj = symmetrize(adj, method="avg")
    mg = MetaGraph(adj, mg.meta)
    nx.write_graphml(mg.g, temp_loc)


def run_leiden(
    mg,
    temp_loc=None,
    implementation="igraph",
    partition_type=la.CPMVertexPartition,
    **kws,
):
    if temp_loc is None:
        temp_loc = f"maggot_models/data/interim/temp-{np.random.randint(1e8)}.graphml"
    else:
        temp_loc = f"maggot_models/data/interim/{temp_loc}.graphml"
    _process_metagraph(mg, temp_loc)
    g = ig.Graph.Read_GraphML(temp_loc)
    os.remove(temp_loc)
    nodes = [int(v["id"]) for v in g.vs]
    if implementation == "igraph":
        vert_part = g.community_leiden(**kws)
    elif implementation == "leidenalg":
        vert_part = la.find_partition(g, partition_type, **kws)
    labels = vert_part.membership
    partition = pd.Series(data=labels, index=nodes)
    return partition, vert_part.modularity


partition, modularity = run_leiden(
    mg,
    implementation="leidenalg",
    resolution_parameter=0.001,
    partition_type=la.CPMVertexPartition,
    weights="weight",
    n_iterations=-1,
)
print(partition.nunique())

# %% [markdown]
# #


pred_labels = partition
pred_labels = pred_labels[pred_labels.index.isin(mg.meta.index)]
partition = pred_labels.astype(int)

class_labels = mg["Merge Class"]
lineage_labels = mg["lineage"]
basename = ""
title = ""


def augment_classes(class_labels, lineage_labels, fill_unk=True):
    if fill_unk:
        classlin_labels = class_labels.copy()
        fill_inds = np.where(class_labels == "unk")[0]
        classlin_labels[fill_inds] = lineage_labels[fill_inds]
        used_inds = np.array(list(CLASS_IND_DICT.values()))
        unused_inds = np.setdiff1d(range(len(cc.glasbey_light)), used_inds)
        lineage_color_dict = dict(
            zip(np.unique(lineage_labels), np.array(cc.glasbey_light)[unused_inds])
        )
        color_dict = {**CLASS_COLOR_DICT, **lineage_color_dict}
        hatch_dict = {}
        for key, val in color_dict.items():
            if key[0] == "~":
                hatch_dict[key] = "//"
            else:
                hatch_dict[key] = ""
    else:
        color_dict = "class"
        hatch_dict = None
    return classlin_labels, color_dict, hatch_dict


lineage_labels = np.vectorize(lambda x: "~" + x)(lineage_labels)
classlin_labels, color_dict, hatch_dict = augment_classes(class_labels, lineage_labels)

# TODO then sort all of them by proportion of sensory/motor
# barplot by merge class and lineage
_, _, order = barplot_text(
    partition,
    classlin_labels,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
    figsize=(24, 18),
    title=title,
    hatch_dict=hatch_dict,
    return_order=True,
)
stashfig(basename + "barplot-mergeclasslin-props")
plt.close()
category_order = np.unique(partition)[order]

fig, axs = barplot_text(
    partition,
    class_labels,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
    figsize=(24, 18),
    title=title,
    hatch_dict=None,
    category_order=category_order,
)
stashfig(basename + "barplot-mergeclass-props")

fig, axs = barplot_text(
    partition,
    class_labels,
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=False,
    figsize=(24, 18),
    title=title,
    hatch_dict=None,
    category_order=category_order,
)
stashfig(basename + "barplot-mergeclass-counts")
plt.close()

# TODO add gridmap

counts = False
weights = False
prob_df = get_blockmodel_df(
    mg.adj, partition, return_counts=counts, use_weights=weights
)
prob_df = prob_df.reindex(category_order, axis=0)
prob_df = prob_df.reindex(category_order, axis=1)
probplot(100 * prob_df, fmt="2.0f", figsize=(20, 20), title=title, font_scale=0.7)
stashfig(basename + f"probplot-counts{counts}-weights{weights}")
plt.close()

