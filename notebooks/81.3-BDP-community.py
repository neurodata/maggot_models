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


def to_minigraph(
    adj,
    labels,
    drop_neg=True,
    remove_diag=True,
    size_scaler=1,
    use_counts=False,
    use_weights=True,
    color_map=None,
):
    # convert the adjacency and a partition to a minigraph based on SBM probs
    prob_df = get_blockmodel_df(
        adj, labels, return_counts=use_counts, use_weights=use_weights
    )
    if drop_neg and ("-1" in prob_df.index):
        prob_df.drop("-1", axis=0, inplace=True)
        prob_df.drop("-1", axis=1, inplace=True)

    if remove_diag:
        adj = prob_df.values
        adj -= np.diag(np.diag(adj))
        prob_df.data = prob_df

    g = nx.from_pandas_adjacency(prob_df, create_using=nx.DiGraph())
    uni_labels, counts = np.unique(labels, return_counts=True)

    # add size attribute base on number of vertices
    size_map = dict(zip(uni_labels, size_scaler * counts))
    nx.set_node_attributes(g, size_map, name="Size")

    # add signal flow attribute (for the minigraph itself)
    mini_adj = nx.to_numpy_array(g, nodelist=uni_labels)
    node_signal_flow = signal_flow(mini_adj)
    sf_map = dict(zip(uni_labels, node_signal_flow))
    nx.set_node_attributes(g, sf_map, name="Signal Flow")

    # add spectral properties
    sym_adj = symmetrize(mini_adj)
    n_components = 10
    latent = AdjacencySpectralEmbed(n_components=n_components).fit_transform(sym_adj)
    for i in range(n_components):
        latent_dim = latent[:, i]
        lap_map = dict(zip(uni_labels, latent_dim))
        nx.set_node_attributes(g, lap_map, name=f"AdjEvec-{i}")

    # add spring layout properties
    pos = nx.spring_layout(g)
    spring_x = {}
    spring_y = {}
    for key, val in pos.items():
        spring_x[key] = val[0]
        spring_y[key] = val[1]
    nx.set_node_attributes(g, spring_x, name="Spring-x")
    nx.set_node_attributes(g, spring_y, name="Spring-y")

    # add colors
    if color_map is None:
        color_map = dict(zip(uni_labels, cc.glasbey_light))
    nx.set_node_attributes(g, color_map, name="Color")
    return g


def adjust_partition(partition, class_labels):
    adjusted_partition = partition.copy().astype(str)
    sens_classes = [
        "sens-AN",
        "sens-MN",
        "sens-ORN",
        "sens-PaN",
        "sens-photoRh5",
        "sens-photoRh6",
        "sens-thermo;AN",
        "sens-vtd",
    ]
    for s in sens_classes:
        inds = np.where(class_labels == s)[0]
        adjusted_partition[inds] = s
    return adjusted_partition


def add_max_weight(df):
    max_pair_edges = df.groupby("edge pair ID", sort=False)["weight"].max()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    df["max_weight"] = itemgetter(*df["edge pair ID"])(edge_max_weight_map)
    asym_inds = df[df["edge pair ID"] == -1].index
    df.loc[asym_inds, "max_weight"] = df.loc[asym_inds, "weight"]
    return df


def edgelist_to_mg(edgelist, meta):
    g = nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=nx.DiGraph)
    nx.set_node_attributes(g, meta.to_dict(orient="index"))
    mg = MetaGraph(g)
    return mg


def run_louvain(g_sym, res, skeleton_labels):
    out_dict = cm.best_partition(g_sym, resolution=res)
    modularity = cm.modularity(out_dict, g_sym)
    partition = np.array(itemgetter(*skeleton_labels)(out_dict))
    part_unique, part_count = np.unique(partition, return_counts=True)
    for uni, count in zip(part_unique, part_count):
        if count < 3:
            inds = np.where(partition == uni)[0]
            partition[inds] = -1
    return partition, modularity


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


def run_experiment(
    graph_type=None, threshold=None, res=None, binarize=None, seed=None, param_key=None
):
    # common names
    if BLIND:
        basename = f"{param_key}-"
        title = param_key
    else:
        basename = f"louvain-res{res}-t{threshold}-{graph_type}-"
        title = f"Louvain, {graph_type}, res = {res}, threshold = {threshold}"

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
    g_sym = nx.to_undirected(mg.g)
    skeleton_labels = np.array(list(g_sym.nodes()))
    partition, modularity = run_louvain(g_sym, res, skeleton_labels)

    partition_series = pd.Series(partition, index=skeleton_labels)
    partition_series.name = param_key

    if SAVEFIGS:
        # get out some metadata
        class_label_dict = nx.get_node_attributes(g_sym, "Merge Class")
        class_labels = np.array(itemgetter(*skeleton_labels)(class_label_dict))
        lineage_label_dict = nx.get_node_attributes(g_sym, "lineage")
        lineage_labels = np.array(itemgetter(*skeleton_labels)(lineage_label_dict))
        lineage_labels = np.vectorize(lambda x: "~" + x)(lineage_labels)
        classlin_labels, color_dict, hatch_dict = augment_classes(
            class_labels, lineage_labels
        )

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

        # TODO add gridmap

        counts = False
        weights = False
        prob_df = get_blockmodel_df(
            mg.adj, partition, return_counts=counts, use_weights=weights
        )
        prob_df = prob_df.reindex(category_order, axis=0)
        prob_df = prob_df.reindex(category_order, axis=1)
        probplot(
            100 * prob_df, fmt="2.0f", figsize=(20, 20), title=title, font_scale=0.7
        )
        stashfig(basename + f"probplot-counts{counts}-weights{weights}")

    return partition_series, modularity


# %% [markdown]
# #
np.random.seed(888888)
n_replicates = 20
param_grid = {
    "graph_type": ["G"],
    "threshold": [0, 1, 2, 3],
    "res": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "binarize": [True, False],
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

