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
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob
from src.visualization import (
    bartreeplot,
    draw_networkx_nice,
    get_color_dict,
    get_colors,
    palplot,
    probplot,
    sankey,
    screeplot,
    stacked_barplot,
    barplot_text,
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
)

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2020-01-29"

SAVEFIGS = True
SAVESKELS = False
SAVEOBJS = True

np.random.seed(23409857)
sns.set_context("talk")


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


def adjust_partition(partition):
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


mg = load_metagraph("G", BRAIN_VERSION)


# %% [markdown]
# #

graph_types = ["Gad"]  # ["Gadn", "Gad", "G"]
n_threshs = [0.01, 0.02]
r_threshs = [1]  # [1, 2, 3]
resolutons = [0.3]  # [0.3, 0.5, 0.7, 1]

for graph_type in graph_types:
    if graph_type[-1] == "n":
        threshs = n_threshs
    else:
        threshs = r_threshs
    for thresh in threshs:
        for r in resolutons:
            mg = load_metagraph(graph_type, version=BRAIN_VERSION)
            edgelist = mg.to_edgelist()
            edgelist = add_max_weight(edgelist)
            edgelist = edgelist[edgelist["max_weight"] > thresh]
            nodelist = list(mg.g.nodes())
            thresh_g = nx.from_pandas_edgelist(
                edgelist, edge_attr=True, create_using=nx.DiGraph
            )
            nx.set_node_attributes(thresh_g, mg.meta.to_dict(orient="index"))
            mg = MetaGraph(thresh_g)
            print(len(mg.meta))
            mg = mg.make_lcc()
            print(len(mg.meta))
            not_pdiff = np.where(~mg["is_pdiff"])[0]
            mg = mg.reindex(not_pdiff)
            print(len(mg.meta))
            g_sym = nx.to_undirected(mg.g)
            skeleton_labels = np.array(list(g_sym.nodes()))
            out_dict = cm.best_partition(g_sym, resolution=r)
            partition = np.array(itemgetter(*skeleton_labels)(out_dict))
            adj = nx.to_numpy_array(g_sym, nodelist=skeleton_labels)

            part_unique, part_count = np.unique(partition, return_counts=True)
            for uni, count in zip(part_unique, part_count):
                if count < 3:
                    inds = np.where(partition == uni)[0]
                    partition[inds] = -1

            mg.meta["signal_flow"] = signal_flow(mg.adj)
            mg.meta["partition"] = partition
            partition_sf = mg.meta.groupby("partition")["signal_flow"].median()
            sort_partition_sf = partition_sf.sort_values(ascending=False)

            basename = f"louvain-res{r}-t{thresh}-{graph_type}-"
            title = f"Louvain, {graph_type}, res = {r}, thresh = {thresh}"

            # barplot by merge class label (more detail)
            class_label_dict = nx.get_node_attributes(g_sym, "Merge Class")
            class_labels = np.array(itemgetter(*skeleton_labels)(class_label_dict))
            lineage_label_dict = nx.get_node_attributes(g_sym, "lineage")
            lineage_labels = np.array(itemgetter(*skeleton_labels)(lineage_label_dict))
            lineage_labels = np.vectorize(lambda x: "u" + x)(lineage_labels)
            fill_unk = True
            if fill_unk:
                fill_inds = np.where(class_labels == "unk")[0]
                class_labels[fill_inds] = lineage_labels[fill_inds]
                used_inds = np.array(list(CLASS_IND_DICT.values()))
                unused_inds = np.setdiff1d(range(len(cc.glasbey_light)), used_inds)
                lineage_color_dict = dict(
                    zip(
                        np.unique(lineage_labels),
                        np.array(cc.glasbey_light)[unused_inds],
                    )
                )
                color_dict = {**CLASS_COLOR_DICT, **lineage_color_dict}
                hatch_dict = {}
                for key, val in color_dict.items():
                    if key[0] == "u":
                        hatch_dict[key] = "//"
                    else:
                        hatch_dict[key] = ""
            else:
                color_dict = "class"

            fig, axs = barplot_text(
                partition,
                class_labels,
                color_dict=color_dict,
                plot_proportions=False,
                norm_bar_width=True,
                figsize=(24, 18),
                title=title,
                hatch_dict=hatch_dict,
            )
            stashfig(basename + "barplot-mergeclass")

            fig, axs = barplot_text(
                partition,
                lineage_labels,
                color_dict=None,
                plot_proportions=False,
                norm_bar_width=True,
                figsize=(24, 18),
                title=title,
            )
            stashfig(basename + "barplot-lineage")

            # sorted heatmap
            heatmap(
                mg.adj,
                transform="simple-nonzero",
                figsize=(20, 20),
                inner_hier_labels=partition,
                hier_label_fontsize=10,
                title=title,
                title_pad=80,
            )
            stashfig(basename + "heatmap")

            # block probabilities
            counts = False
            weights = False
            prob_df = get_blockmodel_df(
                mg.adj, partition, return_counts=counts, use_weights=weights
            )
            prob_df = prob_df.reindex(sort_partition_sf.index, axis=0)
            prob_df = prob_df.reindex(sort_partition_sf.index, axis=1)

            ax = probplot(
                100 * prob_df,
                fmt="2.0f",
                figsize=(20, 20),
                title=f"Louvain, res = {r}, counts = {counts}, weights = {weights}",
            )
            ax.set_ylabel(r"Median signal flow $\to$", fontsize=28)

            stashfig(basename + f"probplot-counts{counts}-weights{weights}")

            adjusted_partition = adjust_partition(partition)
            minigraph = to_minigraph(
                mg.adj, adjusted_partition, use_counts=True, size_scaler=10
            )
            draw_networkx_nice(
                minigraph,
                "Spring-x",
                "Signal Flow",
                sizes="Size",
                colors="Color",
                cmap="Greys",
                vmin=100,
                weight_scale=0.001,
            )
            stashfig(basename + "sbm-drawn-network")

