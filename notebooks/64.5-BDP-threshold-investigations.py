# %% [markdown]
# # Imports
import os
from operator import itemgetter
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.io import savefig, savecsv
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True
BRAIN_VERSION = "2020-04-23"

sns.set_context("talk")

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def threshold_synapses(edgelist_df, start=0, stop=21, min_weight=1):
    # each edge here is paired left right
    # for each edge, there is an attribute "max_syn_weight" which is the max weight
    # from both edges in the pair. if an edge doesnt have a pair on the other side then
    # this is just that edge's weight.

    threshs = np.arange(0, stop)
    rows = []
    for threshold in threshs:
        # get rid of edges less than or equal to the current threshold, but do the
        # thresholding based on the max of the two sides
        thresh_df = edgelist_df[edgelist_df["max_syn_weight"] > threshold].copy()

        # number of unique remaining edges, so a left/right pair is counted once,
        # a singleton edge is also counted once.
        n_left = thresh_df["edge pair ID"].nunique()

        if n_left == 0:  # this threshold removed all synapses
            print(f"Threshold = {threshold} removed all synapses")
            p_sym = np.nan
            p_edges_left = 0
            p_syns_left = 0
        else:
            # NOTE this whole chunk is just to do the min weight thing Albert wanted
            # if min_weight is 1 then this does nothing
            # get the edges which oritinally had a pair in the edgelist
            # edge pair counts is either 1 or 2, how many edges there are in that "pair"
            # since we thresholded based on "max_syn_weight" if there was a pair it will
            # still be here
            # now, get rid of all of the pairs for which one of the edges is < 2
            mirrored_df = thresh_df[thresh_df["edge pair counts"] == 2].copy()
            min_pair_edges = mirrored_df.groupby("edge pair ID", sort=False)[
                "syn_weight"
            ].min()
            edge_min_weight_map = dict(
                zip(min_pair_edges.index.values, min_pair_edges.values)
            )
            mirrored_df["min_syn_weight"] = mirrored_df["edge pair ID"].map(
                edge_min_weight_map
            )
            # threshold again based on minimum synapse weight
            mirrored_df = mirrored_df[mirrored_df["min_syn_weight"] >= min_weight]

            n_sym = mirrored_df["edge pair ID"].nunique()
            p_sym = n_sym / n_left
            p_edges_left = len(thresh_df) / len(edgelist_df)
            p_syns_left = (
                thresh_df["syn_weight"].sum() / edgelist_df["syn_weight"].sum()
            )
        row = {
            "threshold": threshold,
            "Prop. paired edges symmetric": p_sym,
            "Prop. edges left": p_edges_left,
            "Prop. synapses left": p_syns_left,
        }
        rows.append(row)
    return pd.DataFrame(rows)


data = np.array(
    [[3, 1, 2, 3], [3, 1, 2, 2], [4, 2, 2, 3], [4, 2, 2, 4], [4, 3, 1, 4], [5, 4, 1, 5]]
)
fake_edgelist = pd.DataFrame(
    columns=["max_syn_weight", "edge pair ID", "edge pair counts", "syn_weight"],
    data=data,
)
print(fake_edgelist)
out_df = threshold_synapses(fake_edgelist, 0, 7)
print(out_df)
# %% [markdown]
# ##
remove_pdiff = True
input_thresh = 0  # this was whether to remove low input neurons, only for normalized
use_subset = False  # this was whether to look at only some published neurons

graph_types = ["G", "Gad", "Gaa", "Gdd", "Gda"]
for graph_type in graph_types:
    print(graph_type)
    mg = load_metagraph(graph_type, version=BRAIN_VERSION)

    if use_subset:
        meta = mg.meta
        meta["Original index"] = range(len(meta))
        subset_classes = [
            # "KC",
            "MBIN",
            "mPN",
            "tPN",
            "APL",
            "MBON",
            "uPN",
            "vPN",
            "mPN; FFN",
            "bLN",
            "cLN",
            "pLN",
            "sens",
        ]
        subset_inds = meta[meta["Class 1"].isin(subset_classes)]["Original index"]
        mg.reindex(subset_inds)

    print(f"{len(mg.to_edgelist())} original edges")

    def preprocess(mg, remove_pdiff=False):
        n_original_verts = len(mg)

        if remove_pdiff:
            keep_inds = np.where(~mg["partially_differentiated"])[0]
            mg = mg.reindex(keep_inds)
            print(f"Removed {n_original_verts - len(mg.meta)} partially differentiated")

        mg.verify(n_checks=100000, version=BRAIN_VERSION, graph_type=graph_type)

        edgelist_df = mg.to_edgelist(remove_unpaired=True)
        edgelist_df.rename(columns={"weight": "syn_weight"}, inplace=True)
        edgelist_df["norm_weight"] = (
            edgelist_df["syn_weight"] / edgelist_df["target dendrite_input"]
        )

        max_pair_edges = edgelist_df.groupby("edge pair ID", sort=False)[
            "syn_weight"
        ].max()
        edge_max_weight_map = dict(
            zip(max_pair_edges.index.values, max_pair_edges.values)
        )
        edgelist_df["max_syn_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
            edge_max_weight_map
        )
        max_pair_edges = edgelist_df.groupby("edge pair ID", sort=False)[
            "norm_weight"
        ].max()
        edge_max_weight_map = dict(
            zip(max_pair_edges.index.values, max_pair_edges.values)
        )
        edgelist_df["max_norm_weight"] = itemgetter(*edgelist_df["edge pair ID"])(
            edge_max_weight_map
        )

        return edgelist_df

    edgelist_df = preprocess(mg, remove_pdiff=remove_pdiff)
    print(f"{edgelist_df['syn_weight'].max()} max edgeweight")
    n_paired_edges = len(edgelist_df)

    meta = mg.meta
    remove_inds = meta[meta["dendrite_input"] < input_thresh].index
    edgelist_df = edgelist_df[~edgelist_df["target"].isin(remove_inds)]
    n_left_edges = len(edgelist_df)

    print(f"{n_left_edges} edges after preprocessing")

    thresh_result_df = threshold_synapses(edgelist_df)
    thresh_result_df_min2 = threshold_synapses(edgelist_df, min_weight=2)

    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.scatterplot(
        data=thresh_result_df,
        x="threshold",
        y="Prop. paired edges symmetric",
        ax=ax,
        label="Edges symm. (1)",
        s=50,
    )
    sns.scatterplot(
        data=thresh_result_df_min2,
        x="threshold",
        y="Prop. paired edges symmetric",
        ax=ax,
        color="purple",
        label="Edges symm. (2)",
        s=50,
    )
    sns.scatterplot(
        data=thresh_result_df,
        x="threshold",
        y="Prop. edges left",
        ax=ax,
        color="orange",
        label="Edges left",
        s=30,
        marker="d",
    )
    sns.scatterplot(
        data=thresh_result_df,
        x="threshold",
        y="Prop. synapses left",
        ax=ax,
        color="green",
        label="Synapses left",
        s=30,
        marker="d",
    )
    ax.set_title(f"{graph_type}")
    pad = 0.03
    ax.set_ylim((0 - pad, 1 + pad))
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
    ax.set_ylabel("Proportion")
    ax.set_xticks(np.arange(0, 21, 2))
    savename = (
        f"threshold-synapses-{graph_type}-min-dend{input_thresh}"
        + f"-rem-pdiff{remove_pdiff}-subset{use_subset}"
    )
    stashfig(savename)
    stashcsv(thresh_result_df, savename)

