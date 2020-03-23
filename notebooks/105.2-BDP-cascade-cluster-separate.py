# %% [markdown]
# #
import itertools
import os
import time
from itertools import chain

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, Node, RenderTree
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA

from graspy.plot import heatmap, pairplot
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels
from src.traverse import (
    cascades_from_node,
    generate_cascade_tree,
    generate_random_walks,
    path_to_visits,
    to_markov_matrix,
    to_path_graph,
)
from src.visualization import (
    CLASS_COLOR_DICT,
    barplot_text,
    draw_networkx_nice,
    draw_separators,
    matrixplot,
    remove_shared_ax,
    remove_spines,
    screeplot,
    sort_meta,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


#%% Load and preprocess the data

VERSION = "2020-03-09"
print(f"Using version {VERSION}")

graph_type = "Gad"
threshold = 0
weight = "weight"
mg = load_metagraph(graph_type, VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight=weight,
)
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")

# TODO update this with the mixed groups
# TODO make these functional for selecting proper paths

inout = "sensory_to_out"
if inout == "sensory_to_out":
    out_classes = [
        "O_dSEZ",
        "O_dSEZ;CN",
        "O_dSEZ;LHN",
        "O_dVNC",
        "O_dVNC;O_RG",
        "O_dVNC;CN",
        "O_RG",
        "O_dUnk",
        "O_RG-IPC",
        "O_RG-ITP",
        "O_RG-CA-LP",
    ]
    from_groups = [
        ("sens-ORN",),
        ("sens-photoRh5", "sens-photoRh6"),
        ("sens-MN",),
        ("sens-thermo",),
        ("sens-vtd",),
        ("sens-AN",),
    ]
    from_group_names = ["Odor", "Photo", "MN", "Temp", "VTD", "AN"]

if inout == "out_to_sensory":
    from_groups = [
        ("motor-mAN", "motormVAN", "motor-mPaN"),
        ("O_dSEZ", "O_dVNC;O_dSEZ", "O_dSEZ;CN", "LHN;O_dSEZ"),
        ("O_dVNC", "O_dVNC;CN", "O_RG;O_dVNC", "O_dVNC;O_dSEZ"),
        ("O_RG", "O_RG-IPC", "O_RG-ITP", "O_RG-CA-LP", "O_RG;O_dVNC"),
        ("O_dUnk",),
    ]
    from_group_names = ["Motor", "SEZ", "VNC", "RG", "dUnk"]
    out_classes = [
        "sens-ORN",
        "sens-photoRh5",
        "sens-photoRh6",
        "sens-MN",
        "sens-thermo",
        "sens-vtd",
        "sens-AN",
    ]

from_classes = list(chain.from_iterable(from_groups))  # make this a flat list

class_key = "Merge Class"

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
n_verts = len(adj)
meta = mg.meta.copy()
g = mg.g.copy()
meta["inds"] = range(len(meta))

from_inds = meta[meta[class_key].isin(from_classes)]["inds"].values
out_inds = meta[meta[class_key].isin(out_classes)]["inds"].values
ind_map = dict(zip(meta.index, meta["inds"]))
g = nx.relabel_nodes(g, ind_map, copy=True)
out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# ## Organize metadata for the neurons
ids = pd.Series(index=meta["inds"], data=meta.index, name="id")
to_class = ids.map(meta["Merge Class"])
to_class.name = "to_class"
colors = to_class.map(CLASS_COLOR_DICT)
colors.name = "color"
col_df = pd.concat([ids, to_class, colors], axis=1)

# %% [markdown]
# # Setup to generate cascades/random walks

from src.traverse import to_transmission_matrix, TraverseDispatcher, Cascade

p = 0.05
n_init = 1000
seed = 8888
max_hops = 10
cascade_kws = {
    "n_init": n_init,
    "max_hops": max_hops,
    "stop_nodes": [],
    "allow_loops": False,
    "simultaneous": True,
}
np.random.seed(seed)
transition_probs = to_transmission_matrix(adj, p)


def hist_from_cascade(
    transition_probs,
    start_nodes,
    stop_nodes=[],
    allow_loops=False,
    n_init=100,
    simultaneous=True,
    max_hops=10,
):
    dispatcher = TraverseDispatcher(
        Cascade,
        transition_probs,
        allow_loops=allow_loops,
        n_init=n_init,
        max_hops=max_hops,
        stop_nodes=stop_nodes,
    )
    if simultaneous:
        hop_hist = dispatcher.start(start_nodes)
    else:
        n_verts = len(transition_probs)
        hop_hist = np.zeros((n_verts, max_hops))
        for s in start_nodes:
            hop_hist += dispatcher.start(s)
    return hop_hist


# %% [markdown]
# ## Run the cascades, plot outputs for each channel separately

sns.set_context("talk")
print(f"Running cascades, n_init={n_init}")
fg_hop_hists = []
fg_col_meta = []
for fg, fg_name in zip(from_groups, from_group_names):
    print(fg_name)

    # running the cascadese
    from_inds = meta[meta[class_key].isin(fg)]["inds"].values
    hop_hist = hist_from_cascade(transition_probs, from_inds, **cascade_kws).T
    fg_hop_hists.append(hop_hist)

    # add some metadata based on the cascade results just for plotting
    n_visits = np.sum(hop_hist, axis=0)
    weight_sum_visits = (np.arange(1, max_hops + 1)[:, None] * hop_hist).sum(axis=0)
    mean_visit = weight_sum_visits / n_visits
    fg_col_df = col_df.copy()
    fg_col_df["mean_visit"] = mean_visit
    group_means = fg_col_df.groupby("to_class")["mean_visit"].mean()
    fg_col_df["group_mean_visit"] = fg_col_df["to_class"].map(group_means)
    fg_col_meta.append(fg_col_df)

    # plot the super row
    fig, axs = plt.subplots(
        2, 1, figsize=(25, 15), gridspec_kw=dict(height_ratios=[0.95, 0.05])
    )
    ax = axs[0]
    matrixplot(
        np.log10(hop_hist + 1),
        ax=ax,
        col_meta=fg_col_df,
        col_sort_class=["to_class"],
        col_class_order="group_mean_visit",
        col_item_order=["mean_visit"],
        col_colors=CLASS_COLOR_DICT,
        col_ticks=False,
        cbar_kws=dict(fraction=0.05),
    )
    ax = axs[-1]
    ax.axis("off")
    stashfig(f"{fg_name}")

# %% [markdown]
# ## Cluster the "superrows"
from graspy.cluster import AutoGMMCluster

cluster_kws = dict(
    min_components=5, max_components=40, affinity=["euclidean", "manhattan"], n_jobs=-1
)
fg_autoclusters = []
print("Running GMM")
for i, (fg, fg_name) in enumerate(zip(from_groups[:1], from_group_names[:1])):
    print(f"Clustering for {fg_name}")

    # run the clustering on histogram
    hop_hist = fg_hop_hists[i]
    agmm = AutoGMMCluster(**cluster_kws)
    pred_labels = agmm.fit_predict(hop_hist.T)
    results = agmm.results_
    fg_col_meta[i]["pred_labels"] = pred_labels
    fg_autoclusters.append(agmm)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.scatterplot(data=results, y="bic/aic", x="n_components", label="all", ax=ax)
    best_results = results.groupby("n_components")["bic/aic"].min()
    best_results = best_results.reset_index()
    sns.scatterplot(
        data=best_results, y="bic/aic", x="n_components", label="best", ax=ax
    )
    ax.set_ylabel("BIC")
    ax.set_xlabel("K")
    stashfig("odor-bic")

# %% [markdown]
# ##
k = 26
idx = results[results["n_components"] == k]["bic/aic"].idxmin()
model = results.loc[idx, "model"]

fig, axs = plt.subplots(
    2, 1, figsize=(25, 15), gridspec_kw=dict(height_ratios=[0.97, 0.03])
)
ax = axs[0]
meta = fg_col_meta[i].copy()
meta["cluster"] = model.predict(hop_hist.T)
cluster_means = meta.groupby("cluster")["mean_visit"].mean()
meta["cluster_mean_visit"] = meta["cluster"].map(cluster_means)

out = matrixplot(
    np.log10(fg_hop_hists[i] + 1),
    ax=ax,
    col_meta=meta,
    col_sort_class=["cluster"],
    col_class_order="cluster_mean_visit",
    col_item_order=["to_class", "mean_visit"],
    col_ticks=True,
    col_colors="to_class",
    col_palette=CLASS_COLOR_DICT,
    cbar_kws=dict(fraction=0.05),
)

ax = axs[-1]
ax.axis("off")
fg_name = "Odor"
caption = (
    f"Figure x: Hop histogram for cascades from {fg_name}, simultaneous cascades\n"
)
caption += f"from all sensory neurons in {fg_name}, p={p}."
caption += (
    f"Columns (neurons) are sorted into clusters (GMM, K={k}), clusters are sorted\n"
)
caption += f"by mean hop, and within cluster, by the mean hop for an individual neuron."
ax.invert_yaxis()
ax.text(0, 0, caption)  # va="top")
# plt.tight_layout()
stashfig(f"{fg_name}-cluster-k={k}")

