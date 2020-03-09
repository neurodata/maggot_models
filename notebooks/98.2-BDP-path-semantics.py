# %% [markdown]
# #
import csv
import itertools
import os
import time
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import textdistance
from joblib import Parallel, delayed
from pandas.plotting import parallel_coordinates
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid

from graspy.cluster import AutoGMMCluster
from graspy.embed import AdjacencySpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import get_lcc, symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph, preprocess
from src.io import readlol, savecsv, savefig, savelol, saveskels
from src.traverse import (
    generate_random_cascade,
    generate_random_walks,
    path_to_visits,
    to_markov_matrix,
    to_path_graph,
)
from src.visualization import (
    CLASS_COLOR_DICT,
    barplot_text,
    draw_networkx_nice,
    remove_spines,
    screeplot,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def stashlol(df, name, **kws):
    savelol(df, name, foldername=FNAME, save_on=True, **kws)


#%% Load and preprocess the data

VERSION = "2020-03-05"
print(f"Using version {VERSION}")

graph_type = "Gad"
threshold = 0
weight = "weight"
all_out = False
mg = load_metagraph(graph_type, VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=True,
    remove_pdiff=True,
    binarize=False,
    weight=weight,
)
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")

# %% [markdown]
# # Setup the simulations
class_key = "Merge Class"

out_groups = [
    (
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
    )
]

sens_groups = [
    ("sens-ORN",),
    ("sens-photoRh5", "sens-photoRh6"),
    ("sens-PaN",),
    ("sens-MN",),
    ("sens-thermo",),
    ("sens-vtn",),
]

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
n_verts = len(adj)
meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))
ind_map = dict(zip(meta.index, meta["idx"]))
inv_map = dict(zip(meta["idx"], meta.index))

g = nx.relabel_nodes(g, ind_map, copy=True)
prob_mat = to_markov_matrix(adj)
n_walks = 100
max_walk = 30

meta["Merge Class"].unique()

# %% [markdown]
# ## Generate paths SOMEHOW


basename = (
    f"sm-paths-{graph_type}-t{threshold}-w{weight}-nwalks{n_walks}-maxwalk{max_walk}"
)


def run_random_walks(
    sens_classes=None, out_classes=None, seed=None, class_key="Merge Class"
):
    np.random.seed(seed)
    from_inds = meta[meta[class_key].isin(sens_classes)]["idx"].values
    out_inds = meta[meta[class_key].isin(out_classes)]["idx"].values
    paths, _ = generate_random_walks(
        prob_mat,
        from_inds,
        out_inds,
        n_walks=n_walks,
        max_walk=max_walk,
        return_stuck=False,
    )
    stashlol(paths, basename + f"{sens_classes}-{out_classes}")
    return paths


def path_histogram(paths, bins, density=False):
    visit_orders = path_to_visits(paths, n_verts, from_order=True)
    nodes = []
    hists = []
    for node, orders in visit_orders.items():
        hist, _ = np.histogram(orders, bins, density=density)
        nodes.append(node)
        hists.append(hist)
    return np.array(hists)


def random_walk_classes(in_groups, seed=None, class_key="Merge Class"):
    np.random.seed(seed)
    n_replicates = 1
    param_grid = {
        "out_classes": out_groups,
        "sens_classes": in_groups,
        "class_key": [class_key],
    }
    params = list(ParameterGrid(param_grid))
    seeds = np.random.choice(int(1e8), size=n_replicates * len(params), replace=False)

    rep_params = []
    for i, seed in enumerate(seeds):
        p = params[i % len(params)].copy()
        p["seed"] = seed
        rep_params.append(p)

    currtime = time.time()
    paths = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_random_walks)(**p) for p in rep_params
    )
    print(f"{time.time() - currtime} elapsed")
    bins = np.arange(0, 21, 1)

    dfs = []
    for p in paths:
        visit_orders = path_to_visits(p, n_verts, from_order=True)
        nodes = []
        hists = []
        for node, orders in visit_orders.items():
            hist, _ = np.histogram(orders, bins)
            nodes.append(node)
            hists.append(hist)
        data = np.array(hists)
        hist_df = pd.DataFrame(data=data, index=nodes, columns=bins[:-1] + 1)
        hist_df["id"] = hist_df.index.map(inv_map)
        hist_df.set_index("id", inplace=True)
        hist_df.loc[meta.index, "Merge Class"] = meta["Merge Class"]
        dfs.append(hist_df)

    data = []
    for i, df in enumerate(dfs):
        df = df.copy()
        df = df.drop("Merge Class", axis=1)
        data.append(df.values)
    data = np.concatenate(data, axis=1)
    classes = dfs[0]["Merge Class"].values
    return data, bins, classes


def path_clustermap(hist_data, labels, bins):
    colors = np.vectorize(CLASS_COLOR_DICT.get)(labels)
    for metric in ["euclidean"]:
        for linkage in ["average", "complete"]:
            cg = sns.clustermap(
                data=hist_data,
                col_cluster=False,
                row_colors=colors,
                cmap="RdBu_r",
                center=0,
                cbar_pos=None,
                method=linkage,
                metric=metric,
            )
            ax = cg.ax_heatmap
            for i, x in enumerate(
                np.arange(
                    bins.shape[0] - 1, hist_data.shape[-1], step=bins.shape[0] - 1
                )
            ):
                ax.axvline(x, linestyle="--", linewidth=1, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Response over time")
            ax.set_ylabel("Neuron")
            ax.set_title(f"metric={metric}, linkage={linkage}")


def one_iteration(start_labels, class_key="Merge Class"):
    # generate walks
    data, bins, classes = random_walk_classes(
        start_labels, seed=None, class_key=class_key
    )
    log_data = np.log10(data + 1)
    # plot the clustermap
    path_clustermap(log_data, classes, bins)
    # embed and plot by known class
    embedding = PCA(n_components=8).fit_transform(log_data)
    pairplot(embedding, labels=classes, palette=CLASS_COLOR_DICT)
    # cluster
    agm = AutoGMMCluster(min_components=2, max_components=20, n_jobs=-1, verbose=10)
    pred_labels = agm.fit_predict(embedding)
    plt.figure()
    sns.scatterplot(data=agm.results_, x="n_components", y="bic/aic")
    # plot embedding by cluster
    pairplot(embedding, labels=pred_labels, palette=cc.glasbey_light)
    # plot predicted clusters by known class
    stacked_barplot(pred_labels, classes, color_dict=CLASS_COLOR_DICT)
    return pred_labels


# %% [markdown]
# #

pred_labels = one_iteration(sens_groups)
#%%
meta["it1"] = pred_labels
pred_labels = one_iteration((np.unique(pred_labels),), class_key="it1")
