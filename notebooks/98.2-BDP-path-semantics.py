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
from src.traverse import (generate_random_cascade, generate_random_walks,
                          path_to_visits, to_markov_matrix, to_path_graph)
from src.visualization import (CLASS_COLOR_DICT, barplot_text,
                               draw_networkx_nice, remove_spines, screeplot,
                               stacked_barplot)

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
    ("sens-vtn",)
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


def run_random_walks(sens_classes=None, out_classes=None, seed=None, class_key="Merge Class"):
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
    param_grid = {"out_classes": out_groups, "sens_classes": in_groups, "class_key":[class_key]}
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
                np.arange(bins.shape[0] - 1, hist_data.shape[-1], step=bins.shape[0] - 1)
            ):
                ax.axvline(x, linestyle="--", linewidth=1, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Response over time")
            ax.set_ylabel("Neuron")
            ax.set_title(f"metric={metric}, linkage={linkage}")


def one_iteration(start_labels, class_key="Merge Class"):
    # generate walks
    data, bins, classes = random_walk_classes(start_labels, seed=None, class_key=class_key)
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
    sns.scatterplot(data=agm.results_, x='n_components', y='bic/aic')
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

# %% [markdown] 
# # 
run_random_walks([np.unique(pred_labels)], out_classes=out_groups, class_key='it1')
# %% [markdown]
# # Read paths from file



colors = []
sort_labels = []
for key, val in CLASS_COLOR_DICT.items():
    colors.append(val)
    sort_labels.append(key)

colors = np.array(colors)
sort_labels = np.array(sort_labels)
inds = np.argsort(sort_labels)
colors = colors[inds]
sort_labels = sort_labels[inds]

base_loc = "maggot_models/notebooks/outs/98.0-BDP-path-semantics/csvs/"
param_paths = []
bins = np.arange(0, 21, 1)
dfs = []
ridge_df = pd.DataFrame()
for p in params:
    paths = []
    filename = basename + f"{p['sens_classes']}-{p['out_classes']}"
    paths = readlol(filename, foldername=FNAME)
    param_paths.append(paths)
    # The below did note work very well
    visit_orders = path_to_visits(paths, n_verts, from_order=True)
    nodes = []
    hists = []
    for node, orders in visit_orders.items():
        hist, _ = np.histogram(orders, bins)
        nodes.append(node)
        hists.append(hist)
    data = np.array(hists)
    # data = data / np.sum(data, axis=0)[np.newaxis, :]
    hist_df = pd.DataFrame(data=data, index=nodes, columns=bins[:-1] + 1)
    hist_df["id"] = hist_df.index.map(inv_map)
    hist_df.set_index("id", inplace=True)
    hist_df.loc[meta.index, "Merge Class"] = meta["Merge Class"]
    dfs.append(hist_df)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    parallel_coordinates(
        hist_df, "Merge Class", ax=ax, axvlines=False, color=colors, sort_labels=True
    )
    ax.get_legend().remove()
    stashfig(str(p) + "-parallelplot")
    norm_hist = np.histogram(orders, bins, density=True)

# %% [markdown]
# # Try Seaborn ridgeplot
paths = param_paths[0]
rows = []
for path in paths:
    for i, node in enumerate(path):
        skel_id = inv_map[node]
        row = {
            "order": i + 1,
            "node": node,
            "id": skel_id,
            "class": meta.loc[skel_id, "Merge Class"],
        }
        rows.append(row)

all_visit_df = pd.DataFrame(rows)


df = all_visit_df
sort_df = df.groupby("class").median()
sort_values = sort_df.sort_values("order").index

n_per_class = meta.groupby("Merge Class").size()

sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
sns.set_context("talk", font_scale=1)
g = sns.FacetGrid(
    df,
    row="class",
    row_order=sort_values,
    hue="class",
    aspect=4,
    height=2,
    palette=CLASS_COLOR_DICT,
    sharey=True,
)
g.map(sns.kdeplot, "order", clip_on=False, shade=True, alpha=1, lw=1.5, bw=0.2)
g.map(sns.kdeplot, "order", clip_on=False, color="w", lw=1, bw=0.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
g.set_ylabels("")


def vline(x, color, label):
    ax = plt.gca()
    med = np.median(x)
    ax.axvline(med, 0, 0.5, linewidth=2, linestyle="--", color="grey")
    ax.text(
        med + 1, 0.4, int(med), fontweight="bold", color=color, ha="left", va="center"
    )


g.map(vline, "order")


def label(x, color, label):
    ax = plt.gca()
    text = label
    text += f"\n{len(x)} visits"
    n_in_class = n_per_class[label]
    text += f"\n{len(x) / n_in_class:.1f} visits/neuron"
    ax.text(
        0.65,
        0.45,
        text,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


g.map(label, "class")

g.fig.subplots_adjust(hspace=-0.35)

g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
stashfig(f"ridgeplot")


# %% [markdown]
# # make the figure of per cell, how likely is it to be visited after one hop
first_map = all_visit_df[all_visit_df["order"] == 2].groupby("id").size()

first_visit_df = pd.DataFrame(index=meta.index)
first_visit_df["p_first_visit"] = 0
first_visit_df.loc[first_map.index, "p_first_visit"] = (
    first_map.values / first_map.sum()
)

sns.distplot(
    np.log10(first_map.values / first_map.sum())  # bins=np.linspace(0, 0.03, 15)
)

# %% [markdown]
# # Do it from the markov matrix
from_inds = meta[meta[class_key].isin(("sens-ORN",))]["idx"].values

first_prob = probs[from_inds, :].sum(axis=0)

fig, ax = plt.subplots(1, 1)
sns.distplot(first_prob[first_prob != 0])  # bins=np.linspace(0, 0.03, 15)

fig, ax = plt.subplots(1, 1)
sns.distplot(np.log10(first_prob[first_prob != 0]))
# %% [markdown]
# #
p = 0.01
not_probs = (1 - p) ** adj  # probability of none of the synapses causing postsynaptic
probs = 1 - not_probs

# %% [markdown]
# #

log_probs = np.log10(first_prob[first_prob != 0])

plt.plot(np.sort(log_probs))
# %% [markdown]
# #


gmm = GaussianMixture(n_components=2, n_init=20)
pred = gmm.fit_predict(log_probs.reshape((len(log_probs), 1)))
fig, ax = plt.subplots(1, 1)
sns.distplot(log_probs[pred == 0])
sns.distplot(log_probs[pred == 1])

# %%
data = []
for i, df in enumerate(dfs):
    if i != 2:
        df = df.copy()
        df = df.drop("Merge Class", axis=1)
        data.append(df.values)
data = np.concatenate(data, axis=1)
# raw_hist_df = hist_df.drop("Merge Class", axis=1)
# raw_hist_data = raw_hist_df.values
raw_hist_data = data
raw_hist_data = np.log10(raw_hist_data + 1)

# %% [markdown]
# #

# nci_hc_complete = linkage(y=nci_data, method="complete", metric="euclidean")

# nci_hc_complete_4_clusters = cut_tree(
#     nci_hc_complete, n_clusters=4
# )  # Printing transpose just for space

# pd.crosstab(
#     index=nci_data.index,
#     columns=nci_hc_complete_4_clusters.T[0],
#     rownames=["Cancer Type"],
#     colnames=["Cluster"],
# )
# %% [markdown]
# #


embedding = PCA(n_components=8).fit_transform(raw_hist_data)
pairplot(embedding, labels=dfs[0]["Merge Class"].values, palette=CLASS_COLOR_DICT)


agg = AgglomerativeClustering(n_clusters=10, affinity="euclidean", linkage="average")
labels = agg.fit_predict(raw_hist_data)
pairplot(embedding, labels=labels, palette=cc.glasbey_light)

# %% [markdown]
# #


agm = AutoGMMCluster(min_components=2, max_components=20, n_jobs=-1
agm.fit(embedding)

# %% [markdown] 
# # 
# agm.results_.groupby(["affinity", "covariance_type", "linkage"])

# %% [markdown] 
# # 

new_groups = agm.predict(embedding)

stacked_barplot(new_groups, meta["Merge Class"].values, color_dict=CLASS_COLOR_DICT)
# %% [markdown] 
# # 

pairplot(embedding, labels=new_groups, palette='tab10')

# %% [markdown] 
# #
