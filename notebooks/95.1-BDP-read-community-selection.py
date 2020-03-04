# %% [markdown]
# #
import os
from pathlib import Path

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import ParameterGrid

from graspy.utils import cartprod
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig
from src.utils import get_blockmodel_df
from src.visualization import (
    CLASS_COLOR_DICT,
    CLASS_IND_DICT,
    barplot_text,
    probplot,
    remove_spines,
    stacked_barplot,
)


mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)
BRAIN_VERSION = "2020-03-02"


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


# %% [markdown]
# # Load the runs

run_dir = Path("81.3-BDP-community")
base_dir = Path("./maggot_models/notebooks/outs")
block_file = base_dir / run_dir / "csvs" / "block-labels.csv"
block_df = pd.read_csv(block_file, index_col=0)

run_names = block_df.columns.values
n_runs = len(block_df.columns)
block_pairs = cartprod(range(n_runs), range(n_runs))

opt_dir = Path("94.1-BDP-community-selection")
param_file = base_dir / opt_dir / "csvs" / "best_params.csv"
param_df = pd.read_csv(param_file, index_col=0)

best_block_df = block_df[param_df.index]
# %% [markdown]
# #

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="MB-ARI", y="MB-cls", ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="MB-cls", y="AL-cls", ax=ax)


# %% [markdown]
# # plot results

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(data=param_df, x="MB-ARI", y="pairedness", ax=axs[0])
sns.scatterplot(data=param_df, x="MB-ARI", y="adj_pairedness", ax=axs[1])

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.scatterplot(data=param_df, x="pairedness", y="adj_pairedness", ax=axs[1])
# %% [markdown]
# #

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="MB-roARI", y="pairedness", ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="MB-moARI", y="pairedness", ax=ax)

# %% [markdown]
# #

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=param_df, x="AL-moARI", y="pairedness", ax=ax)


# %% [markdown]
# #
pg = sns.PairGrid(
    data=param_df,
    x_vars=["MB-ARI", "MB-roARI", "MB-moARI"],
    y_vars=["MB-ARI", "MB-roARI", "MB-moARI"],
)
pg.map(sns.scatterplot)

pg = sns.PairGrid(
    data=param_df,
    x_vars=["AL-ARI", "AL-roARI", "AL-moARI"],
    y_vars=["AL-ARI", "AL-roARI", "AL-moARI"],
)
pg.map(sns.scatterplot)

# %% [markdown]
# #
plot_vars = ["rank_MB-moARI", "rank_AL-moARI", "rank_pairedness"]
pg = sns.PairGrid(
    data=param_df,
    x_vars=plot_vars,
    y_vars=plot_vars,
    hue="Parameters",
    palette=cc.glasbey_light,
    height=4,
)
pg.map(sns.scatterplot, s=30)
# stashfig("pairwise")
# %% [markdown]
# #
rank_df = param_df.rank(axis=0, ascending=False)
param_df.loc[rank_df.index, "rank_MB-ARI"] = rank_df["MB-ARI"]
param_df.loc[rank_df.index, "rank_MB-cls"] = rank_df["MB-cls"]
param_df.loc[rank_df.index, "rank_MB-moARI"] = rank_df["MB-moARI"]
param_df.loc[rank_df.index, "rank_MB-roARI"] = rank_df["MB-roARI"]
param_df.loc[rank_df.index, "rank_AL-ARI"] = rank_df["AL-ARI"]
param_df.loc[rank_df.index, "rank_AL-cls"] = rank_df["AL-cls"]
param_df.loc[rank_df.index, "rank_AL-moARI"] = rank_df["AL-moARI"]
param_df.loc[rank_df.index, "rank_AL-roARI"] = rank_df["AL-roARI"]
param_df.loc[rank_df.index, "rank_pairedness"] = rank_df["pairedness"]
param_df.loc[rank_df.index, "rank_adj_pairedness"] = rank_df["adj_pairedness"]

#%%
param_df.sort_values("pairedness", ascending=False)

# %% [markdown]
# # Plot a candidate

# idx = sort_index[2]
idx = "LorenBerglund"
preprocess_params = dict(param_df.loc[idx, ["binarize", "threshold"]])
graph_type = param_df.loc[idx, "graph_type"]
mg = load_metagraph(graph_type, version=BRAIN_VERSION)
mg = preprocess(mg, sym_threshold=True, remove_pdiff=True, **preprocess_params)

labels = np.zeros(len(mg.meta))

pred_labels = best_block_df[idx]
pred_labels = pred_labels[pred_labels.index.isin(mg.meta.index)]
partition = pred_labels.astype(int)
title = idx
class_labels = mg["Merge Class"]
lineage_labels = mg["lineage"]
basename = idx


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
probplot(100 * prob_df, fmt="2.0f", figsize=(20, 20), title=title, font_scale=0.7)
stashfig(basename + f"probplot-counts{counts}-weights{weights}")


# %% Extract the antennal lobe clusters
communities = [2, 3]

from src.io import saveskels

for comm in communities:
    inds = partition[partition == comm].index
    comm_meta = mg.meta.loc[inds]
    class_labels = comm_meta["Merge Class"]
    colors = np.vectorize(CLASS_COLOR_DICT.get)(class_labels)
    saveskels(
        f"cluster-{comm}",
        inds,
        class_labels,
        colors,
        multiout=False,
        palette=None,
        foldername=FNAME,
    )
    saveskels(
        f"cluster-{comm}",
        inds,
        class_labels,
        colors,
        multiout=True,
        palette=None,
        foldername=FNAME,
    )

# %% [markdown]
# # Plot the adjacency matrices for a community

import community as cm
from operator import itemgetter


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


from graspy.plot import heatmap
from graspy.utils import symmetrize

communities = [2, 3]

al_classes = [
    "bLN-Duet",
    "bLN-Trio",
    "cLN",
    "keystone",
    "mPN-multi",
    "mPN-olfac",
    "mPN;FFN-multi",
    "pLN",
    "uPN",
    "sens-ORN",
]

for comm in communities:
    comm_mg = mg.copy()
    ids = partition[partition == comm].index
    inds = comm_mg.meta.index.isin(ids)
    comm_mg = comm_mg.reindex(inds)
    is_al = comm_mg.meta["Merge Class"].isin(al_classes)
    heatmap(
        comm_mg.adj,
        inner_hier_labels=comm_mg["Merge Class"],
        outer_hier_labels=is_al,
        hier_label_fontsize=7,
        figsize=(20, 20),
        cbar=False,
    )
    adj = comm_mg.adj.copy()
    adj = symmetrize(adj, method="avg")
    sym_mg = MetaGraph(adj, comm_mg.meta)
    g_sym = sym_mg.g
    skeleton_labels = np.array(list(g_sym.nodes()))
    sub_partition, modularity = run_louvain(g_sym, 1, skeleton_labels)
    sub_partition = pd.Series(data=sub_partition, index=skeleton_labels)
    sub_partition.name = "sub-partition"
    sub_partition = sub_partition.reindex(comm_mg.meta.index)
    heatmap(
        comm_mg.adj,
        inner_hier_labels=sub_partition.values,
        hier_label_fontsize=7,
        figsize=(20, 20),
        cbar=False,
        sort_nodes=True,
    )

# %%

# %%

# %% [markdown]
# #
label_series = pd.Series(data=np.zeros(len(mg)), index=mg.meta.index)
for i, comm in enumerate(communities):
    inds = partition[partition == comm].index
    label_series[inds] = i + 1

heatmap(
    mg.adj,
    inner_hier_labels=label_series.values,
    hier_label_fontsize=7,
    figsize=(20, 20),
    cbar=False,
)
stashfig("whole-adj")

# %% [markdown]
# #

heatmap(mg.adj, hier_label_fontsize=7, figsize=(20, 20))

# %% [markdown]
# #
comm_mg

perm_inds = np.random.permutation(len(comm_mg))
adj = comm_mg.adj
labels = comm_mg["Merge Class"]

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
heatmap(
    adj[np.ix_(perm_inds, perm_inds)],
    inner_hier_labels=labels[perm_inds],
    hier_label_fontsize=7,
    figsize=(20, 20),
    cbar=False,
    ax=axs[0],
    sort_nodes=True,
)

heatmap(
    adj,
    inner_hier_labels=labels,
    hier_label_fontsize=7,
    figsize=(20, 20),
    cbar=False,
    ax=axs[1],
    sort_nodes=True,
)

# %% [markdown]
# #

# meta["class_size"] = meta.apply(
#     lambda x: class_size[tuple(x[sc] for sc in sort_class)], axis=1
# )  # HACK my dumbass multiindex map function


#%%

from graspy.utils import pass_to_ranks
from src.visualization import gridmap, CLASS_COLOR_DICT
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

is_al.name = "AL"


# Inputs
mg = comm_mg
data = comm_mg.adj.copy()
plot_type = "heatmap"
# sort_class = ["AL", "Merge Class"]
sort_class = ["Merge Class"]
meta = mg.meta.copy()  # TODO may have to construct this in some cases
ax = None
tick_fontsize = 10
sizes = (10, 40)
cmap = "RdBu_r"
gridline_kws = dict(color="grey", linestyle="--", alpha=0.7, linewidth=1)
spinestyle_kws = dict(linestyle="-", linewidth=1, alpha=0.7, color="black")
border = True
_, ax = plt.subplots(1, 1, figsize=(20, 20))
use_colors = True
use_ticks = True
tick_pad = [0, 0]
base_tick_pad = 5

# initialize a metadataframe
meta = pd.concat((meta, is_al), axis=1)
# TODO check for bad values in the columns, if there, rename these

# sort the metadata
total_sort_by = []
for sc in sort_class:
    class_size = meta.groupby(sc).size()
    # negative so we can sort alphabetical still in one line
    meta[f"{sc}_size"] = -meta[sc].map(class_size)
    total_sort_by.append(f"{sc}_size")
    total_sort_by.append(sc)
total_sort_by.append("dendrite_input")  # TODO placeholder for degree

meta["idx"] = range(len(meta))
sort_meta = meta.sort_values(total_sort_by, inplace=False)
perm_inds = sort_meta.idx.values

# handle transformations of the data itself
# TODO do it the graspy way
data = data[np.ix_(perm_inds, perm_inds)]
data = pass_to_ranks(data)

# get locations
sort_meta["idx"] = range(len(sort_meta))
# for the gridlines
first_df = sort_meta.groupby(sort_class, sort=False).first()
first_inds = list(first_df["idx"].values)[1:]  # skip since we have spines
# for the tic locs
middle_df = sort_meta.groupby(sort_class, sort=False).mean()
middle_inds = list(middle_df["idx"].values)
middle_labels = list(middle_df.index)

if ax is None:
    _, ax = plt.subplots(1, 1, figsize=(10, 10))

# do the actual plotting!
if plot_type == "heatmap":
    sns.heatmap(data, cmap=cmap, ax=ax, vmin=0, center=0, cbar=False, square=True)
elif plot_type == "scattermap":
    gridmap(data, ax=ax, sizes=sizes, border=False)

ax.axis("square")
ax.set_ylim(len(data), -1)
ax.set_xlim(-1, len(data))

# add grid lines separating classes
if plot_type == "heatmap":
    boost = 0
elif plot_type == "scattermap":
    boost = 0.5
for t in first_inds:
    ax.axvline(t - boost, **gridline_kws)
    ax.axhline(t - boost, **gridline_kws)

if use_colors:  # TODO experimental!
    ax.set_xticks([])
    ax.set_yticks([])
    for sc in sort_class:  # TODO this will break for more than one category
        divider = make_axes_locatable(ax)
        left_cax = divider.append_axes("left", size="3%", pad=0, sharey=ax)
        top_cax = divider.append_axes("top", size="3%", pad=0, sharex=ax)
        # left_cax.set_ylim(ax.get_ylim())ƒ
        classes = sort_meta[sc].values
        class_colors = np.vectorize(CLASS_COLOR_DICT.get)(
            classes
        )  # TODO make not specific

        from matplotlib.colors import ListedColormap

        # make colormap
        uni_classes = np.unique(classes)
        class_map = dict(zip(uni_classes, range(len(uni_classes))))
        color_list = []
        for u in uni_classes:
            color_list.append(CLASS_COLOR_DICT[u])
        lc = ListedColormap(color_list)
        classes = np.vectorize(class_map.get)(classes)
        classes = classes.reshape(len(classes), 1)
        sns.heatmap(
            classes,
            cmap=lc,
            cbar=False,
            yticklabels=False,
            xticklabels=False,
            ax=left_cax,
            square=False,
        )
        classes = classes.T  # reshape(len(classes), 1)
        sns.heatmap(
            classes,
            cmap=lc,
            cbar=False,
            yticklabels=False,
            xticklabels=False,
            ax=top_cax,
            square=False,
        )


if use_ticks:
    if use_ticks and use_colors:
        top_tick_ax = top_cax
        left_tick_ax = left_cax
        top_tick_ax.set_yticks([])
        left_tick_ax.set_xticks([])
    else:
        top_tick_ax = left_tick_ax = ax

    # add tick labels and locs
    top_tick_ax.set_xticks(middle_inds)
    top_tick_ax.set_xticklabels(middle_labels)
    left_tick_ax.set_yticks(middle_inds)
    left_tick_ax.set_yticklabels(middle_labels)

    # modify the padding / offset every other tick
    for i, axis in enumerate([top_tick_ax.xaxis, left_tick_ax.yaxis]):
        axis.set_major_locator(plt.FixedLocator(middle_inds[0::2]))
        axis.set_minor_locator(plt.FixedLocator(middle_inds[1::2]))
        axis.set_minor_formatter(plt.FormatStrFormatter("%s"))

    top_tick_ax.tick_params(which="minor", pad=tick_pad[i] + base_tick_pad, length=5)
    top_tick_ax.tick_params(which="major", pad=base_tick_pad, length=5)
    left_tick_ax.tick_params(which="minor", pad=tick_pad[i] + base_tick_pad, length=5)
    left_tick_ax.tick_params(which="major", pad=base_tick_pad, length=5)

    top_tick_ax.set_xticklabels(middle_labels[0::2])
    top_tick_ax.set_xticklabels(middle_labels[1::2], minor=True)
    left_tick_ax.set_yticklabels(middle_labels[0::2])
    left_tick_ax.set_yticklabels(middle_labels[1::2], minor=True)
    top_tick_ax.xaxis.tick_top()

    # set tick size and rotation
    for tick in top_tick_ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_fontsize(tick_fontsize)
    for tick in top_tick_ax.get_xticklabels(minor=True):
        tick.set_rotation(45)
        tick.set_fontsize(tick_fontsize)
    for tick in left_tick_ax.get_yticklabels():
        tick.set_fontsize(tick_fontsize)
    for tick in left_tick_ax.get_yticklabels(minor=True):
        tick.set_fontsize(tick_fontsize)

    if use_colors and use_ticks:
        shax = ax.get_shared_x_axes()
        shay = ax.get_shared_y_axes()
        shax.remove(ax)
        shay.remove(ax)
        xticker = mpl.axis.Ticker()
        for axis in [ax.xaxis, ax.yaxis]:
            axis.major = xticker
            axis.minor = xticker
            loc = mpl.ticker.NullLocator()
            fmt = mpl.ticker.NullFormatter()
            axis.set_major_locator(loc)
            axis.set_major_formatter(fmt)
            axis.set_minor_locator(loc)
            axis.set_minor_formatter(fmt)

# spines
if border:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(spinestyle_kws["color"])
        spine.set_linewidth(spinestyle_kws["linewidth"])
        spine.set_linestyle(spinestyle_kws["linestyle"])
        spine.set_alpha(spinestyle_kws["alpha"])

# stashfig("sorted-adj", dpi=300)


# %%

# %% [markdown]
# #
pred_labels = pred_labels[pred_labels.index.isin(mg.meta.index)]
partition = pred_labels.astype(int)
fig, axs = barplot_text(
    sub_partition.values,
    comm_mg["Merge Class"],
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
    figsize=(24, 18),
    title=title,
    hatch_dict=None,
)

# %% [markdown]
# #


counts = False
weights = False
prob_df = get_blockmodel_df(
    mg.adj, partition, return_counts=counts, use_weights=weights
)
prob_df = prob_df.reindex(category_order, axis=0)
prob_df = prob_df.reindex(category_order, axis=1)
probplot(100 * prob_df, fmt="2.0f", figsize=(20, 20), title=title, font_scale=0.7)
stashfig(basename + f"probplot-counts{counts}-weights{weights}")


#%%
from src.hierarchy import signal_flow

from graspy.embed import AdjacencySpectralEmbed


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
        prob_df = pd.DataFrame(data=adj, index=prob_df.index, columns=prob_df.columns)

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


sns.set_context("talk")
plt.style.use("seaborn-white")
sns.set_palette("deep")

ad_mg = load_metagraph("Gad", version=BRAIN_VERSION)
ad_mg = preprocess(ad_mg, sym_threshold=True, remove_pdiff=True, **preprocess_params)
mg.meta["idx"] = range(len(mg))
ad_mg.reindex()

minigraph = to_minigraph(ad_mg.adj, partition)
from src.visualization import draw_networkx_nice

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
# fig, axs = barplot_text(
#     sub_partition.values,
#     comm_mg["Merge Class"],
#     color_dict=color_dict,
#     plot_proportions=False,
#     norm_bar_width=True,
#     figsize=(24, 18),
#     title=title,
#     hatch_dict=None,
# )
from src.visualization import stacked_barplot

stacked_barplot(
    partition,
    mg["Merge Class"],
    color_dict=color_dict,
    plot_proportions=False,
    norm_bar_width=True,
    hatch_dict=None,
    ax=axs[0],
)
draw_networkx_nice(
    minigraph,
    "Spring-x",
    "Spring-y",
    sizes="Size",
    colors="Color",
    ax=axs[1],
    weight_scale=20,
    vmin=0.0001,
)
axs[1].set_xlabel("")
axs[1].set_ylabel("")
for ax in axs:
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

stashfig("try-clusterplot")

