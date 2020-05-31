# %% [markdown]
# ##
import os
import warnings
from itertools import chain

import colorcet as cc
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import poisson
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm
from umap import UMAP

from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import pairplot
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplace,
)
from src.align import Procrustes
from src.cluster import BinaryCluster, MaggotCluster, get_paired_inds
from src.data import load_metagraph
from src.flow import fit_gm_exp, make_exp_match
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.pymaid import start_instance
from src.traverse import Cascade, RandomWalk, to_markov_matrix, to_transmission_matrix
from src.utils import get_blockmodel_df
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    remove_shared_ax,
    remove_spines,
    screeplot,
    set_axes_equal,
    stacked_barplot,
)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "lightgrey",
    "ytick.color": "dimgrey",
    "xtick.color": "dimgrey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


# %% [markdown]
# ##

# parameters of clustering
metric = "bic"
bic_ratio = 1
d = 8  # embedding dimension
method = "color_iso"

basename = f"-method={method}-d={d}-bic_ratio={bic_ratio}"
title = f"Method={method}, d={d}, BIC ratio={bic_ratio}"

exp = "137.3-BDP-omni-clust"

# load data
cluster_meta = readcsv("meta" + basename, foldername=exp, index_col=0)
cluster_meta["lvl0_labels"] = cluster_meta["lvl0_labels"].astype(str)

# parameters of Bhat plotting
graph_type = "Gad"
level = 3
label_key = f"lvl{level}_labels"
use_sides = False
use_super = False
use_sens = False
if use_sides:
    label_key += "_side"
basename = (
    f"-{graph_type}-lvl={level}-sides={use_sides}-sup={use_super}-sens={use_sens}"
)

full_mg = load_metagraph(graph_type)
if use_super:
    super_mg = load_metagraph("Gs")
    super_mg = super_mg.reindex(full_mg.meta.index, use_ids=True)
    full_mg = MetaGraph(full_mg.adj + super_mg.adj, full_mg.meta)

meta = full_mg.meta

# get labels from the clustering
cluster_labels = cluster_meta[label_key]
meta["label"] = cluster_labels

# deal with sensories - will overwrite cluster labels for ORN
if use_sens:
    sens_meta = meta[meta["class1"] == "sens"]
    meta.loc[sens_meta.index, "label"] = sens_meta["merge_class"]

# deal with supers
if use_super:
    brain_names = ["Brain Hemisphere left", "Brain Hemisphere right"]
    super_meta = meta[meta["class1"] == "super"].copy()
    super_meta = super_meta[~super_meta["name"].isin(brain_names)]
    if not use_sides:

        def strip_side(x):
            x = x.replace("_left", "")
            x = x.replace("_right", "")
            return x

        super_meta["name"] = super_meta["name"].map(strip_side)

    meta.loc[super_meta.index, "label"] = super_meta["name"]

labeled_inds = meta[~meta["label"].isna()].index
full_mg = full_mg.reindex(labeled_inds, use_ids=True)
full_mg.meta.loc[:, "inds"] = range(len(full_mg))
print(len(full_mg))
print(full_mg["label"].unique())
# unlabeled_meta = full_mg.meta[full_mg.meta["label"].isna()]


def calc_bar_params(sizes, label, mid, palette=None):
    if palette is None:
        palette = CLASS_COLOR_DICT
    heights = sizes.loc[label]
    n_in_bar = heights.sum()
    offset = mid - n_in_bar / 2
    starts = heights.cumsum() - heights + offset
    colors = np.vectorize(palette.get)(heights.index)
    return heights, starts, colors


def plot_bar(meta, mid, ax, orientation="horizontal", width=0.7):
    if orientation == "horizontal":
        method = ax.barh
        ax.xaxis.set_visible(False)
        remove_spines(ax)
    elif orientation == "vertical":
        method = ax.bar
        ax.yaxis.set_visible(False)
        remove_spines(ax)
    sizes = meta.groupby("merge_class").size()
    sizes /= sizes.sum()
    starts = sizes.cumsum() - sizes
    colors = np.vectorize(CLASS_COLOR_DICT.get)(starts.index)
    for i in range(len(sizes)):
        method(mid, sizes[i], width, starts[i], color=colors[i])


meta = full_mg.meta
adj = full_mg.adj
fig, axs = plt.subplots(1, 2, figsize=(20, 10))


labels = meta["label"]
bar_ratio = 0.05
use_weights = True
use_counts = True
sort_method = "sf"
alpha = 0.05
width = 0.9
log = False
basename += f"-weights={use_weights}-counts={use_counts}-sort={sort_method}-log={log}"

blockmodel_df = get_blockmodel_df(
    adj, labels, return_counts=use_counts, use_weights=use_weights
)
heatmap_kws = dict(square=True, cmap="Reds", cbar_kws=dict(shrink=0.7))
data = blockmodel_df.values
# data = pass_to_ranks(data)
# data = np.log10(data + 1)
if log:
    data = np.log10(data)
    data[~np.isfinite(data)] = 0
    blockmodel_df = pd.DataFrame(
        data=data, index=blockmodel_df.index, columns=blockmodel_df.columns
    )

if sort_method == "sf":
    sf = signal_flow(data)
    perm = np.argsort(-sf)
if sort_method == "gm":
    perm, score = fit_gm_exp(data, alpha, 1, 0, n_init=20, return_best=True)


blockmodel_df = blockmodel_df.iloc[perm, perm]
data = blockmodel_df.values

uni_labels = blockmodel_df.index.values

ax = axs[0]
adjplot(data, ax=ax, cbar=False)
# sns.heatmap(data, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
divider = make_axes_locatable(ax)
top_ax = divider.append_axes("top", size=f"{bar_ratio*100}%", pad=0, sharex=ax)
left_ax = divider.append_axes("left", size=f"{bar_ratio*100}%", pad=0, sharey=ax)
remove_shared_ax(top_ax)
remove_shared_ax(left_ax)
mids = np.arange(len(data)) + 0.5

for i, label in enumerate(uni_labels):
    temp_meta = meta[meta["label"] == label]
    plot_bar(temp_meta, mids[i], left_ax, orientation="horizontal", width=width)
    plot_bar(temp_meta, mids[i], top_ax, orientation="vertical", width=width)

ax.yaxis.set_visible(True)
ax.yaxis.tick_right()
ax.yaxis.set_ticks(np.arange(len(data)) + 0.5)
ax.yaxis.set_ticklabels(uni_labels, fontsize=10, color="dimgrey", va="center")
ax.yaxis.set_tick_params(rotation=0, color="dimgrey")

if len(uni_labels) <= 10:
    pal = sns.color_palette("tab10")
elif len(uni_labels) <= 20:
    pal = sns.color_palette("tab20")
else:
    pal = cc.glasbey_light
color_map = dict(zip(uni_labels, pal))
ticklabels = axs[0].get_yticklabels()
for t in ticklabels:
    text = t.get_text()
    t.set_color(color_map[text])


remove_diag = True

# convert the adjacency and a partition to a minigraph based on SBM probs
prob_df = blockmodel_df
if remove_diag:
    adj = prob_df.values
    adj -= np.diag(np.diag(adj))
    prob_df = pd.DataFrame(data=adj, index=prob_df.index, columns=prob_df.columns)

g = nx.from_pandas_adjacency(prob_df, create_using=nx.DiGraph())
uni_labels, counts = np.unique(labels, return_counts=True)

# add size attribute base on number of vertices
size_map = dict(zip(uni_labels, counts))
nx.set_node_attributes(g, size_map, name="Size")

# add signal flow attribute (for the minigraph itself)
mini_adj = nx.to_numpy_array(g, nodelist=uni_labels)
node_signal_flow = signal_flow(mini_adj)
sf_map = dict(zip(uni_labels, node_signal_flow))
nx.set_node_attributes(g, sf_map, name="Signal Flow")

# rank signal flow
sort_inds = np.argsort(node_signal_flow)
rank_inds = np.argsort(sort_inds)
rank_sf_map = dict(zip(uni_labels, rank_inds))
nx.set_node_attributes(g, rank_sf_map, name="rank_sf")

# add spectral properties
sym_adj = symmetrize(mini_adj)
n_components = 5
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

nx.set_node_attributes(g, color_map, name="Color")

ax = axs[1]

x_pos_key = "AdjEvec-1"
y_pos_key = "rank_sf"
x_pos = nx.get_node_attributes(g, x_pos_key)
y_pos = nx.get_node_attributes(g, y_pos_key)

# all_x_pos = list(x_pos.items())
# all_y_pos = list(y_pos.items())
# y_max = max(all_y_pos)

if use_counts:
    vmin = 1000
    weight_scale = 1 / 2000
else:
    weight_scale = 1
    vmin = 0.01

draw_networkx_nice(
    g,
    x_pos_key,
    y_pos_key,
    colors="Color",
    sizes="Size",
    weight_scale=weight_scale,
    vmin=vmin,
    ax=ax,
    y_boost=0.3,
)
stashfig(f"layout-x={x_pos_key}-y={y_pos_key}" + basename)



# %%
