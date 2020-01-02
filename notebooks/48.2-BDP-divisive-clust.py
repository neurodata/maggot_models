# %% [markdown]
# # Imports
import json
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from joblib.parallel import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import adjusted_rand_score, silhouette_score
from spherecluster import SphericalKMeans

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks
from src.cluster import DivisiveCluster
from src.data import load_everything
from src.embed import lse
from src.hierarchy import signal_flow
from src.io import savefig
from src.utils import export_skeleton_json, get_sbm_prob
from src.visualization import clustergram, palplot, sankey, stacked_barplot

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2019-12-18"

SAVEFIGS = True
SAVESKELS = True
SAVEOBJS = True

PTR = True
if PTR:
    ptr_type = "PTR"
else:
    ptr_type = "Raw"

ONLY_RIGHT = False
if ONLY_RIGHT:
    brain_type = "Right Hemisphere"
else:
    brain_type = "Full Brain"

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


def stashskel(name, ids, labels, palette=None, **kws):
    if SAVESKELS:
        return export_skeleton_json(
            name, ids, labels, palette=palette, foldername=FNAME, **kws
        )


def stashobj(obj, name, **kws):
    foldername = FNAME
    subfoldername = "objs"
    pathname = "./maggot_models/notebooks/outs"
    if SAVEOBJS:
        path = Path(pathname)
        if foldername is not None:
            path = path / foldername
            if not os.path.isdir(path):
                os.mkdir(path)
            if subfoldername is not None:
                path = path / subfoldername
                if not os.path.isdir(path):
                    os.mkdir(path)
        with open(path / str(name + ".pickle"), "wb") as f:
            pickle.dump(obj, f)


def preprocess_graph(adj, class_labels, skeleton_labels):
    # sort by number of synapses
    degrees = adj.sum(axis=0) + adj.sum(axis=1)
    sort_inds = np.argsort(degrees)[::-1]
    adj = adj[np.ix_(sort_inds, sort_inds)]
    class_labels = class_labels[sort_inds]
    skeleton_labels = skeleton_labels[sort_inds]

    # remove disconnected nodes
    adj, lcc_inds = get_lcc(adj, return_inds=True)
    class_labels = class_labels[lcc_inds]
    skeleton_labels = skeleton_labels[lcc_inds]

    # remove pendants
    degrees = np.count_nonzero(adj, axis=0) + np.count_nonzero(adj, axis=1)
    not_pendant_mask = degrees != 1
    not_pendant_inds = np.array(range(len(degrees)))[not_pendant_mask]
    adj = adj[np.ix_(not_pendant_inds, not_pendant_inds)]
    class_labels = class_labels[not_pendant_inds]
    skeleton_labels = skeleton_labels[not_pendant_inds]
    return adj, class_labels, skeleton_labels


def bartreeplot(
    dc,
    class_labels,
    show_props=True,
    print_props=True,
    text_pad=0.01,
    inverse_memberships=True,
    figsize=(24, 23),
    title=None,
    palette=cc.glasbey_light,
    color_dict=None,
):
    # gather necessary info from model
    linkage, labels = dc.build_linkage(bic_distance=False)  # hackily built like scipy's
    pred_labels = dc.predict(latent)
    uni_class_labels, uni_class_counts = np.unique(class_labels, return_counts=True)
    uni_pred_labels, uni_pred_counts = np.unique(pred_labels, return_counts=True)

    # set up the figure
    fig = plt.figure(figsize=figsize)
    r = fig.canvas.get_renderer()
    gs0 = plt.GridSpec(1, 2, figure=fig, width_ratios=[0.2, 0.8], wspace=0)
    gs1 = plt.GridSpec(1, 1, figure=fig, width_ratios=[0.2], wspace=0.1)

    # title the plot
    plt.suptitle(title, y=0.92, fontsize=30, x=0.5)

    # plot the dendrogram
    ax0 = fig.add_subplot(gs0[0])

    dendr_data = dendrogram(
        linkage,
        orientation="left",
        labels=labels,
        color_threshold=0,
        above_threshold_color="k",
        ax=ax0,
    )
    ax0.axis("off")
    ax0.set_title("Dendrogram", loc="left")

    # get the ticks from the dendrogram to apply to the bar plot
    ticks = ax0.get_yticks()

    # plot the barplot (and ticks to the right of them)
    leaf_names = np.array(dendr_data["ivl"])[::-1]
    ax1 = fig.add_subplot(gs0[1], sharey=ax0)
    ax1, prop_data, uni_class, subcategory_colors = stacked_barplot(
        pred_labels,
        class_labels,
        label_pos=ticks,
        category_order=leaf_names,
        ax=ax1,
        bar_height=5,
        horizontal_pad=0,
        palette=palette,
        norm_bar_width=show_props,
        return_data=True,
        color_dict=color_dict,
    )
    ax1.set_frame_on(False)
    ax1.yaxis.tick_right()

    if show_props:
        ax1_title = "Cluster proportion of known cell types"
    else:
        ax1_title = "Cluster counts by known cell types"

    ax1_title = ax1.set_title(ax1_title, loc="left")
    transformer = ax1.transData.inverted()
    bbox = ax1_title.get_window_extent(renderer=r)
    bbox_points = bbox.get_points()
    out_points = transformer.transform(bbox_points)
    xlim = ax1.get_xlim()
    ax1.text(
        xlim[1], out_points[0][1], "Cluster name (size)", verticalalignment="bottom"
    )

    # plot the cluster compositions as text to the right of the bars
    gs0.update(right=0.4)
    ax2 = fig.add_subplot(gs1[0], sharey=ax0)
    ax2.axis("off")
    gs1.update(left=0.48)

    text_kws = {
        "verticalalignment": "center",
        "horizontalalignment": "left",
        "fontsize": 12,
        "alpha": 1,
        "weight": "bold",
    }

    ax2.set_xlim((0, 1))
    transformer = ax2.transData.inverted()

    cluster_sizes = prop_data.sum(axis=1)
    for i, y in enumerate(ticks):
        x = 0
        for j, (colname, color) in enumerate(zip(uni_class, subcategory_colors)):
            prop = prop_data[i, j]
            if prop > 0:
                if inverse_memberships:
                    prop = prop / uni_class_counts[j]
                    name = f"{colname} ({prop:3.0%})"
                else:
                    if print_props:
                        name = f"{colname} ({prop / cluster_sizes[i]:3.0%})"
                    else:
                        name = f"{colname} ({prop})"
                text = ax2.text(x, y, name, color=color, **text_kws)
                bbox = text.get_window_extent(renderer=r)
                bbox_points = bbox.get_points()
                out_points = transformer.transform(bbox_points)
                width = out_points[1][0] - out_points[0][0]
                x += width + text_pad

    # deal with title for the last plot column based on options
    if inverse_memberships:
        ax2_title = "Known cell type (percentage of cell type in cluster)"
    else:
        if print_props:
            ax2_title = "Known cell type (percentage of cluster)"
        else:
            ax2_title = "Known cell type (count in cluster)"
    ax2.set_title(ax2_title, loc="left")


# Set up plotting constants
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=0.8)


# %% [markdown]
# # Load the data

adj, class_labels, side_labels, skeleton_labels = load_everything(
    "Gad",
    version=BRAIN_VERSION,
    return_keys=["Merge Class", "Hemisphere"],
    return_ids=True,
)


# select the right hemisphere
if ONLY_RIGHT:
    right_inds = np.where(side_labels == "R")[0]
    adj = adj[np.ix_(right_inds, right_inds)]
    class_labels = class_labels[right_inds]
    skeleton_labels = skeleton_labels[right_inds]

adj, class_labels, skeleton_labels = preprocess_graph(
    adj, class_labels, skeleton_labels
)
known_inds = np.where(class_labels != "Unk")[0]

# %% [markdown]
# # Embedding
n_verts = adj.shape[0]
latent = lse(adj, N_COMPONENTS, regularizer=None, ptr=PTR)
# pairplot(latent, labels=class_labels, title=embed)
latent_dim = latent.shape[1] // 2
print(f"ZG chose dimension {latent_dim} + {latent_dim}")
# %% [markdown]
# # Fitting divisive cluster model
start = timer()
dc = DivisiveCluster(n_init=N_INIT, cluster_method=CLUSTER_METHOD)
dc.fit(latent)
end = end = timer()
print()
print(f"DivisiveCluster took {(end - start)/60.0} minutes to fit")
print()
dc.print_tree(print_val="bic_ratio")
pred_labels = dc.predict(latent)

# %% [markdown]
# # Plotting divisive cluster hierarchy results


def get_colors(labels, pal=cc.glasbey_light, to_int=False, color_dict=None):
    uni_labels = np.unique(labels)
    if to_int:
        uni_labels = [int(i) for i in uni_labels]
    if color_dict is None:
        color_dict = get_color_dict(labels, pal=pal, to_int=to_int)
    colors = np.array(itemgetter(*labels)(color_dict))
    return colors


def get_color_dict(labels, pal="tab10", to_int=False):
    uni_labels = np.unique(labels)
    if to_int:
        uni_labels = [int(i) for i in uni_labels]
    if isinstance(pal, str):
        pal = sns.color_palette(pal, n_colors=len(uni_labels))
    color_dict = dict(zip(uni_labels, pal))
    return color_dict


n_classes = len(np.unique(class_labels))
class_color_dict = get_color_dict(class_labels, pal=cc.glasbey_cool)
pred_color_dict = get_color_dict(pred_labels, pal=cc.glasbey_warm)
all_color_dict = {**class_color_dict, **pred_color_dict}

title = (
    f"Divisive hierarchical clustering,"
    + f" {cluster_type}, {embed_type} ({latent_dim} + {latent_dim}), {ptr_type},"
    + f" {brain_type}, {graph_type}"
)
name_base = f"-{cluster_type}-{embed_type}-{ptr_type}-{brain_type}-{graph_type}"


fig, ax = plt.subplots(1, 1, figsize=(20, 30))
sankey(ax, class_labels, pred_labels, aspect=20, fontsize=16, colorDict=all_color_dict)
ax.axis("off")
ax.set_title(title, fontsize=30)

fig, ax = plt.subplots(1, 1, figsize=(20, 30))
sankey(ax, pred_labels, class_labels, aspect=20, fontsize=16, colorDict=all_color_dict)
ax.axis("off")
ax.set_title(title, fontsize=30)

# %% [markdown]
# #

sns.set_context("talk", font_scale=0.8)

bartreeplot(
    dc,
    class_labels,
    show_props=True,
    print_props=False,
    inverse_memberships=False,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-props" + name_base)
bartreeplot(
    dc,
    class_labels,
    show_props=False,
    print_props=True,
    inverse_memberships=False,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-counts" + name_base)
bartreeplot(
    dc,
    class_labels,
    show_props=True,
    inverse_memberships=True,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-props-inv" + name_base)
bartreeplot(
    dc,
    class_labels,
    show_props=False,
    inverse_memberships=True,
    title=title,
    color_dict=class_color_dict,
)
stashfig("bartree-counts-inv" + name_base)

# %% [markdown]
# #


# clustergram(adj, class_labels, pred_labels, title=title, color_dict=all_color_dict)
# generate colormap


# def get_color_dict(true_labels, pred_labels):
#     color_dict = {}
#     classes = np.unique(true_labels)
#     unk_ind = np.where(classes == "Unk")[0]  # hacky but it looks nice!
#     purp_ind = 4
#     in_purp_class = classes[purp_ind]
#     classes[unk_ind] = in_purp_class
#     classes[purp_ind] = "Unk"
#     known_palette = sns.color_palette("tab10", n_colors=len(classes))
#     for i, true_label in enumerate(classes):
#         color = known_palette[i]
#         color_dict[true_label] = color

#     classes = np.unique(pred_labels)
#     known_palette = sns.color_palette("gray", n_colors=len(classes))
#     for i, pred_label in enumerate(classes):
#         color = known_palette[i]
#         color_dict[pred_label] = color
#     return color_dict


# %% [markdown]
# #
colors = generate_colors(pred_labels)
stashskel("skels" + name_base, skeleton_labels, pred_labels, palette=colors)

# %% [markdown]
# # Fitting divisive cluster model
CLUSTER_METHOD = "auto-gmm"
cluster_type = "AutoGMM"
start = timer()
dc = DivisiveCluster(n_init=N_INIT, cluster_method=CLUSTER_METHOD)
dc.fit(latent)
end = end = timer()
print()
print(f"DivisiveCluster took {(end - start)/60.0} minutes to fit")
print()
dc.print_tree(print_val="bic_ratio")
# %% [markdown]
# # Plotting divisive cluster hierarchy results

title = (
    f"Divisive hierarchical clustering, {cluster_type}, {embed_type}, {ptr_type},"
    + f" {brain_type}, {graph_type}"
)

name_base = f"-{cluster_type}-{embed_type}-{GRAPH_TYPE}"
bartreeplot(dc, show_props=True, inverse_memberships=False, title=title)
stashfig("bartree-props" + name_base)
bartreeplot(dc, show_props=False, inverse_memberships=False, title=title)
stashfig("bartree-counts" + name_base)
bartreeplot(dc, show_props=True, inverse_memberships=True, title=title)
stashfig("bartree-props-inv" + name_base)
