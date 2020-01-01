# %% [markdown]
# # Imports
import json
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

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
from src.utils import export_skeleton_json
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


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


def stashskel(name, ids, colors, palette=None, **kws):
    if SAVESKELS:
        return export_skeleton_json(
            name, ids, colors, palette=palette, foldername=FNAME, **kws
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
    text_pad=0.01,
    inverse_memberships=True,
    figsize=(24, 23),
    title=None,
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
        palette="tab20",
        norm_bar_width=show_props,
        return_data=True,
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

    for i, y in enumerate(ticks):
        x = 0
        for j, (colname, color) in enumerate(zip(uni_class, subcategory_colors)):
            prop = prop_data[i, j]
            if prop > 0:
                if inverse_memberships:
                    if show_props:
                        # find the size of the cluster, multiply by prop to get count
                        # get size of known cluster, divide to get proportion
                        cluster_name = leaf_names[i]
                        ind = np.where(uni_pred_labels == cluster_name)[0][0]
                        cluster_size = uni_pred_counts[ind]
                        prop = cluster_size * prop
                    prop = prop / uni_class_counts[j]
                    name = f"{colname} ({prop:3.0%})"
                else:
                    if show_props:
                        name = f"{colname} ({prop:3.0%})"
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
        if show_props:
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
from graspy.simulations import er_np, sbm


def n_to_labels(n):
    """Converts n vector (sbm input) to an array of labels
    
    Parameters
    ----------
    n : list or array
        length K vector indicating num vertices in each block
    
    Returns
    -------
    np.array
        shape (n_verts), indicator of what block each vertex 
        is in
    """
    n = np.array(n)
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


B1 = np.array([[0.3, 0.25, 0.25], [0.25, 0.3, 0.25], [0.25, 0.25, 0.7]])
B2 = np.array([[0.4, 0.25, 0.25], [0.25, 0.4, 0.25], [0.25, 0.25, 0.4]])
B3 = np.array([[0.25, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.25]])

n = np.array([300, 600, 600, 600, 700, 600, 300, 400]).astype(float)
n = n.astype(int)
block_labels = n_to_labels(n)
n_verts = np.sum(n)
global_p = 0.01
prop = np.array(
    [
        [0.4, 0.2, 0.4],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        [0.4, 0.2, 0.4],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        [0.4, 0.2, 0.4],
    ]
)
n_blocks = len(prop)
subblock_labels = block_labels.copy()

for i, (n_in_block, block_prop) in enumerate(zip(n, prop)):
    block_n = []
    for p in block_prop:
        num = int(p * n_in_block)
        block_n.append(num)
    temp_labels = n_to_labels(block_n) + n_blocks + i * 3
    subblock_labels[block_labels == i] = temp_labels


B_list = [B1, B2, B3, B1, B3, B3, B2, B1]
# B_list = [B1, B2, B1, B1, B3, B3, B1, B2]

graph = er_np(n_verts, global_p)
for i, n_sub_verts in enumerate(n):
    p = prop[i, :]
    n_vec = n_sub_verts * p
    n_vec = n_vec.astype(int)
    B = B_list[i]
    subgraph = sbm(n_vec, B)
    inds = block_labels == i
    graph[np.ix_(inds, inds)] = subgraph

heatmap(
    graph,
    figsize=(15, 15),
    cbar=False,
    inner_hier_labels=subblock_labels,
    outer_hier_labels=block_labels,
)

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import pairplot

ase = LaplacianSpectralEmbed(form="R-DAD")
latent = ase.fit_transform(graph)
pairplot(latent)

norm_latent = latent.copy()
norm_latent /= np.linalg.norm(latent, axis=1)[:, np.newaxis]
pairplot(norm_latent, labels=block_labels)

# %% [markdown]
# # Embedding
adj = graph
n_verts = adj.shape[0]
class_labels = block_labels

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
# %% [markdown]
# # Plotting divisive cluster hierarchy results

title = (
    f"Divisive hierarchical clustering, {cluster_type}, {embed_type}, {ptr_type},"
    + f" {brain_type}, {graph_type}"
)
class_labels = subblock_labels
name_base = f"-{cluster_type}-{embed_type}-{ptr_type}-{brain_type}-{graph_type}"
bartreeplot(dc, class_labels, show_props=True, inverse_memberships=False, title=title)
stashfig("bartree-props" + name_base)
bartreeplot(dc, class_labels, show_props=False, inverse_memberships=False, title=title)
stashfig("bartree-counts" + name_base)
bartreeplot(dc, class_labels, show_props=True, inverse_memberships=True, title=title)
stashfig("bartree-props-inv" + name_base)

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
bartreeplot(dc, class_labels, show_props=True, inverse_memberships=False, title=title)
stashfig("bartree-props" + name_base)
bartreeplot(dc, class_labels, show_props=False, inverse_memberships=False, title=title)
stashfig("bartree-counts" + name_base)
bartreeplot(dc, class_labels, show_props=True, inverse_memberships=True, title=title)
stashfig("bartree-props-inv" + name_base)

