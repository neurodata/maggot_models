#%%
from scipy.optimize import quadratic_assignment

import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson

from graspy.embed import OmnibusEmbed, selectSVD
from graspy.match import GraphMatch
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    to_laplace,
)

# from src.cluster import BinaryCluster
from src.data import load_metagraph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.utils import get_paired_inds
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
    set_theme,
)


colors = sns.color_palette("Set1")
palette = dict(zip(["Left", "Right"], colors))

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "lightgrey",
    "ytick.color": "grey",
    "xtick.color": "grey",
    "axes.labelcolor": "dimgrey",
    "text.color": "black",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}
set_theme(rc_dict=rc_dict)


def plot_adjs(left, right):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    adjplot(
        left,
        plot_type="scattermap",
        sizes=(2, 2),
        ax=axs[0],
        title=r"Left $\to$ left",
        color=palette["Left"],
    )
    adjplot(
        right,
        plot_type="scattermap",
        sizes=(2, 2),
        ax=axs[1],
        title=r"Right $\to$ right",
        color=palette["Right"],
    )


def load_graph(graph_type):
    master_mg = load_metagraph(graph_type, path=None, version=None)
    mg = MetaGraph(master_mg.adj, master_mg.meta)
    mg = mg.remove_pdiff()
    meta = mg.meta.copy()

    # remove low degree neurons
    degrees = mg.calculate_degrees()
    quant_val = np.quantile(degrees["Total edgesum"], 0.0)  # 0.05
    idx = meta[degrees["Total edgesum"] > quant_val].index
    # print(quant_val)
    mg = mg.reindex(idx, use_ids=True)
    mg = mg.make_lcc()
    meta = mg.meta

    # TODO the following block needs to be cleaned up, should make this easy to do with
    # MetaGraph class
    temp_meta = meta[meta["left"] | meta["right"]]
    unpair_idx = temp_meta[~temp_meta["pair"].isin(temp_meta.index)].index
    meta.loc[unpair_idx, ["pair", "pair_id"]] = -1

    left_idx = meta[meta["left"]].index
    left_mg = MetaGraph(mg.adj, mg.meta)
    left_mg = left_mg.reindex(left_idx, use_ids=True)
    left_mg = left_mg.sort_values(["pair_id"], ascending=False)
    #     print(len(left_mg))
    right_idx = meta[meta["right"]].index
    right_mg = MetaGraph(mg.adj, mg.meta)
    right_mg = right_mg.reindex(right_idx, use_ids=True)
    right_mg = right_mg.sort_values(["pair_id"], ascending=False)
    #     right_mg = right_mg.reindex(right_mg.meta.index[: len(left_mg)], use_ids=True)
    #     print(len(right_mg))
    n_pairs = len(right_mg.meta[right_mg.meta["pair_id"] != -1])
    print(n_pairs)
    left_adj = left_mg.adj
    right_adj = right_mg.adj
    #     shrink = min([len(left_adj),len(right_adj)])
    #     left_adj = left_adj[:shrink]
    #     right_adj = right_adj[:shrink]
    #     print(len(left_adj))
    #     print(len(right_adj))
    return left_mg, right_mg, n_pairs, meta


left_mg, right_mg, n_pairs, meta = load_graph("G")
left_adj = left_mg.adj
right_adj = right_mg.adj
left_adj_t = left_adj[:n_pairs, :n_pairs]
right_adj_t = right_adj[:n_pairs, :n_pairs]

#%%
import time

start = time.time()
options = {"maximize": True, "shuffle_input": True, "tol": 1e-14, "maxiter": 50}
res = min(
    [quadratic_assignment(left_adj_t, right_adj_t, options=options) for i in range(10)],
    key=lambda x: x.fun,
)
print(time.time() - start)
print(sum(res.col_ind == np.arange(n_pairs)) / n_pairs)

#%%
left_meta = left_mg.meta[:n_pairs].copy()
left_meta["_inds"] = range(len(left_meta))
left_meta = left_meta.sort_values(["simple_group"])
sort_inds = left_meta["_inds"]
perm_inds = res.col_ind
sort_left_adj = left_adj_t[np.ix_(sort_inds, sort_inds)]
sort_right_adj = right_adj_t[np.ix_(sort_inds, sort_inds)]
perm_right_adj = sort_right_adj[np.ix_(perm_inds, perm_inds)]
shuffle_inds = np.random.permutation(n_pairs)
shuffle_right_adj = right_adj_t[np.ix_(shuffle_inds, shuffle_inds)]
# plot_adjs(sort_left_adj, shuffle_right_adj)
# plot_adjs(sort_left_adj, sort_right_adj)
# plot_adjs(sort_left_adj, perm_right_adj)
#%%


FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, format="png", dpi=300, **kws)


rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "black",
    "ytick.color": "grey",
    "xtick.color": "grey",
    "axes.labelcolor": "dimgrey",
    "text.color": "black",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}
set_theme(rc_dict=rc_dict, font_scale=1.75)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
adjplot(
    sort_left_adj,
    plot_type="scattermap",
    sizes=(2, 2),
    ax=ax,
    title=r"Left",
    color=palette["Left"],
)
stashfig("left-left-adj")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
adjplot(
    shuffle_right_adj,
    plot_type="scattermap",
    sizes=(2, 2),
    ax=ax,
    title=r"Right (unknown permutation)",
    color=palette["Right"],
)
stashfig("right-right-adj-shuffle")


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
adjplot(
    perm_right_adj,
    plot_type="scattermap",
    sizes=(2, 2),
    ax=ax,
    title=r"Right (after matching)",
    color=palette["Right"],
)
stashfig("right-right-adj-perm")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
adjplot(
    sort_right_adj,
    plot_type="scattermap",
    sizes=(2, 2),
    ax=ax,
    title=r"Right",
    color=palette["Right"],
)
stashfig("right-right-adj")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
adjplot(
    sort_right_adj,
    plot_type="scattermap",
    sizes=(2, 2),
    ax=ax,
    color=palette["Right"],
)
stashfig("right-right-adj-no-title")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
adjplot(
    sort_left_adj,
    plot_type="scattermap",
    sizes=(2, 2),
    ax=ax,
    color=palette["Left"],
)
stashfig("left-left-adj-no-title")


#%%


from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import AdjacencySpectralEmbed, select_dimension
from graspologic.plot import pairplot
from graspologic.utils import (
    augment_diagonal,
    binarize,
    multigraph_lcc_intersection,
    pass_to_ranks,
)


def embed(adj, n_components=40, ptr=False):
    if ptr:
        adj = pass_to_ranks(adj)
    elbow_inds, elbow_vals = select_dimension(augment_diagonal(adj), n_elbows=4)
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    out_latent, in_latent = ase.fit_transform(adj)
    return out_latent, in_latent, ase.singular_values_, elbow_inds


ll_adj = left_adj_t
rr_adj = right_adj_t

n_components = 8
max_n_components = 40
preprocess = "binarize"

if preprocess == "binarize":
    ll_adj = binarize(ll_adj)
    rr_adj = binarize(rr_adj)

left_out_latent, left_in_latent, left_sing_vals, left_elbow_inds = embed(
    ll_adj, n_components=max_n_components
)
right_out_latent, right_in_latent, right_sing_vals, right_elbow_inds = embed(
    rr_adj, n_components=max_n_components
)

#%%

from giskard.plot import simple_scatterplot

scatter_kws = dict(
    figsize=(4, 4),
    palette=palette,
    spines_off=False,
)
labels = np.array(n_pairs * ["Left"])
simple_scatterplot(left_out_latent, labels=labels, **scatter_kws)
stashfig("left_latent")

labels = np.array(n_pairs * ["Right"])
simple_scatterplot(right_out_latent, labels=labels, **scatter_kws)
stashfig("right_latent")


labels = np.array(n_pairs * ["Left"] + n_pairs * ["Right"])
out_latent = np.concatenate((left_out_latent, right_out_latent), axis=0)
simple_scatterplot(out_latent, labels=labels, **scatter_kws)
stashfig("left_right_latent")


from graspologic.align import OrthogonalProcrustes

labels = np.array(n_pairs * ["Left"] + n_pairs * ["Right"])
right_rot_latent = OrthogonalProcrustes().fit_transform(
    right_out_latent, left_out_latent
)
out_latent = np.concatenate((left_out_latent, right_rot_latent), axis=0)
simple_scatterplot(out_latent, labels=labels, **scatter_kws)
stashfig("left_right_aligned")
