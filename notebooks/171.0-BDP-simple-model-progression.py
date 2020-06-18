# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import tplquad
from scipy.special import comb
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import pairwise_distances

import pymaid
from graspy.utils import pass_to_ranks
from hyppo.ksample import KSample
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.hierarchy import signal_flow
from src.io import readcsv, savecsv, savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    get_mid_map,
    gridmap,
    matrixplot,
    remove_axis,
    remove_shared_ax,
    remove_spines,
    set_axes_equal,
    set_style,
    stacked_barplot,
)

set_style()

np.random.seed(8888)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="png", dpi=200, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# load data
mg = load_metagraph("G")
mg = mg.make_lcc()
meta = mg.meta

# %% [markdown]
# ##
log_scale = False
distplot_kws = dict(kde=False)
ylabel = "Out degree (weighted)"
xlabel = "In degree (weighted)"
degrees = mg.calculate_degrees()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

median_degrees = np.median(degrees, axis=0)
median_in_degree = median_degrees[0]
median_out_degree = median_degrees[1]

n_bins_log = 50
n_bins_linear = 300
if log_scale:
    degrees[degrees == 0] += 0.5
    ax.set_yscale("log")
    ax.set_xscale("log")
    bins = np.geomspace(degrees.min().min(), degrees.max().max(), n_bins_log)
else:
    start = -1 / (n_bins_linear - 1) * degrees.max().max()
    bins = np.linspace(start, degrees.max().max(), n_bins_linear)

sns.scatterplot(
    data=degrees, x="In edgesum", y="Out edgesum", s=5, alpha=0.2, linewidth=0, ax=ax
)
ax.scatter(
    median_in_degree, median_out_degree, s=200, marker="+", color="black", linewidth=2
)

ax.set_ylabel(ylabel)
ax.set_xlabel(xlabel)


divider = make_axes_locatable(ax)
top_ax = divider.append_axes("top", size="20%", sharex=ax)
sns.distplot(degrees["In edgesum"], ax=top_ax, bins=bins, **distplot_kws)
top_ax.xaxis.set_visible(False)
top_ax.yaxis.set_visible(False)

right_ax = divider.append_axes("right", size="20%", sharey=ax)
sns.distplot(
    degrees["Out edgesum"], ax=right_ax, vertical=True, bins=bins, **distplot_kws
)
right_ax.yaxis.set_visible(False)
right_ax.xaxis.set_visible(False)
ax.axis("square")


pearsons = np.corrcoef(degrees["In edgesum"], degrees["Out edgesum"])[0, 1]
spearmans, _ = spearmanr(degrees["In edgesum"], degrees["Out edgesum"])
ax.text(
    0.7,
    0.05,
    f"Pearson's: {pearsons:0.3f}\nSpearman's: {spearmans:0.3f}",
    transform=ax.transAxes,
    fontsize=12,
)

if not log_scale:
    ax.set_xlim((-20, 500))
    ax.set_ylim((-20, 500))

stashfig(f"neuron-weighted-dergee-log_scale={log_scale}")

no_in_degree_ids = degrees[degrees["In edgesum"] < 1].index
no_in_degree = meta.loc[no_in_degree_ids]
stashcsv(no_in_degree, "no_in_degree")
no_out_degree_ids = degrees[degrees["Out edgesum"] < 1].index
no_out_degree = meta.loc[no_out_degree_ids]
stashcsv(no_out_degree, "no_out_degree")


# %%
adj = mg.adj
inds = np.nonzero(adj)
weights = adj.ravel().copy()  # [inds]
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
bins = np.linspace(-1, weights.max(), int(weights.max() + 2)) + 0.01
sns.distplot(
    weights, ax=ax, hist_kws=dict(log=True), kde=False, norm_hist=True, bins=bins
)
ax.set_ylabel("Density")
ax.set_xlabel("Edge weight (# synapses)")
stashfig(f"edge-weights-log-y")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
bins = np.geomspace(0.1, weights.max(), 50)
weights[weights == 0] = 0.1
sns.distplot(
    weights, ax=ax, hist_kws=dict(log=True), kde=False, norm_hist=True, bins=bins
)
ax.set_xscale("log")
ax.set_xlim((0.09, ax.get_xlim()[1]))
# ax.set_xticks([0.1, ax.get_xticks()[1:]])
ax.set_ylabel("Density")
ax.set_xlabel("Edge weight (# synapses)")
stashfig(f"edge-weights-log-y-x")


# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# # ax.set_xscale("log")
# sns.distplot(np.log10(weights), ax=ax, kde=False)
# ax.set_ylabel("Density")
# ax.set_xlabel("Edge weight (# synapses)")
# ax.axvline(median, c="red", linestyle="--")
# stashfig(f"edge-weights-log-x")
# median = np.median(weights)
# ax.axvline(median, c="red", linestyle="--")
# ax.set_xscale("log")
# ax.set_yscale("log")

# %% [markdown]
# ## For the two hemispheres plot edge density and edge weight distribution


def binarize(x):
    new_x = x.copy()
    new_x[new_x != 0] = 1
    return new_x


meta = mg.meta
meta["inds"] = range(len(meta))
left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]

ll_adj = adj[np.ix_(left_inds, left_inds)]
rr_adj = adj[np.ix_(right_inds, right_inds)]
lr_adj = adj[np.ix_(left_inds, right_inds)]
rl_adj = adj[np.ix_(right_inds, left_inds)]

densities = np.zeros((2, 2))
densities[0, 0] = binarize(ll_adj).mean()
densities[0, 1] = binarize(lr_adj).mean()
densities[1, 0] = binarize(rl_adj).mean()
densities[1, 1] = binarize(rr_adj).mean()
density_df = pd.DataFrame(
    data=densities, index=["Left", "Right"], columns=["Left", "Right"]
)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

sns.heatmap(density_df, annot=True, cmap="Reds", vmin=0, square=True, cbar=False, ax=ax)
plt.setp(ax.get_yticklabels(), va="center")
ax.set_title("Connection probability")
stashfig("hemisphere-connection-prob")
# %% [markdown]
# ##
fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
xlim = (0, adj.max())

for i, source in enumerate(["left", "right"]):
    for j, target in enumerate(["left", "right"]):
        ax = axs[i, j]
        source_inds = meta[meta[source]]["inds"]
        target_inds = meta[meta[target]]["inds"]
        incidence = adj[np.ix_(source_inds, target_inds)]
        inds = np.nonzero(incidence)
        weights = incidence[inds]
        sns.distplot(weights, ax=ax, kde=False, hist_kws=dict(log=True), norm_hist=True)
        ax.set_xlim(xlim)
        ax.text(
            0.7, 0.7, f"{densities[i, j]:.3g}", transform=ax.transAxes, color="black"
        )


# TODO make this "only titles"
axs[0, 0].set_title(r"Left $\to$ left", color="black")
axs[0, 1].set_title(r"Left $\to$ right", color="black")
axs[1, 0].set_title(r"Right $\to$ left", color="black")
axs[1, 1].set_title(r"Right $\to$ right", color="black")
fig.text(0.5, 0.0, "Edge weight (# synapses)", ha="center", va="top")
# TODO add numbers (ticks) to each x axis
# TODO add density numbers to each plot
plt.tight_layout()
stashfig("hemisphere-edge-weights")

# %% [markdown]
# ## Parallel coords

df = []
for side in ["left", "right"]:
    sub_inds = meta[meta[side]]["inds"]
    for degree_type in ["source", "target"]:
        if degree_type == "source":
            degrees = adj[sub_inds, :].sum(axis=0)
        elif degree_type == "target":
            degrees = adj[:, sub_inds].sum(axis=1)
        degree_series = pd.Series(
            data=degrees, index=meta.index, name=f"{degree_type}-{side}"
        )
        df.append(degree_series)

degree_df = pd.DataFrame(df).T
degree_df["side"] = meta["hemisphere"]

# %% [markdown]
# ##
def plot_degree_parallel(degree_df):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    pal = sns.color_palette("deep", 3)
    pd.plotting.parallel_coordinates(
        degree_df, "side", ax=ax, color=pal, linewidth=0.05
    )
    ax.set_yscale("log")
    names = ["From left", "To left", "From right", "To right"]
    ax.set_xticklabels(names)
    ax.get_legend().remove()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    lines = ax.get_legend().get_lines()
    [l.set_alpha(1) for l in lines]
    [l.set_linewidth(1) for l in lines]
    ax.set_ylabel("Degree (# synapses)")
    return fig, ax


plot_degree_parallel(degree_df)
stashfig("degree-par-coords")

# %% [markdown]
# ##

from src.pymaid import start_instance

start_instance()

contra_ids = np.array(pymaid.get_skids_by_annotation("mw brain contralateral"))
contra_ids = contra_ids[np.isin(contra_ids, degree_df.index)]
fig, ax = plot_degree_parallel(degree_df.loc[contra_ids])
ax.set_title(f"Contralateral neurons ({len(contra_ids)})")
stashfig("degree-par-coords-contra")

ipsi_ids = np.array(pymaid.get_skids_by_annotation("mw brain ipsilateral"))
ipsi_ids = ipsi_ids[np.isin(ipsi_ids, degree_df.index)]
fig, ax = plot_degree_parallel(degree_df.loc[ipsi_ids])
ax.set_title(f"Ipsilateral neurons ({len(ipsi_ids)})")
stashfig("degree-par-coords-ipsi")
