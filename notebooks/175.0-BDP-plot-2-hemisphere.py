# %% [markdown]
# ##
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
from src.cluster import BinaryCluster
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
)

print(mpl.__version__)

# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

# # plotting settings
# rc_dict = {
#     "axes.spines.right": False,
#     "axes.spines.top": False,
#     "axes.formatter.limits": (-3, 3),
#     "figure.figsize": (6, 3),
#     "figure.dpi": 100,
#     "axes.edgecolor": "lightgrey",
#     "ytick.color": "grey",
#     "xtick.color": "grey",
#     "axes.labelcolor": "dimgrey",
#     "text.color": "dimgrey",
# }
# for key, val in rc_dict.items():
#     mpl.rcParams[key] = val
# context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)

# sns.set_context(context)
sns.set_context("talk")

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


# %% [markdown]
# ## Load and preprocess data
graph_type = "G"
VERSION = "2020-05-26"
master_mg = load_metagraph(graph_type, version=VERSION)
mg = MetaGraph(master_mg.adj, master_mg.meta)
mg = mg.remove_pdiff()
meta = mg.meta.copy()

# remove low degree neurons
degrees = mg.calculate_degrees()
quant_val = np.quantile(degrees["Total edgesum"], 0.05)
idx = meta[degrees["Total edgesum"] > quant_val].index
print(quant_val)
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
print(len(left_mg))
right_idx = meta[meta["right"]].index
right_mg = MetaGraph(mg.adj, mg.meta)
right_mg = right_mg.reindex(right_idx, use_ids=True)
right_mg = right_mg.sort_values(["pair_id"], ascending=False)
right_mg = right_mg.reindex(right_mg.meta.index[: len(left_mg)], use_ids=True)
print(len(right_mg))

assert (right_mg.meta["pair_id"].values == left_mg.meta["pair_id"].values).all()

# %% [markdown]
# ## Pair the unpaired neurons using graph matching

n_pairs = len(right_mg.meta[right_mg.meta["pair_id"] != -1])
left_adj = left_mg.adj
right_adj = right_mg.adj
left_seeds = right_seeds = np.arange(n_pairs)
currtime = time.time()
gm = GraphMatch(n_init=5, init_method="rand", eps=2.0, shuffle_input=False)
gm.fit(left_adj, right_adj, seeds_A=left_seeds, seeds_B=right_seeds)
print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for graph matching")

# %% [markdown]
# ##
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
perm_inds = gm.perm_inds_
true_pairs = n_pairs * ["paired"] + (len(left_adj) - n_pairs) * ["unpaired"]
pal = sns.color_palette("deep", 3)
color_dict = dict(zip(["unpaired", "paired"], pal[1:]))
_, _, top, _ = adjplot(
    left_adj,
    ax=axs[0],
    plot_type="scattermap",
    sizes=(1, 1),
    colors=true_pairs,
    palette=color_dict,
    color=pal[0],
    ticks=True,
)
# axs[0].set_ylim(axs[0].get_ylim()[::-1])
top.set_title("Left")
_, _, top, _ = adjplot(
    right_adj[np.ix_(perm_inds, perm_inds)],
    ax=axs[1],
    plot_type="scattermap",
    sizes=(1, 1),
    colors=true_pairs,
    palette=color_dict,
    color=pal[0],
    ticks=True,
)
top.set_title("Right")
plt.tight_layout()
stashfig("gm-results-adj")

# %% [markdown]
# ##
left_ids = left_mg.meta.index
right_ids = right_mg.meta.index.copy()
right_ids = right_ids[perm_inds]
ids = np.concatenate((left_ids, right_ids))
# %% [markdown]
# ##
new_mg = mg.reindex(ids, use_ids=True).copy()
new_mg.meta.loc[left_ids, "new_pair_id"] = range(len(left_ids))
new_mg.meta.loc[right_ids, "new_pair_id"] = range(len(right_ids))

# %% [markdown]
# ##
pal = sns.color_palette("deep", 5)
_, _, top, _ = adjplot(
    new_mg.adj,
    meta=new_mg.meta,
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    sort_class=["hemisphere"],
    item_order=["merge_class", "new_pair_id"],
    color=pal[4],
    ticks=True,
)
stashfig("bilateral-adj")


# %% [markdown]
# ##


# %% [markdown]
# ##
# sort_mg = MetaGraph(new_mg.adj, new_mg.meta)
# sort_mg = sort_mg.sort_values(["merge_class", "new_pair_id"])
full_adj = new_mg.adj
left_inds = np.arange(len(left_ids))
right_inds = np.arange(len(right_ids)) + len(left_ids)

ll_adj = full_adj[np.ix_(left_inds, left_inds)]
rr_adj = full_adj[np.ix_(right_inds, right_inds)]
lr_adj = full_adj[np.ix_(left_inds, right_inds)]
rl_adj = full_adj[np.ix_(right_inds, left_inds)]
adjs = [ll_adj, rr_adj, lr_adj, rl_adj]

# %% [markdown]
# ##
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_square_verts(x, length):
    return [[x, 0, 0], [x, length, 0], [x, length, length], [x, 0, length], [x, 0, 0]]


def scatter_adj_3d(A, x, scale=1, ax=None, c="grey", zorder=1):
    inds = np.nonzero(A)
    edges = A[inds]
    xs = len(edges) * [x]  # dummy variable for the "pane" x position
    ys = inds[1]  # target
    zs = inds[0]  # source
    ax.scatter(
        xs, ys, zs, s=scale * edges, c=c, zorder=zorder
    )  # note: zorder doesn't work
    return ax


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.force_zorder = True  # key to make this look right

n_verts = len(adjs[0])
n_graphs = len(adjs)
step = n_verts

pal = sns.color_palette("deep", 8)[4:]
for i, temp_adj in enumerate(adjs):
    x = i * step - 5  # try moving the points just in front of the pane
    # note this doesn't quite work til MPL bug is fixed
    # REF: open PR https://github.com/matplotlib/matplotlib/pull/14508
    ax = scatter_adj_3d(
        binarize(temp_adj),
        x,
        ax=ax,
        scale=0.05,  # size of the dots
        c=[pal[i]],
        zorder=n_graphs * 2 - i * 2 + 1,
    )
    vert_list = [get_square_verts(i * step, n_verts)]
    pc = Poly3DCollection(
        vert_list, edgecolors="dimgrey", facecolors="white", linewidths=1, alpha=0.85
    )
    pc.set_zorder(n_graphs * 2 - i * 2)
    ax.add_collection3d(pc)


# x will index the graphs
ax.set_xlim((-10, step * (n_graphs - 1)))
ax.set_ylim((0, n_verts))
ax.set_zlim((0, n_verts))

# set camera position
ax.azim = 135  # -45
ax.elev = 20

# # for testing
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
ylim = ax.get_ylim()
ax.set_ylim(ylim[::-1])
zlim = ax.get_zlim()
ax.set_zlim(zlim[::-1])
# xlim = ax.get_xlim()
# ax.set_xlim(xlim[::-1])
# ax.invert_zaxis()
# ax.invert_yaxis()
# ax.invert_xaxis()

pad = 0.01
label_loc = "bottom"
names = [r"L $\to$ L", r"R $\to$ R", r"L $\to$ R", r"R $\to$ L"]
for i, name in enumerate(names):
    if label_loc == "top":
        ax.text(i * step, n_verts + pad * n_verts, 0 - pad * n_verts, name, ha="center")
    elif label_loc == "bottom":
        ax.text(
            i * step,
            0 - pad * n_verts,
            n_verts + pad * n_verts,
            name,
            ha="center",
            va="top",
        )
ax.axis("off")
stashfig("stacked-adj-hemispheres")


# %%
for adj in adjs:
    adjplot(adj, plot_type="scattermap")

# %%
