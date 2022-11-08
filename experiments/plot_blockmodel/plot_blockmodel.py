#%%
import datetime
import time
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


from graspologic.utils import remove_loops

from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.data import load_maggot_graph
from src.io import savefig
from src.utils import get_blockmodel_df
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    draw_networkx_nice,
    remove_shared_ax,
    remove_spines,
    set_theme,
)
from src.data import load_palette
from src.visualization import HUE_KEY

set_theme()

t0 = time.time()
palette = load_palette()

print("HUE_KEY = ", HUE_KEY)

out_path = Path("maggot_models/experiments/plot_blockmodel/")


FORMAT = "png"


def stashfig(name, format=FORMAT, **kws):
    savefig(
        name, pathname=out_path / "figs", format=format, dpi=300, save_on=True, **kws
    )
    savefig(name, pathname=out_path / "figs", format="pdf", save_on=True, **kws)


def estimate_spring_rank_P(A, ranks, beta):
    H = ranks[:, None] - ranks[None, :] - 1
    H = np.multiply(H, H)
    H *= 0.5
    P = np.exp(-beta * H)
    P *= np.mean(A) / np.mean(P)  # TODO I might be off by a constant here
    return P


def signal_flow(A):
    """Implementation of the signal flow metric from Varshney et al 2011

    Parameters
    ----------
    A : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    A = A.copy()
    A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z


def rank_signal_flow(A):
    sf = signal_flow(A)
    perm_inds = np.argsort(-sf)
    return perm_inds


def calculate_p_upper(A):
    A = remove_loops(A)
    n = len(A)
    triu_inds = np.triu_indices(n, k=1)
    upper_triu_sum = A[triu_inds].sum()
    total_sum = A.sum()
    upper_triu_p = upper_triu_sum / total_sum
    return upper_triu_p


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


mg = load_maggot_graph()


#%%
# CLUSTER_KEY = "co_cluster_n_clusters=85"
# CLUSTER_KEY = f"dc_labels_level={4}"
n_components = 10
min_split = 32
i = 4
CLUSTER_KEY = f"dc_level_{i}_n_components={n_components}_min_split={min_split}"
mg = mg[~mg.nodes[CLUSTER_KEY].isna()]
mg.nodes[CLUSTER_KEY] = mg.nodes[CLUSTER_KEY].astype("Int64")
labels = mg.nodes[CLUSTER_KEY]
meta = mg.nodes

bar_ratio = 0.05
alpha = 0.05
width = 0.9
use_weights = True
use_counts = False
adj = mg.sum.adj
blockmodel_df = get_blockmodel_df(
    adj, labels, return_counts=use_counts, use_weights=use_weights
)
perm_inds = rank_signal_flow(blockmodel_df.values)
blockmodel_df = blockmodel_df.reindex(
    index=blockmodel_df.index[perm_inds], columns=blockmodel_df.index[perm_inds]
)

order_map = dict(
    zip(blockmodel_df.index, np.arange(len(blockmodel_df)) / len(blockmodel_df))
)

heatmap_kws = dict(square=True, cmap="Reds", cbar_kws=dict(shrink=0.7))
data = blockmodel_df.values
data = np.sqrt(data)
blockmodel_df = pd.DataFrame(
    data=data, index=blockmodel_df.index, columns=blockmodel_df.columns
)
x = data.ravel()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(data=x[x != 0], ax=ax, bins=200)
data[data < 0.07] = 0

#%%

data = blockmodel_df.values
uni_labels = blockmodel_df.index.values

fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

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
    temp_meta = meta[meta[CLUSTER_KEY] == label]
    plot_bar(temp_meta, mids[i], left_ax, orientation="horizontal", width=width)
    plot_bar(temp_meta, mids[i], top_ax, orientation="vertical", width=width)

ax.yaxis.set_visible(True)
# ax.yaxis.tick_right()
# ax.yaxis.set_ticks(np.arange(len(data)) + 0.5)
# ax.yaxis.set_ticklabels(uni_labels, fontsize=10, color="dimgrey", va="center")
# ax.yaxis.set_tick_params(rotation=0, color="dimgrey")

left_ax.invert_xaxis()

if len(uni_labels) <= 10:
    pal = sns.color_palette("tab10")
elif len(uni_labels) <= 20:
    pal = sns.color_palette("tab20")
else:
    pal = cc.glasbey_light
color_map = dict(zip(uni_labels, pal))
ticklabels = axs[0].get_yticklabels()
for t in ticklabels:
    text = int(t.get_text())
    t.set_color(color_map[text])


remove_diag = True

# convert the adjacency and a partition to a minigraph based on SBM probs
# prob_df = blockmodel_df
# if remove_diag:
#     adj = prob_df.values
#     adj -= np.diag(np.diag(adj))
#     prob_df = pd.DataFrame(data=adj, index=prob_df.index, columns=prob_df.columns)
prob_df = blockmodel_df
prob_df = prob_df.sort_index(axis=0).sort_index(axis=1)
g = nx.from_pandas_adjacency(prob_df, create_using=nx.DiGraph)
uni_labels, counts = np.unique(labels, return_counts=True)


# add size attribute base on number of vertices
size_map = dict(zip(uni_labels, counts))
nx.set_node_attributes(g, size_map, name="Size")

# add spring layout properties
pos = nx.kamada_kawai_layout(g, weight=None)
spring_x = {}
spring_y = {}
for key, val in pos.items():
    spring_x[key] = val[0]
    spring_y[key] = val[1]


def normalize_pos(pos_dict):
    min_val = min(pos_dict.values())
    for key, val in pos_dict.items():
        pos_dict[key] = val - min_val

    max_val = max(pos_dict.values())
    for key, val in pos_dict.items():
        pos_dict[key] = val / max_val
    return pos_dict


spring_x = normalize_pos(spring_x)
spring_y = normalize_pos(spring_y)


nx.set_node_attributes(g, spring_x, name="Spring-x")
nx.set_node_attributes(g, spring_y, name="Spring-y")

nx.set_node_attributes(g, order_map, name="Order")

nx.set_node_attributes(g, color_map, name="Color")

ax = axs[1]

x_pos_key = "Spring-x"
y_pos_key = "Spring-y"
x_pos = nx.get_node_attributes(g, x_pos_key)
y_pos = nx.get_node_attributes(g, y_pos_key)

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
    y_boost=0,
    draw_labels=False,
    draw_nodes=False,
    size_scale=5,
)


for node, data in g.nodes(data=True):
    x = data[x_pos_key]
    y = data[y_pos_key]
    radius = np.sqrt(data["Size"]) / 500
    label = node
    sub_meta = meta[meta[CLUSTER_KEY] == label]
    sizes = sub_meta.groupby(HUE_KEY).size()
    sizes /= sizes.sum()
    colors = np.vectorize(palette.get)(sizes.index)
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        radius=radius,
        center=(x, y),
        normalize=True,
    )
    for wedge in wedges:
        wedge.set_zorder(100000)

ax.set(xlim=(-0.05, 1.05), ylim=(1.05, -0.05), xlabel="", ylabel="")
ax.set_facecolor("white")
stashfig(f"sbm-plot-cluster_label={CLUSTER_KEY}-x={x_pos_key}-y={y_pos_key}")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
