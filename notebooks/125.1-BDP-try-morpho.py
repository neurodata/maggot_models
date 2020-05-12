# %% [markdown]
# ##
import os
import matplotlib as mpl

# mpl.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

import pymaid
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    adjplot,
    barplot_text,
    gridmap,
    matrixplot,
    set_axes_equal,
    stacked_barplot,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


mg = load_metagraph("G")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

start_instance()

skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)


def get_connectors(nl):
    connectors = pymaid.get_connectors(nl)
    connectors.set_index("connector_id", inplace=True)
    connectors.drop(
        [
            "confidence",
            "creation_time",
            "edition_time",
            "tags",
            "creator",
            "editor",
            "type",
        ],
        inplace=True,
        axis=1,
    )
    details = pymaid.get_connector_details(connectors.index.values)
    details.set_index("connector_id", inplace=True)
    connectors = pd.concat((connectors, details), ignore_index=False, axis=1)
    connectors.reset_index(inplace=True)
    return connectors


def set_view_params(ax, azim=-90, elev=0, dist=5):
    ax.azim = azim
    ax.elev = elev
    ax.dist = dist
    set_axes_equal(ax)


# params
label = "KC"
volume_names = ["PS_Neuropil_manual"]


class_ids = meta[meta["class1"] == label].index.values
ids = [int(i) for i in class_ids]
nl = pymaid.get_neurons(class_ids)
print(f"Plotting {len(nl)} neurons for label {label}.")
connectors = get_connectors(nl)
outputs = connectors[connectors["presynaptic_to"].isin(class_ids)]
# I hope this is a valid assumption?
inputs = connectors[~connectors["presynaptic_to"].isin(class_ids)]

# %% [markdown]
# ##
sns.set_context("talk", font_scale=1.5)
fig = plt.figure(figsize=(30, 30))
fig.suptitle(label, y=0.93)
gs = plt.GridSpec(3, 3, figure=fig, wspace=0, hspace=0)

views = ["front", "side", "top"]
view_params = [
    dict(azim=-90, elev=0, dist=5),
    dict(azim=0, elev=0, dist=5),
    dict(azim=-90, elev=90, dist=5),
]
view_dict = dict(zip(views, view_params))

volumes = [pymaid.get_volume(v) for v in volume_names]


def plot_volumes(ax):
    pymaid.plot2d(volumes, ax=ax, method="3d")
    for c in ax.collections:
        if isinstance(c, Poly3DCollection):
            c.set_alpha(0.03)


def add_subplot(row, col):
    ax = fig.add_subplot(gs[row, col], projection="3d")
    axs[row, col] = ax
    return ax


axs = np.empty((3, 3), dtype="O")

# plot neuron skeletons
row = 0
for i, view in enumerate(views):
    ax = add_subplot(row, i)
    # pymaid.plot2d(ids, color=skeleton_color_dict, ax=ax, connectors=False, method="3d")
    plot_volumes(ax)
    set_view_params(ax, **view_dict[view])

axs[0, 0].text2D(
    x=0.1,
    y=0.8,
    s="Skeletons",
    ha="center",
    va="bottom",
    color="grey",
    rotation=90,
    transform=fig.transFigure,
)

# plot inputs
row = 1
for i, view in enumerate(views):
    ax = add_subplot(row, i)
    connector_locs = inputs[["x", "y", "z"]].values
    pymaid.plot2d(connector_locs, color="orchid", ax=ax, method="3d")
    plot_volumes(ax)
    set_view_params(ax, **view_dict[view])

axs[1, 0].text2D(
    x=0.1,
    y=0.49,
    s="Inputs",
    ha="center",
    va="bottom",
    color="grey",
    rotation=90,
    transform=fig.transFigure,
)

# plot outputs
row = 2
for i, view in enumerate(views):
    ax = add_subplot(row, i)
    connector_locs = outputs[["x", "y", "z"]].values
    pymaid.plot2d(
        connector_locs, color="orchid", ax=ax, method="3d", cn_mesh_colors=True
    )
    plot_volumes(ax)
    set_view_params(ax, **view_dict[view])

axs[2, 0].text2D(
    x=0.1,
    y=0.16,
    s="Outputs",
    ha="center",
    va="bottom",
    color="grey",
    rotation=90,
    transform=fig.transFigure,
)

plt.tight_layout()
# plt.show()
stashfig(f"try-neuro-morpho-label={label}")
# %% [markdown]
# # ##


data = inputs
plot_vars = np.array(["x", "y", "z"])

scale = 0.2

data_mat = data[plot_vars].values

mins = []
maxs = []
for i in range(data_mat.shape[1]):
    dmin = data_mat[:, i].min()
    dmax = data_mat[:, i].max()
    mins.append(dmin)
    maxs.append(dmax)
mins = np.array(mins)
maxs = np.array(maxs)
ranges = maxs - mins
mins = mins - ranges * scale
maxs = maxs + ranges * scale

kernel = gaussian_kde(data_mat.T)

# print("making meshgrid")
# X, Y, Z = np.mgrid[
#     mins[0] : maxs[0] : 100j, mins[1] : maxs[1] : 100j, mins[2] : maxs[2] : 100j
# ]
# positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
# print("evaluating meshgrid")
# Z = np.reshape(kernel(positions).T, X.shape)

# %% [markdown]
# ##
from src.visualization import remove_axis, remove_spines


def plot_connectors(data, x, y, ax):

    # ax.axis("off")
    # remove_axis(ax)

    sns.scatterplot(
        data=data,
        y=plot_vars[y],
        x=plot_vars[x],
        s=3,
        alpha=0.05,
        ax=ax,
        linewidth=0,
        color="black",
    )
    unused = np.setdiff1d([0, 1, 2], [x, y])[0]
    projection = Z.sum(axis=unused)
    if x > y:
        projection = projection.T
    ax.imshow(
        np.rot90(projection),  # integrate out the unused dim
        cmap=plt.cm.Reds,
        extent=[mins[x], maxs[x], mins[y], maxs[y]],
        vmin=0,
    )
    # print(Z.shape)
    ax.set_xlim([mins[x], maxs[x]])
    ax.set_ylim([maxs[y], mins[y]])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    remove_spines(ax)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Input KDEs - {label}", y=1)
axes_names = [
    r"$\leftarrow$ L $\quad$ R $\rightarrow$",
    r"$\leftarrow$ D $\quad$ V $\rightarrow$",
    r"$\leftarrow$ A $\quad$ P $\rightarrow$",
]

ax = axs[0]
plot_connectors(data, 0, 1, ax)
ax.set_xlabel(axes_names[0])
ax.set_ylabel(axes_names[1])

ax = axs[1]
plot_connectors(data, 2, 1, ax)
ax.set_xlabel(axes_names[2])
ax.set_ylabel(axes_names[1])

ax = axs[2]
plot_connectors(data, 0, 2, ax)
ax.set_xlabel(axes_names[0])
ax.set_ylabel(axes_names[2])
ax.invert_yaxis()

plt.tight_layout()
stashfig(f"morpho-kde-label={label}")

# %% [markdown]
# ##
from scipy.integrate import tplquad


def evaluate(x, y, z):
    return kernel((x, y, z))


tplquad(evaluate, mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
