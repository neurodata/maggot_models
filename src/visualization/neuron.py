import numpy as np
from src.pymaid import start_instance
import pymaid
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from src.visualization import CLASS_COLOR_DICT, stacked_barplot, set_axes_equal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_neurons(meta, key=None, label=None, barplot=False):

    if label is not None:
        ids = list(meta[meta[key] == label].index.values)
    else:
        ids = list(meta.index.values)
    ids = [int(i) for i in ids]

    new_ids = []
    for i in ids:
        try:
            pymaid.get_neuron(
                i, raise_missing=True, with_connectors=False, with_tags=False
            )
            new_ids.append(i)
        except:
            print(f"Missing neuron {i}, not plotting it.")

    ids = new_ids
    meta = meta.loc[ids]

    fig = plt.figure(figsize=(30, 10))

    gs = plt.GridSpec(2, 3, figure=fig, wspace=0, hspace=0, height_ratios=[0.8, 0.2])

    skeleton_color_dict = dict(
        zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
    )

    ax = fig.add_subplot(gs[0, 0], projection="3d")

    pymaid.plot2d(
        ids,
        color=skeleton_color_dict,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=True,
    )
    ax.azim = -90
    ax.elev = 0
    ax.dist = 5
    set_axes_equal(ax)

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    pymaid.plot2d(
        ids,
        color=skeleton_color_dict,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=True,
    )
    ax.azim = 0
    ax.elev = 0
    ax.dist = 5
    set_axes_equal(ax)

    ax = fig.add_subplot(gs[0, 2], projection="3d")
    pymaid.plot2d(
        ids,
        color=skeleton_color_dict,
        ax=ax,
        connectors=False,
        method="3d",
        autoscale=True,
    )
    ax.azim = -90
    ax.elev = 90
    ax.dist = 5
    set_axes_equal(ax)

    if barplot:
        ax = fig.add_subplot(gs[1, :])
        temp_meta = meta[meta[key] == label]
        cat = temp_meta[key + "_side"].values
        subcat = temp_meta["merge_class"].values
        stacked_barplot(
            cat,
            subcat,
            ax=ax,
            color_dict=CLASS_COLOR_DICT,
            category_order=np.unique(cat),
        )
        ax.get_legend().remove()

    # fig.suptitle(label)
    return fig, ax


views = ["front", "side", "top"]
view_params = [
    dict(azim=-90, elev=0, dist=5),
    dict(azim=0, elev=0, dist=5),
    dict(azim=-45, elev=90, dist=5),
]
view_dict = dict(zip(views, view_params))
volume_names = ["PS_Neuropil_manual"]


def set_view_params(ax, azim=-90, elev=0, dist=5):
    ax.azim = azim
    ax.elev = elev
    ax.dist = dist
    set_axes_equal(ax)


def plot_volumes(volumes, ax):
    pymaid.plot2d(volumes, ax=ax, method="3d", autoscale=False)
    for c in ax.collections:
        if isinstance(c, Poly3DCollection):
            c.set_alpha(0.02)


import pandas as pd


def plot_3view(
    data,
    axs,
    palette=None,
    connectors=False,
    connectors_only=False,
    label_by=None,
    alpha=1,
    s=1,
    row_title=None,
    **kws,
):
    volumes = [pymaid.get_volume(v) for v in volume_names]
    for i, view in enumerate(views):
        ax = axs[i]
        if label_by is None:
            pymaid.plot2d(
                data,
                color=palette,
                ax=ax,
                method="3d",
                connectors=connectors,
                connectors_only=connectors_only,
                **kws,
            )
        else:
            uni_labels = np.unique(data[label_by])
            for ul in uni_labels:
                temp_data = data[data[label_by] == ul]
                color = palette[ul]
                scatter_kws = dict(s=s, alpha=alpha, color=color)
                pymaid.plot2d(
                    temp_data[["x", "y", "z"]],
                    ax=ax,
                    method="3d",
                    connectors=connectors,
                    connectors_only=connectors_only,
                    scatter_kws=scatter_kws,
                    **kws,
                )
        set_view_params(ax, **view_dict[view])
        plot_volumes(volumes, ax)
        if row_title is not None:
            ax = axs[0]
            ax.text2D(
                x=0,
                y=0.5,
                s=row_title,
                ha="right",
                va="center",
                color="grey",
                rotation=90,
                transform=ax.transAxes,
            )


# mins = data.min()[["x", "y", "z"]]
# maxs = data.max()[["x", "y", "z"]]
# ax.set_xlim(mins[0], maxs[0])
# ax.set_xlim(mins[1], maxs[1])
# ax.set_xlim(mins[2], maxs[2])
