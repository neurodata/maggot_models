import numpy as np
from src.pymaid import start_instance
import pymaid


def plot_neurons(meta, key=None, label=None, barplot=False):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization import CLASS_COLOR_DICT, stacked_barplot, set_axes_equal

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
