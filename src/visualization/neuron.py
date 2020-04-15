import numpy as np


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    # REF: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


from src.pymaid import start_instance
import pymaid


def plot_neurons(meta, key, label):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization import CLASS_COLOR_DICT, stacked_barplot

    ids = list(meta[meta[key] == label].index.values)
    ids = [int(i) for i in ids]
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
    ax.dist = 6
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
    ax.dist = 6
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
    ax.dist = 6
    set_axes_equal(ax)

    ax = fig.add_subplot(gs[1, :])
    temp_meta = meta[meta[key] == label]
    cat = temp_meta[key + "_side"].values
    subcat = temp_meta["merge_class"].values
    stacked_barplot(
        cat, subcat, ax=ax, color_dict=CLASS_COLOR_DICT, category_order=np.unique(cat)
    )
    ax.get_legend().remove()

    fig.suptitle(label)
    return fig, ax
