import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from .visualize import gridmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


def sort_meta(meta, sort_class, sort_node=[]):
    meta = meta.copy()
    total_sort_by = []
    for sc in sort_class:
        class_size = meta.groupby(sc).size()
        # negative so we can sort alphabetical still in one line
        meta[f"{sc}_size"] = -meta[sc].map(class_size)
        total_sort_by.append(f"{sc}_size")
        total_sort_by.append(sc)
    total_sort_by += sort_node
    meta["idx"] = range(len(meta))
    meta.sort_values(total_sort_by, inplace=True)
    perm_inds = meta.idx.values
    return perm_inds


def _get_tick_info(sort_meta, sort_class):
    """ Assumes meta is already sorted
    """
    if sort_meta is not None and sort_class is not None:
        # get locations
        sort_meta["idx"] = range(len(sort_meta))
        # for the gridlines
        first_df = sort_meta.groupby(sort_class, sort=False).first()
        first_inds = list(first_df["idx"].values)[1:]  # skip since we have spines
        # for the tic locs
        middle_df = sort_meta.groupby(sort_class, sort=False).mean()
        middle_inds = list(middle_df["idx"].values)
        middle_labels = list(middle_df.index)
        return first_inds, middle_inds, middle_labels
    else:
        return None, None, None


def _draw_seperators(
    ax,
    row_meta=None,
    col_meta=None,
    row_sort_class=None,
    col_sort_class=None,
    plot_type="heatmap",
    gridline_kws=None,
    use_colors=False,
    use_ticks=True,
    tick_fontsize=10,
    minor_ticking=False,
):

    if gridline_kws is None:
        gridline_kws = dict(color="grey", linestyle="--", alpha=0.7, linewidth=1)

    if plot_type == "heatmap":
        boost = 0
    elif plot_type == "scattermap":
        boost = 0.5

    axes_info = {}
    if col_meta is not None:
        axes_info["x"] = {
            "meta": col_meta,
            "sort_class": col_sort_class,
            "axis": ax.xaxis,
        }
    if row_meta is not None:
        axes_info["y"] = {
            "meta": row_meta,
            "sort_class": row_sort_class,
            "axis": ax.yaxis,
        }

    for axis_name, axis_info in axes_info.items():
        axis = axis_info["axis"]
        first_inds, middle_inds, middle_labels = _get_tick_info(
            axis_info["meta"], axis_info["sort_class"]
        )

        for t in first_inds:
            if axis_name == "x":
                ax.axvline(t - boost, **gridline_kws)
            else:
                ax.axhline(t - boost, **gridline_kws)

        if use_colors:  # TODO experimental!
            raise NotImplementedError()
            axis.set_ticks([])
            axis.set_ticks([])
            for sc in axis_info[
                "sort_class"
            ]:  # TODO this will break for more than one category
                divider = make_axes_locatable(ax)
                left_cax = divider.append_axes("left", size="3%", pad=0, sharey=ax)
                top_cax = divider.append_axes("top", size="3%", pad=0, sharex=ax)
                # left_cax.set_ylim(ax.get_ylim())ƒ
                classes = sort_meta[sc].values
                class_colors = np.vectorize(CLASS_COLOR_DICT.get)(
                    classes
                )  # TODO make not specific

                from matplotlib.colors import ListedColormap

                # make colormap
                uni_classes = np.unique(classes)
                class_map = dict(zip(uni_classes, range(len(uni_classes))))
                color_list = []
                for u in uni_classes:
                    color_list.append(CLASS_COLOR_DICT[u])
                lc = ListedColormap(color_list)
                classes = np.vectorize(class_map.get)(classes)
                classes = classes.reshape(len(classes), 1)
                sns.heatmap(
                    classes,
                    cmap=lc,
                    cbar=False,
                    yticklabels=False,
                    xticklabels=False,
                    ax=left_cax,
                    square=False,
                )
                classes = classes.T  # reshape(len(classes), 1)
                sns.heatmap(
                    classes,
                    cmap=lc,
                    cbar=False,
                    yticklabels=False,
                    xticklabels=False,
                    ax=top_cax,
                    square=False,
                )

        if use_ticks:
            if use_ticks and use_colors:
                top_tick_ax = top_cax
                left_tick_ax = left_cax
                top_tick_ax.set_yticks([])
                left_tick_ax.set_xticks([])
            else:
                top_tick_ax = left_tick_ax = ax

            # add tick labels and locs
            if axis_name == "x":
                top_tick_ax.set_xticks(middle_inds)
                if minor_ticking:
                    top_tick_ax.set_xticklabels(middle_labels[0::2])
                    top_tick_ax.set_xticklabels(middle_labels[1::2], minor=True)
                else:
                    top_tick_ax.set_xticklabels(
                        middle_labels, rotation=45, ha="left", va="bottom"
                    )
                top_tick_ax.xaxis.tick_top()
                # for tick in top_tick_ax.get_xticklabels():
                #     tick.set_rotation(45)
                #     tick.set_fontsize(tick_fontsize)
                # for tick in top_tick_ax.get_xticklabels(minor=True):
                #     tick.set_rotation(45)
                #     tick.set_fontsize(tick_fontsize)
            elif axis_name == "y":
                left_tick_ax.set_yticks(middle_inds)
                if minor_ticking:
                    left_tick_ax.set_yticklabels(middle_labels[0::2])
                    left_tick_ax.set_yticklabels(middle_labels[1::2], minor=True)
                else:
                    left_tick_ax.set_yticklabels(middle_labels)
                for tick in left_tick_ax.get_yticklabels():
                    tick.set_fontsize(tick_fontsize)
                for tick in left_tick_ax.get_yticklabels(minor=True):
                    tick.set_fontsize(tick_fontsize)

            # modify the padding / offset every other tick
            # for i, axis in enumerate([top_tick_ax.xaxis, left_tick_ax.yaxis]):
            #     axis.set_major_locator(plt.FixedLocator(middle_inds[0::2]))
            #     axis.set_minor_locator(plt.FixedLocator(middle_inds[1::2]))
            #     axis.set_minor_formatter(plt.FormatStrFormatter("%s"))

            # top_tick_ax.tick_params(
            #     which="minor", pad=tick_pad[i] + base_tick_pad, length=5
            # )
            # top_tick_ax.tick_params(which="major", pad=base_tick_pad, length=5)
            # left_tick_ax.tick_params(
            #     which="minor", pad=tick_pad[i] + base_tick_pad, length=5
            # )
            # left_tick_ax.tick_params(which="major", pad=base_tick_pad, length=5)

            # set tick size and rotation

            if use_colors and use_ticks:
                shax = ax.get_shared_x_axes()
                shay = ax.get_shared_y_axes()
                shax.remove(ax)
                shay.remove(ax)
                xticker = mpl.axis.Ticker()
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.major = xticker
                    axis.minor = xticker
                    loc = mpl.ticker.NullLocator()
                    fmt = mpl.ticker.NullFormatter()
                    axis.set_major_locator(loc)
                    axis.set_major_formatter(fmt)
                    axis.set_minor_locator(loc)
                    axis.set_minor_formatter(fmt)


def matrixplot(
    data,
    ax=None,
    plot_type="heatmap",
    row_meta=None,
    col_meta=None,
    row_sort_class=None,
    col_sort_class=None,
    border=True,
    use_ticks=True,
    use_colors=False,
    minor_ticking=False,
    cmap="RdBu_r",
    sizes=(10, 40),
    square=False,
):
    tick_fontsize = 10

    spinestyle_kws = dict(linestyle="-", linewidth=1, alpha=0.7, color="black")

    tick_pad = [0, 0]
    base_tick_pad = 5

    # sort the data and metadata
    if row_meta is not None and row_sort_class is not None:
        row_perm_inds = sort_meta(row_meta, row_sort_class)
    else:
        row_perm_inds = np.arange(data.shape[0])
    if col_meta is not None and col_sort_class is not None:
        col_perm_inds = sort_meta(col_meta, col_sort_class)
    else:
        col_perm_inds = np.arange(data.shape[1])
    data = data[np.ix_(row_perm_inds, col_perm_inds)]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    # do the actual plotting!
    if plot_type == "heatmap":
        sns.heatmap(data, cmap=cmap, ax=ax, vmin=0, center=0, cbar=False)
    elif plot_type == "scattermap":
        gridmap(data, ax=ax, sizes=sizes, border=False)

    if square:
        ax.axis("square")
    ax.set_ylim(data.shape[0], 0)
    ax.set_xlim(0, data.shape[1])

    # spines
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(spinestyle_kws["color"])
            spine.set_linewidth(spinestyle_kws["linewidth"])
            spine.set_linestyle(spinestyle_kws["linestyle"])
            spine.set_alpha(spinestyle_kws["alpha"])
