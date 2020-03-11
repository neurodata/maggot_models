import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from .visualize import gridmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


def sort_meta(meta, sort_class, sort_item=None, class_order="size"):
    if sort_item is None:
        sort_item = []
    meta = meta.copy()
    total_sort_by = []
    for sc in sort_class:
        if class_order == "size":
            class_size = meta.groupby(sc).size()
            # negative so we can sort alphabetical still in one line
            meta[f"{sc}_size"] = -meta[sc].map(class_size)
            total_sort_by.append(f"{sc}_size")
        else:
            class_value = meta.groupby(sc)[class_order].first()
            meta[f"{sc}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_order")
        total_sort_by.append(sc)
    total_sort_by += sort_item
    meta["sort_idx"] = range(len(meta))
    meta.sort_values(total_sort_by, inplace=True)
    perm_inds = meta["sort_idx"].values
    return perm_inds, meta


def _get_tick_info(sort_meta, sort_class):
    """ Assumes meta is already sorted
    """
    if sort_meta is not None and sort_class is not None:
        # get locations
        sort_meta["sort_idx"] = range(len(sort_meta))
        # for the gridlines
        first_df = sort_meta.groupby(sort_class, sort=False).first()
        first_inds = list(first_df["sort_idx"].values)[1:]  # skip since we have spines
        # for the tic locs
        middle_df = sort_meta.groupby(sort_class, sort=False).mean()
        middle_inds = np.array(middle_df["sort_idx"].values) + 0.5
        middle_labels = list(middle_df.index)
        return first_inds, middle_inds, middle_labels
    else:
        return None, None, None


def remove_shared_ax(ax):
    shax = ax.get_shared_x_axes()
    shay = ax.get_shared_y_axes()
    shax.remove(ax)
    shay.remove(ax)
    for axis in [ax.xaxis, ax.yaxis]:
        ticker = mpl.axis.Ticker()
        axis.major = ticker
        axis.minor = ticker
        loc = mpl.ticker.NullLocator()
        fmt = mpl.ticker.NullFormatter()
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(fmt)


def get_colors(labels, palette, desat=0.7):
    if isinstance(palette, dict):
        colors = np.vectorize(palette.get)(labels)
        return colors
    elif isinstance(palette, str):
        uni_labels = np.unique(labels)
        palette = sns.color_palette(palette, n_colors=len(uni_labels), desat=desat)
        return get_colors(labels, dict(zip(uni_labels, palette)))


def draw_separators(
    ax,
    ax_type="x",
    sort_meta=None,
    sort_class=None,
    divider=None,
    colors=None,
    plot_type="heatmap",
    gridline_kws=None,
    use_ticks=True,
    tick_fontsize=10,
    minor_ticking=False,
    tick_rot=45,
):

    if gridline_kws is None:
        gridline_kws = dict(color="grey", linestyle="--", alpha=0.7, linewidth=1)

    if plot_type == "heatmap":
        boost = 0
    elif plot_type == "scattermap":
        boost = 0.5

    # get info about the separators
    first_inds, middle_inds, middle_labels = _get_tick_info(sort_meta, sort_class)

    if ax_type == "x":
        axis = ax.xaxis
    else:
        axis = ax.yaxis

    # draw the border lines
    for t in first_inds:
        if ax_type == "x":
            ax.axvline(t - boost, **gridline_kws)
        else:
            ax.axhline(t - boost, **gridline_kws)

    if colors is not None:
        if divider is None:
            divider = make_axes_locatable(ax)
        axis.set_ticks([])
        axis.set_ticks([])
        sort_class = sort_class[:1]  # TODO fix
        for sc in sort_class:  # TODO this will break for more than one category
            if ax_type == "x":
                cax = divider.append_axes("top", size="3%", pad=0, sharex=ax)
            elif ax_type == "y":
                cax = divider.append_axes("left", size="3%", pad=0, sharey=ax)

            classes = sort_meta[sc].values
            # colors = get_colors(classes, colors)

            from matplotlib.colors import ListedColormap

            # make colormap
            uni_classes = np.unique(classes)
            class_map = dict(zip(uni_classes, range(len(uni_classes))))
            color_dict = colors  # TODO make this work when not a dict
            color_sorted = np.vectorize(color_dict.get)(uni_classes)

            lc = ListedColormap(color_sorted)
            class_indicator = np.vectorize(class_map.get)(classes)
            if ax_type == "x":
                class_indicator = class_indicator.reshape(1, len(classes))
            elif ax_type == "y":
                class_indicator = class_indicator.reshape(len(classes), 1)
            sns.heatmap(
                class_indicator,
                cmap=lc,
                cbar=False,
                yticklabels=False,
                xticklabels=False,
                ax=cax,
                square=False,
            )

    if use_ticks:
        if use_ticks and colors is not None:
            tick_ax = cax
        else:
            tick_ax = ax
        if colors is not None:
            remove_shared_ax(cax)
        # add tick labels and locs
        if ax_type == "x":
            tick_ax.set_xticks(middle_inds)
            if minor_ticking:
                tick_ax.set_xticklabels(middle_labels[0::2])
                tick_ax.set_xticklabels(middle_labels[1::2], minor=True)
            else:
                if tick_rot != 0:
                    tick_ax.set_xticklabels(
                        middle_labels, rotation=tick_rot, ha="left", va="bottom"
                    )
                else:
                    tick_ax.set_xticklabels(
                        middle_labels, rotation=tick_rot, ha="center", va="bottom"
                    )
            tick_ax.xaxis.tick_top()
        elif ax_type == "y":
            tick_ax.set_yticks(middle_inds)
            if minor_ticking:
                tick_ax.set_yticklabels(middle_labels[0::2])
                tick_ax.set_yticklabels(middle_labels[1::2], minor=True)
            else:
                tick_ax.set_yticklabels(
                    middle_labels, rotation=tick_rot, ha="right", va="center"
                )
    if colors is not None:
        return cax


def matrixplot(
    data,
    ax=None,
    plot_type="heatmap",
    row_meta=None,
    col_meta=None,
    row_sort_class=None,
    col_sort_class=None,
    row_colors=None,
    col_colors=None,
    row_class_order="size",
    col_class_order="size",
    row_item_order=None,
    col_item_order=None,
    row_ticks=True,
    col_ticks=True,
    border=True,
    minor_ticking=False,
    cmap="RdBu_r",
    sizes=(10, 40),
    square=False,
    gridline_kws=None,
    spinestyle_kws=None,
    tick_rot=0,
):
    # TODO probably remove these
    tick_fontsize = 10
    tick_pad = [0, 0]
    base_tick_pad = 5

    if spinestyle_kws is None:
        spinestyle_kws = dict(linestyle="-", linewidth=1, alpha=0.7, color="black")

    # sort the data and metadata
    if row_meta is not None and row_sort_class is not None:
        row_perm_inds, row_meta = sort_meta(
            row_meta,
            row_sort_class,
            class_order=row_class_order,
            sort_item=row_item_order,
        )
    else:
        row_perm_inds = np.arange(data.shape[0])
    if col_meta is not None and col_sort_class is not None:
        col_perm_inds, col_meta = sort_meta(
            col_meta,
            col_sort_class,
            class_order=col_class_order,
            sort_item=col_item_order,
        )
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

    divider = make_axes_locatable(ax)

    top_cax = draw_separators(
        ax,
        divider=divider,
        ax_type="x",
        sort_meta=col_meta,
        sort_class=col_sort_class,
        colors=col_colors,
        plot_type=plot_type,
        use_ticks=col_ticks,
        tick_rot=tick_rot,
        gridline_kws=gridline_kws,
    )
    left_cax = draw_separators(
        ax,
        divider=divider,
        ax_type="y",
        sort_meta=row_meta,
        sort_class=row_sort_class,
        colors=row_colors,
        plot_type=plot_type,
        use_ticks=row_ticks,
        tick_rot=0,
        gridline_kws=gridline_kws,
    )

    # spines
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(spinestyle_kws["color"])
            spine.set_linewidth(spinestyle_kws["linewidth"])
            spine.set_linestyle(spinestyle_kws["linestyle"])
            spine.set_alpha(spinestyle_kws["alpha"])

    return ax, divider, top_cax, left_cax
