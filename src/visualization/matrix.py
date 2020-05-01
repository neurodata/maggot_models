import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from src.visualization import gridmap, remove_spines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.colors import ListedColormap


def sort_meta(length, meta, sort_class, sort_item=None, class_order="size"):
    if meta is None or len(meta) == 0:
        return np.arange(length), meta
    meta = meta.copy()
    total_sort_by = []
    for sc in sort_class:
        if class_order == "size":
            class_size = meta.groupby(sc).size()
            # negative so we can sort alphabetical still in one line
            meta[f"{sc}_size"] = -meta[sc].map(class_size)
            total_sort_by.append(f"{sc}_size")
        elif len(class_order) > 0:
            for co in class_order:
                class_value = meta.groupby(sc)[co].mean()
                meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
                total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    total_sort_by += sort_item
    meta["sort_idx"] = range(len(meta))
    if len(total_sort_by) > 0:
        meta.sort_values(total_sort_by, inplace=True)
    perm_inds = meta["sort_idx"].values
    return perm_inds, meta


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


def draw_colors(
    ax, divider=None, ax_type="x", colors=None, palette="tab10", sort_meta=None
):
    if len(colors) > 0:
        if ax_type == "x":
            cax = divider.append_axes("top", size="3%", pad=0, sharex=ax)
        elif ax_type == "y":
            cax = divider.append_axes("left", size="3%", pad=0, sharey=ax)

        classes = sort_meta[
            colors[0]
        ]  # TODO eventually could allow for multiple sets of colors

        if isinstance(palette, dict):
            color_dict = palette
        elif isinstance(palette, str):
            color_dict = dict(
                zip(classes.unique(), sns.color_palette(palette, classes.nunique()))
            )
        # make colormap
        uni_classes = np.unique(classes)
        class_map = dict(zip(uni_classes, range(len(uni_classes))))
        color_sorted = np.vectorize(color_dict.get)(uni_classes)
        # HACK fix the below 3 lines
        # matrix is flipped when using color palette sometimes
        color_sorted = np.array(color_sorted)
        if len(color_sorted) != len(uni_classes):
            color_sorted = color_sorted.T
        lc = ListedColormap(color_sorted)
        # map each class to an integer to use to make the color heatmap
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
        return cax
    else:
        return ax


def _get_separator_info(sort_meta, sort_class):
    """ Assumes meta is already sorted
    """
    if sort_meta is None and sort_class is None:
        return None
    # sort_meta[sort_class].fillna("", inplace=True)
    sort_meta["sort_idx"] = range(len(sort_meta))
    first_df = sort_meta.groupby(sort_class, sort=False).first()
    sep_inds = list(first_df["sort_idx"].values)
    last_df = sort_meta.groupby(sort_class, sort=False).last()
    sep_inds.append(last_df["sort_idx"].values[-1] + 1)
    return sep_inds


def _get_tick_info(sort_meta, sort_class):
    if sort_meta is None and sort_class is None:
        return None, None

    # first_df = sort_meta.groupby(sort_class, sort=False).first()

    middle_df = sort_meta.groupby(sort_class, sort=False).mean()
    middle_inds = np.array(middle_df["sort_idx"].values) + 0.5
    middle_labels = list(middle_df.index.get_level_values(sort_class[0]))

    # need to return the location of the tick, the label, and the divider
    return middle_inds, middle_labels


def draw_ticks(
    tick_ax,
    sort_meta=None,
    sort_class=None,
    ax_type="x",
    tick_rot=0,
    tick_ax_border=True,
):
    tick_inds, tick_labels = _get_tick_info(sort_meta, sort_class)
    if tick_rot != 0:
        ha = "center"
        va = "bottom"
    else:
        ha = "center"
        va = "bottom"
    if ax_type == "x":
        tick_ax.set_xticks(tick_inds)
        tick_ax.set_xticklabels(tick_labels, rotation=tick_rot, ha=ha, va=va)
        tick_ax.xaxis.tick_top()
    else:
        tick_ax.set_yticks(tick_inds)
        tick_ax.set_yticklabels(tick_labels, ha="right", va="center")

    if tick_ax_border:
        sep_inds = _get_separator_info(sort_meta, sort_class)
        for t in sep_inds:
            if ax_type == "x":
                tick_ax.axvline(t, color="black", linestyle="-", alpha=1, linewidth=2)
            else:
                tick_ax.axhline(t, color="black", linestyle="-", alpha=1, linewidth=2)


def draw_separators(
    ax,
    ax_type="x",
    sort_meta=None,
    sort_class=None,
    plot_type="heatmap",
    gridline_kws=None,
):
    """[summary]
    
    Parameters
    ----------
    ax : [type]
        [description]
    ax_type : str, optional
        [description], by default "x"
    sort_meta : [type], optional
        [description], by default None
    sort_class : [type], optional
        [description], by default None
    divider : [type], optional
        [description], by default None
    colors : dict or string, optional
        By default, None, no colors are plotted for categorical labels
        if dict, should map elements of `sort_class` to a color
        if string, should be a palette specification (see mpl/sns) # TODO broken
    plot_type : str, optional
        [description], by default "heatmap"
    gridline_kws : [type], optional
        [description], by default None
    use_ticks : bool, optional
        [description], by default True
    tick_fontsize : int, optional
        [description], by default 10
    minor_ticking : bool, optional
        [description], by default False
    tick_rot : int, optional
        [description], by default 45
    
    Returns
    -------
    [type]
        [description]
    """
    if len(sort_class) > 0:
        if gridline_kws is None:
            gridline_kws = dict(color="grey", linestyle="--", alpha=0.7, linewidth=1)

        if plot_type == "heatmap":
            boost = 0
        elif plot_type == "scattermap":
            boost = 0.5

        sep_inds = _get_separator_info(sort_meta, sort_class)

        if ax_type == "x":
            lims = ax.get_xlim()
            drawer = ax.axvline
        else:
            lims = ax.get_ylim()
            drawer = ax.axhline

        # draw the  lines
        for t in sep_inds:
            if t not in lims:  # avoid drawing lines on the borders
                drawer(t - boost, **gridline_kws)


def _process_meta(meta, sort_class):
    if meta is None and sort_class is None:
        return None, None
    elif meta is not None and sort_class is None:
        return meta, []
    elif isinstance(meta, pd.DataFrame):
        if isinstance(sort_class, str):
            sort_class = [sort_class]
        else:
            try:  # if sort class is a single element
                iter(sort_class)
            except TypeError:
                raise TypeError("`sort_class` must be an iterable or string")
    elif isinstance(sort_class, pd.Series) and meta is None:
        meta = sort_class.to_frame(name=0)
        sort_class = [0]
    elif isinstance(sort_class, list) and meta is None:
        meta = pd.DataFrame({i: elem for i, elem in enumerate(sort_class)})
        sort_class = list(range(meta.shape[1]))
    elif isinstance(sort_class, np.ndarray) and meta is None:
        meta = pd.DataFrame(sort_class)
        sort_class = [0]
    else:
        raise ValueError("Improper metadata spec for matrixplot")
    return meta, sort_class


def _check_length(item, name, length):
    if length != len(item):
        raise ValueError(
            f"Length of {name} must be the same as corresponding data axis"
        )


def _check_item_in_meta(meta, item, name):
    if item is None:
        return []
    if isinstance(item, str):
        item = [item]
    else:
        try:
            iter(item)
        except TypeError:
            msg = (
                f"{name} must be an iterable or string corresponding to columns in meta"
            )
            raise TypeError(msg)
    for col_name in item:
        if col_name not in meta.columns:
            raise ValueError(f"{name} is not a column in the meta dataframe.")
    return item


def _item_to_df(item, name, length):
    if item is None:
        return None, []

    if isinstance(item, pd.Series):
        _check_length(item, name, length)
        item_meta = item.to_frame(name=f"{name}_0")
    elif isinstance(item, list):
        if len(item) == length:  # assuming elements of list are metadata
            item = [item]
        for elem in item:
            _check_length(elem, name, length)
        item_meta = pd.DataFrame({f"{name}_{i}": elem for i, elem in enumerate(item)})
    elif isinstance(item, np.ndarray):
        if item.ndim > 2:
            raise ValueError(f"Numpy array passed as {name} must be 1 or 2d.")
        _check_length(item, name, length)
        if item.ndim < 2:
            item = np.atleast_2d(item).T
        item_meta = pd.DataFrame(
            data=item, columns=[f"{name}_{i}" for i in range(item.shape[1])]
        )
    else:
        raise ValueError(f"{name} must be a pd.Series, np.array, or list.")

    item = list(item_meta.columns.values)
    return item_meta, item


def _check_sorting_kws(length, meta, sort_class, class_order, item_order, colors):
    if isinstance(meta, pd.DataFrame):
        # if meta is here, than everything else must be column item in meta
        _check_length(meta, "meta", length)
        sort_class = _check_item_in_meta(meta, sort_class, "sort_class")
        class_order = _check_item_in_meta(meta, class_order, "class_order")
        item_order = _check_item_in_meta(meta, item_order, "item_order")
        colors = _check_item_in_meta(meta, colors, "colors")
    else:
        # otherwise, arguments can be a hodgepodge of stuff
        sort_class_meta, sort_class = _item_to_df(sort_class, "sort_class", length)
        class_order_meta, class_order = _item_to_df(class_order, "class_order", length)
        item_order_meta, item_order = _item_to_df(item_order, "item_order", length)
        color_meta, colors = _item_to_df(colors, "colors", length)
        metas = []
        for m in [sort_class_meta, class_order_meta, item_order_meta, color_meta]:
            if m is not None:
                metas.append(m)
        if len(metas) > 0:
            meta = pd.concat(metas, axis=1)
        else:
            meta = pd.DataFrame()
    return meta, sort_class, class_order, item_order, colors


def _check_data(data):
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a np.ndarray.")
    if data.ndim != 2:
        raise ValueError("data must have dimension 2.")


def _check_boolean_inputs(*args):
    pass


def matrixplot(
    data,
    ax=None,
    plot_type="heatmap",
    row_meta=None,
    col_meta=None,
    row_sort_class=None,
    col_sort_class=None,
    row_class_order=None,
    col_class_order=None,
    row_ticks=True,
    col_ticks=True,
    row_item_order=None,
    col_item_order=None,
    row_colors=None,
    col_colors=None,
    row_palette="tab10",
    col_palette="tab10",
    col_highlight=None,
    row_highlight=None,
    col_tick_pad=None,
    row_tick_pad=None,
    border=True,
    minor_ticking=False,
    tick_rot=0,
    center=0,
    cmap="RdBu_r",
    sizes=(5, 10),
    square=False,
    gridline_kws=None,
    spinestyle_kws=None,
    highlight_kws=None,
    **kws,
):
    """Plotting matrices
    
    Parameters
    ----------
    data : np.ndarray, ndim=2
        matrix to plot
    ax : matplotlib axes object, optional
        [description], by default None
    plot_type : str, optional
        One of "heatmap" or "scattermap", by default "heatmap"
    row_meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        [description], by default None
    col_meta : [type], optional
        [description], by default None
    row_sort_class : list or np.ndarray, optional
        [description], by default None
    col_sort_class : list or np.ndarray, optional
        [description], by default None
    row_colors : dict, optional
        [description], by default None
    col_colors : dict, optional
        [description], by default None
    row_class_order : str, optional
        [description], by default "size"
    col_class_order : str, optional
        [description], by default "size"
    row_item_order : string or list of string, optional
        attribute in meta by which to sort elements within a class, by default None
    col_item_order : [type], optional
        [description], by default None
    row_ticks : bool, optional
        [description], by default True
    col_ticks : bool, optional
        [description], by default True
    border : bool, optional
        [description], by default True
    minor_ticking : bool, optional
        [description], by default False
    cmap : str, optional
        [description], by default "RdBu_r"
    sizes : tuple, optional
        [description], by default (10, 40)
    square : bool, optional
        [description], by default False
    gridline_kws : [type], optional
        [description], by default None
    spinestyle_kws : [type], optional
        [description], by default None
    tick_rot : int, optional
        [description], by default 0
    
    Returns
    -------
    [type]
        [description]
    """

    _check_data(data)

    plot_type_opts = ["scattermap", "heatmap"]
    if plot_type not in plot_type_opts:
        raise ValueError(f"`plot_type` must be one of {plot_type_opts}")

    row_meta, row_sort_class, row_class_order, row_item_order, row_colors = _check_sorting_kws(
        data.shape[0],
        row_meta,
        row_sort_class,
        row_class_order,
        row_item_order,
        row_colors,
    )

    col_meta, col_sort_class, col_class_order, col_item_order, col_colors = _check_sorting_kws(
        data.shape[1],
        col_meta,
        col_sort_class,
        col_class_order,
        col_item_order,
        col_colors,
    )

    # sort the data and metadata
    row_perm_inds, row_meta = sort_meta(
        data.shape[0],
        row_meta,
        row_sort_class,
        class_order=row_class_order,
        sort_item=row_item_order,
    )
    col_perm_inds, col_meta = sort_meta(
        data.shape[1],
        col_meta,
        col_sort_class,
        class_order=col_class_order,
        sort_item=col_item_order,
    )
    data = data[np.ix_(row_perm_inds, col_perm_inds)]

    # draw the main heatmap/scattermap
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    if plot_type == "heatmap":
        sns.heatmap(data, cmap=cmap, ax=ax, center=center, **kws)
    elif plot_type == "scattermap":
        gridmap(data, ax=ax, sizes=sizes, border=False, **kws)

    if square:
        ax.axis("square")

    ax.set_ylim(data.shape[0], 0)
    ax.set_xlim(0, data.shape[1])

    # this will let us make axes for the colors and ticks as necessary
    divider = make_axes_locatable(ax)

    # draw colors
    # note that top_cax and left_cax may = ax if no colors are requested
    top_cax = draw_colors(
        ax,
        divider=divider,
        ax_type="x",
        colors=col_colors,
        palette=col_palette,
        sort_meta=col_meta,
    )
    top_cax.xaxis.set_label_position("top")

    left_cax = draw_colors(
        ax,
        divider=divider,
        ax_type="y",
        colors=row_colors,
        palette=row_palette,
        sort_meta=row_meta,
    )

    remove_shared_ax(ax)

    # draw separators
    draw_separators(
        ax,
        ax_type="x",
        sort_meta=col_meta,
        sort_class=col_sort_class,
        plot_type=plot_type,
        gridline_kws=gridline_kws,
    )
    draw_separators(
        ax,
        ax_type="y",
        sort_meta=row_meta,
        sort_class=row_sort_class,
        plot_type=plot_type,
        gridline_kws=gridline_kws,
    )

    # draw ticks
    if len(col_sort_class) > 0 and col_ticks:
        if col_tick_pad is None:
            col_tick_pad = len(col_sort_class) * [0.5]

        tick_ax = top_cax  # start with the axes we already have
        tick_ax_border = False
        rev_col_sort_class = list(col_sort_class[::-1])

        for i, sc in enumerate(rev_col_sort_class):
            if i > 0:  # add a new axis for ticks
                tick_ax = divider.append_axes(
                    "top", size="1%", pad=col_tick_pad[i], sharex=ax
                )
                remove_shared_ax(tick_ax)
                tick_ax.spines["right"].set_visible(True)
                tick_ax.spines["top"].set_visible(True)
                tick_ax.spines["left"].set_visible(True)
                tick_ax.spines["bottom"].set_visible(False)
                tick_ax_border = True

            draw_ticks(
                tick_ax,
                col_meta,
                rev_col_sort_class[i:],
                ax_type="x",
                tick_rot=tick_rot,
                tick_ax_border=tick_ax_border,
            )
            ax.xaxis.set_label_position("top")

    if len(row_sort_class) > 0 and row_ticks:
        tick_ax = left_cax  # start with the axes we already have
        tick_ax_border = False
        rev_row_sort_class = list(row_sort_class[::-1])
        if row_tick_pad is None:
            row_tick_pad = len(row_sort_class) * [0.5]

        for i, sc in enumerate(rev_row_sort_class):
            if i > 0:  # add a new axis for ticks
                tick_ax = divider.append_axes(
                    "left", size="1%", pad=row_tick_pad[i], sharey=ax
                )
                remove_shared_ax(tick_ax)
                tick_ax.spines["right"].set_visible(False)
                tick_ax.spines["top"].set_visible(True)
                tick_ax.spines["bottom"].set_visible(True)
                tick_ax.spines["left"].set_visible(True)
                tick_ax_border = True

            draw_ticks(
                tick_ax,
                row_meta,
                rev_row_sort_class[i:],
                ax_type="y",
                tick_ax_border=tick_ax_border,
            )

    # if highlight_kws is None:
    #     highlight_kws = dict(color="black", linestyle="-", linewidth=1)
    # if col_highlight is not None:
    #     draw_separators(
    #         ax,
    #         divider=divider,
    #         # tick_ax=tick_ax,
    #         ax_type="x",
    #         sort_meta=col_meta,
    #         all_sort_class=col_sort_class,
    #         level_sort_class=col_highlight,
    #         plot_type=plot_type,
    #         use_ticks=False,
    #         gridline_kws=highlight_kws,
    #         tick_ax_border=False,
    #     )
    # if row_highlight is not None:
    #     draw_separators(
    #         ax,
    #         divider=divider,
    #         # tick_ax=tick_ax,
    #         ax_type="y",
    #         sort_meta=row_meta,
    #         all_sort_class=row_sort_class,
    #         level_sort_class=row_highlight,
    #         plot_type=plot_type,
    #         use_ticks=False,
    #         gridline_kws=highlight_kws,
    #         tick_ax_border=False,
    #     )

    # spines
    if spinestyle_kws is None:
        spinestyle_kws = dict(linestyle="-", linewidth=1, alpha=0.7, color="black")
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(spinestyle_kws["color"])
            spine.set_linewidth(spinestyle_kws["linewidth"])
            spine.set_linestyle(spinestyle_kws["linestyle"])
            spine.set_alpha(spinestyle_kws["alpha"])

    return ax, divider, top_cax, left_cax


def adjplot(
    data,
    ax=None,
    plot_type="heatmap",
    meta=None,
    sort_class=None,
    class_order=None,
    item_order=None,
    colors=None,
    highlight=None,
    palette="tab10",
    ticks=True,
    border=True,
    tick_rot=0,
    center=0,
    cmap="RdBu_r",
    sizes=(5, 10),
    square=True,
    gridline_kws=None,
    spinestyle_kws=None,
    highlight_kws=None,
    col_tick_pad=None,
    row_tick_pad=None,
    **kws,
):
    outs = matrixplot(
        data,
        ax=ax,
        plot_type=plot_type,
        row_meta=meta,
        col_meta=meta,
        row_sort_class=sort_class,
        col_sort_class=sort_class,
        row_class_order=class_order,
        col_class_order=class_order,
        row_item_order=item_order,
        col_item_order=item_order,
        row_colors=colors,
        col_colors=colors,
        row_palette=palette,
        col_palette=palette,
        row_highlight=highlight,
        col_highlight=highlight,
        row_ticks=ticks,
        col_ticks=ticks,
        border=border,
        tick_rot=tick_rot,
        center=center,
        cmap=cmap,
        sizes=sizes,
        square=square,
        gridline_kws=gridline_kws,
        spinestyle_kws=spinestyle_kws,
        highlight_kws=highlight_kws,
        row_tick_pad=row_tick_pad,
        col_tick_pad=col_tick_pad,
        **kws,
    )
    return outs

