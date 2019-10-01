import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _sort_inds(graph, inner_labels, outer_labels, sort_nodes):
    sort_df = pd.DataFrame(columns=("inner_labels", "outer_labels"))
    sort_df["inner_labels"] = inner_labels
    sort_df["outer_labels"] = outer_labels

    # get frequencies of the different labels so we can sort by them
    inner_label_counts = _get_freq_vec(inner_labels)
    outer_label_counts = _get_freq_vec(outer_labels)

    # inverse counts so we can sort largest to smallest
    # would rather do it this way so can still sort alphabetical for ties
    sort_df["inner_counts"] = len(inner_labels) - inner_label_counts
    sort_df["outer_counts"] = len(outer_labels) - outer_label_counts

    # get node edge sums (not exactly degrees if weighted)
    node_edgesums = graph.sum(axis=1) + graph.sum(axis=0)
    sort_df["node_edgesums"] = node_edgesums.max() - node_edgesums

    if sort_nodes:
        by = [
            "outer_counts",
            "outer_labels",
            "inner_counts",
            "inner_labels",
            "node_edgesums",
        ]
    else:
        by = ["outer_counts", "outer_labels", "inner_counts", "inner_labels"]
    sort_df.sort_values(by=by, kind="mergesort", inplace=True)

    sorted_inds = sort_df.index.values
    return sorted_inds


def _sort_graph(graph, inner_labels, outer_labels, sort_nodes):
    inds = _sort_inds(graph, inner_labels, outer_labels, sort_nodes)
    graph = graph[inds, :][:, inds]
    return graph


def _get_freqs(inner_labels, outer_labels=None):
    # use this because unique would give alphabetical
    _, outer_freq = _unique_like(outer_labels)
    outer_freq_cumsum = np.hstack((0, outer_freq.cumsum()))

    # for each group of outer labels, calculate the boundaries of the inner labels
    inner_freq = np.array([])
    for i in range(outer_freq.size):
        start_ind = outer_freq_cumsum[i]
        stop_ind = outer_freq_cumsum[i + 1]
        _, temp_freq = _unique_like(inner_labels[start_ind:stop_ind])
        inner_freq = np.hstack([inner_freq, temp_freq])
    inner_freq_cumsum = np.hstack((0, inner_freq.cumsum()))

    return inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum


def _get_freq_vec(vals):
    # give each set of labels a vector corresponding to its frequency
    _, inv, counts = np.unique(vals, return_counts=True, return_inverse=True)
    count_vec = counts[inv]
    return count_vec


def _unique_like(vals):
    # gives output like
    uniques, inds, counts = np.unique(vals, return_index=True, return_counts=True)
    inds_sort = np.argsort(inds)
    uniques = uniques[inds_sort]
    counts = counts[inds_sort]
    return uniques, counts


# assume that the graph has already been plotted in sorted form
def _plot_groups(
    ax, divider, graph, sorted_inds, inner_labels, outer_labels=None, fontsize=30
):
    inner_labels = np.array(inner_labels)
    plot_outer = True
    if outer_labels is None:
        outer_labels = np.ones_like(inner_labels)
        plot_outer = False

    # sorted_inds = _sort_inds(graph, inner_labels, outer_labels, False)
    inner_labels = inner_labels[sorted_inds]
    outer_labels = outer_labels[sorted_inds]

    inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum = _get_freqs(
        inner_labels, outer_labels
    )
    inner_unique, _ = _unique_like(inner_labels)
    outer_unique, _ = _unique_like(outer_labels)

    # n_verts = graph.shape[0]
    axline_kws = dict(linestyle="dashed", lw=0.9, alpha=0.3, zorder=3, color="grey")
    # draw lines
    for x in inner_freq_cumsum[1:-1]:
        ax.vlines(x, 0, graph.shape[0] + 1, **axline_kws)
        # ax.hlines(x, 0, graph.shape[1] + 1, **axline_kws)

    # add specific lines for the borders of the plot
    pad = 0.0001
    low = pad
    high = 1 - pad
    ax.plot((low, low), (low, high), transform=ax.transAxes, **axline_kws)
    ax.plot((low, high), (low, low), transform=ax.transAxes, **axline_kws)
    ax.plot((high, high), (low, high), transform=ax.transAxes, **axline_kws)
    ax.plot((low, high), (high, high), transform=ax.transAxes, **axline_kws)

    # generic curve that we will use for everything
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
    tan = np.tan(lx)
    curve = np.hstack((tan[::-1], tan))

    # divider = make_axes_locatable(ax)

    # inner curve generation
    inner_tick_loc = inner_freq.cumsum() - inner_freq / 2
    inner_tick_width = inner_freq / 2
    # outer curve generation
    outer_tick_loc = outer_freq.cumsum() - outer_freq / 2
    outer_tick_width = outer_freq / 2

    # top inner curves
    ax_x = divider.new_vertical(size="5%", pad=0.0, pack_start=True)
    ax.figure.add_axes(ax_x)
    _plot_brackets(
        ax_x,
        np.tile(inner_unique, len(outer_unique)),
        inner_tick_loc,
        inner_tick_width,
        curve,
        "inner",
        "x",
        graph.shape[1],
        fontsize,
    )
    return ax


def _plot_brackets(
    ax, group_names, tick_loc, tick_width, curve, level, axis, max_size, fontsize
):
    for x0, width in zip(tick_loc, tick_width):
        x = np.linspace(x0 - width, x0 + width, 1000)
        if axis == "x":
            ax.plot(x, curve, c="k")
            ax.patch.set_alpha(0)
        elif axis == "y":
            ax.plot(curve, x, c="k")
            ax.patch.set_alpha(0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis=axis, which="both", length=0, pad=7)
    for direction in ["left", "right", "bottom", "top"]:
        ax.spines[direction].set_visible(False)
    if axis == "x":
        ax.set_xticks(tick_loc)
        ax.set_xticklabels(
            group_names,
            fontsize=fontsize,
            verticalalignment="center",
            horizontalalignment="right",
            rotation=90,
            rotation_mode="anchor",
        )
        # ax.xaxis.set_label_position("bottom")
        # ax.xaxis.tick_top()
        ax.xaxis.labelpad = 200
        ax.set_xlim(0, max_size)
        ax.tick_params(axis="x", which="major", pad=5 + fontsize / 4)
    elif axis == "y":
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(group_names, fontsize=fontsize, verticalalignment="center")
        ax.set_ylim(0, max_size)
        ax.invert_yaxis()


def incidence_plot(adj, classes, from_class):
    """Plots non-square adjacency, sorts by class label, sums input to columns for 
    marginal
    
    Parameters
    ----------
    adj : np.array
        n x n adjacency matrix
    classes : np.ndarray
        n-length indicator of class membership for sorting nodes 
    from_class : str 
        which class to select on the left

    Returns
    -------
    ax
        matplotlib axes
    """
    sort_inds = _sort_inds(adj, classes, np.ones_like(classes), True)
    sort_adj = _sort_graph(adj, classes, np.ones_like(classes), True)
    sort_classes = classes[sort_inds]
    proj_inds = np.where(sort_classes == from_class)[0]
    clipped_adj = sort_adj[proj_inds, :]

    plt.figure(figsize=(30, 10))
    xs, ys = np.meshgrid(
        range(1, clipped_adj.shape[1] + 1), range(1, clipped_adj.shape[0] + 1)
    )
    nonzero_inds = np.nonzero(clipped_adj.ravel())
    x = xs.ravel()[nonzero_inds]
    y = ys.ravel()[nonzero_inds]
    weights = clipped_adj.ravel()[nonzero_inds]
    ax = sns.scatterplot(x=x, y=y, size=weights, legend=False)

    plt.ylabel(from_class)
    plt.title(from_class, pad=100)

    divider = make_axes_locatable(ax)

    ax_top = divider.new_vertical(size="25%", pad=0.0, pack_start=False)
    ax.figure.add_axes(ax_top)
    sums = clipped_adj.sum(axis=0)
    ax_top.bar(range(1, clipped_adj.shape[1] + 1), sums, width=5)
    ax_top.set_xlim((0, clipped_adj.shape[1]))
    ax_top.axis("off")
    ax_top.hlines(0.05, 0, clipped_adj.shape[1] + 1, color="r", linestyle="--")

    ax = _plot_groups(
        ax, divider, clipped_adj, sort_inds, classes, outer_labels=None, fontsize=10
    )
    ax.set_xlim((0, clipped_adj.shape[1]))
    ax.set_ylim((0, clipped_adj.shape[0]))
    ax.axis("off")
    return ax
