import math
from collections import defaultdict
from operator import itemgetter

import colorcet as cc
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.utils import check_array, check_consistent_length

from graspy.embed import select_dimension, selectSVD
from graspy.models import SBMEstimator
from graspy.plot import heatmap
from graspy.utils import binarize, cartprod
from src.utils import get_sbm_prob, savefig

from .manual_colors import CLASS_COLOR_DICT


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

    #
    if not isinstance(from_class, list):
        from_class = [from_class]

    all_proj_inds = []
    for i, class_name in enumerate(from_class):
        print(class_name)
        proj_inds = np.where(sort_classes == class_name)[0]
        all_proj_inds += list(proj_inds)
    print(all_proj_inds)
    all_proj_inds = np.unique(all_proj_inds)
    print(all_proj_inds)
    # pred_cell_ids = np.setdiff1d(pred_cell_ids, PREDEFINED_IDS)

    #

    # proj_inds = np.where(sort_classes == from_class)[0]
    clipped_adj = sort_adj[all_proj_inds, :]

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
        ax, divider, clipped_adj, sort_inds, classes, outer_labels=None, fontsize=14
    )
    ax.set_xlim((0, clipped_adj.shape[1]))
    ax.set_ylim((0, clipped_adj.shape[0]))
    ax.axis("off")
    return ax


def screeplot(
    X,
    title="Scree plot",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    cumulative=False,
    show_first=40,
    n_elbows=4,
):
    r"""
    Plots the distribution of singular values for a matrix, either showing the 
    raw distribution or an empirical CDF (depending on ``cumulative``)

    Parameters
    ----------
    X : np.ndarray (2D)
        input matrix 
    title : string, default : 'Scree plot'
        plot title 
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font 
        elements.
    figsize : tuple of length 2, default (10, 5)
        size of the figure (width, height)
    cumulative : boolean, default: True
        whether or not to plot a cumulative cdf of singular values 
    show_first : int or None, default: None 
        whether to restrict the plot to the first ``show_first`` components

    Returns
    -------
    ax : matplotlib axis object
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )
    # check_array(X)
    if show_first is not None:
        if not isinstance(show_first, int):
            msg = "show_first must be an int"
            raise TypeError(msg)
    if not isinstance(cumulative, bool):
        msg = "cumulative must be a boolean"
        raise TypeError(msg)
    # n_components = min(X.shape) - 1
    # _, D, _ = selectSVD(X, n_components=n_components, algorithm="full")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elbow_locs, elbow_vals = select_dimension(X, n_elbows=n_elbows)
    elbow_locs = np.array(elbow_locs)
    elbow_vals = np.array(elbow_vals)
    # D /= D.sum()
    D = elbow_vals / elbow_vals.sum()
    if cumulative:
        y = np.cumsum(D[:show_first])
    else:
        y = D[:show_first]
    _ = plt.figure(figsize=figsize)
    ax = plt.gca()
    xlabel = "Component"
    if cumulative:
        ylabel = "Variance explained"
    else:
        ylabel = "Normalized singular value"
    with sns.plotting_context(context=context, font_scale=font_scale):
        rng = range(1, len(y) + 1)
        plt.scatter(rng, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(elbow_locs, y[elbow_locs - 1], c="r")
        plt.ylim((y.min() - y.min() / 10, y.max() + (y.max() / 10)))
    return ax


def _check_common_inputs(
    figsize=None,
    height=None,
    title=None,
    context=None,
    font_scale=None,
    legend_name=None,
    title_pad=None,
    hier_label_fontsize=None,
):
    # Handle figsize
    if figsize is not None:
        if not isinstance(figsize, tuple):
            msg = "figsize must be a tuple, not {}.".format(type(figsize))
            raise TypeError(msg)

    # Handle heights
    if height is not None:
        if not isinstance(height, (int, float)):
            msg = "height must be an integer or float, not {}.".format(type(height))
            raise TypeError(msg)

    # Handle title
    if title is not None:
        if not isinstance(title, str):
            msg = "title must be a string, not {}.".format(type(title))
            raise TypeError(msg)

    # Handle context
    if context is not None:
        if not isinstance(context, str):
            msg = "context must be a string, not {}.".format(type(context))
            raise TypeError(msg)
        elif context not in ["paper", "notebook", "talk", "poster"]:
            msg = "context must be one of (paper, notebook, talk, poster), \
                not {}.".format(
                context
            )
            raise ValueError(msg)

    # Handle font_scale
    if font_scale is not None:
        if not isinstance(font_scale, (int, float)):
            msg = "font_scale must be an integer or float, not {}.".format(
                type(font_scale)
            )
            raise TypeError(msg)

    # Handle legend name
    if legend_name is not None:
        if not isinstance(legend_name, str):
            msg = "legend_name must be a string, not {}.".format(type(legend_name))
            raise TypeError(msg)

    if hier_label_fontsize is not None:
        if not isinstance(hier_label_fontsize, (int, float)):
            msg = "hier_label_fontsize must be a scalar, not {}.".format(
                type(legend_name)
            )
            raise TypeError(msg)

    if title_pad is not None:
        if not isinstance(title_pad, (int, float)):
            msg = "title_pad must be a scalar, not {}.".format(type(legend_name))
            raise TypeError(msg)


# -*- coding: utf-8 -*-
"""
Produces simple Sankey Diagrams with matplotlib.
@author: Anneya Golob & marcomanz & pierre-sassoulas & jorwoods
                      .-.
                 .--.(   ).--.
      <-.  .-.-.(.->          )_  .--.
       `-`(     )-'             `)    )
         (o  o  )                `)`-'
        (      )                ,)
        ( ()  )                 )
         `---"\    ,    ,    ,/`
               `--' `--' `--'
                |  |   |   |
                |  |   |   |
                '  |   '   |
"""


# matplotlib.use("Agg")


class PySankeyException(Exception):
    pass


class NullsInFrame(PySankeyException):
    pass


class LabelMismatch(PySankeyException):
    pass


def check_data_matches_labels(labels, data, side):
    if len(labels > 0):
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch(
                "{0} labels and data do not match.{1}".format(side, msg)
            )


def add_label_counts(labels):
    new_labels = labels.copy().astype("<U64")
    uni_labels, counts = np.unique(labels, return_counts=True)
    new_names = []
    for name, count in zip(uni_labels, counts):
        inds = np.where(labels == name)[0]
        new_name = str(name) + f" ({count})"
        new_labels[inds] = new_name
        new_names.append(new_name)
    return new_labels, dict(zip(uni_labels, new_names))


def rename_keys(d, keys):
    return dict([(keys.get(k), v) for k, v in d.items()])


def sankey(
    ax,
    left,
    right,
    leftWeight=None,
    rightWeight=None,
    colorDict=None,
    leftLabels=None,
    rightLabels=None,
    aspect=4,
    rightColor=False,
    fontsize=14,
    figureName=None,
    closePlot=False,
    palette="Set1",
    append_counts=True,
):
    """
    Make Sankey Diagram showing flow from left-->right

    Inputs:
        left = NumPy array of object labels on the left of the diagram
        right = NumPy array of corresponding labels on the right of the diagram
            len(right) == len(left)
        leftWeight = NumPy array of weights for each strip starting from the
            left of the diagram, if not specified 1 is assigned
        rightWeight = NumPy array of weights for each strip starting from the
            right of the diagram, if not specified the corresponding leftWeight
            is assigned
        colorDict = Dictionary of colors to use for each label
            {'label':'color'}
        leftLabels = order of the left labels in the diagram
        rightLabels = order of the right labels in the diagram
        aspect = vertical extent of the diagram in units of horizontal extent
        rightColor = If true, each strip in the diagram will be be colored
                    according to its left label
    Ouput:
        None
    """
    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))

    if len(rightWeight) == 0:
        rightWeight = leftWeight

    if append_counts:
        left, new_left_map = add_label_counts(left)
        right, new_right_map = add_label_counts(right)
        new_left_map.update(new_right_map)
        colorDict = rename_keys(colorDict, new_left_map)

    # plt.figure()
    # plt.rc("text", usetex=False)
    # plt.rc("font", family="serif")

    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame(
        {
            "left": left,
            "right": right,
            "leftWeight": leftWeight,
            "rightWeight": rightWeight,
        },
        index=range(len(left)),
    )

    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise NullsInFrame("Sankey graph does not support null values.")

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(
        np.r_[dataFrame.left.unique(), dataFrame.right.unique()]
    ).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(dataFrame.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame["left"], "left")

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(dataFrame.right.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame["right"], "right")
    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        # palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = (
                "The colorDict parameter is missing values for the following labels : "
            )
            msg += "{}".format(", ".join(missing))
            raise ValueError(msg)

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
            ].leftWeight.sum()
            rightDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
            ].rightWeight.sum()
        ns_l[leftLabel] = leftDict
        ns_r[leftLabel] = rightDict

    # Determine positions of left label patches and total widths
    leftWidths = defaultdict()
    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD["left"] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
        if i == 0:
            myD["bottom"] = 0
            myD["top"] = myD["left"]
        else:
            myD["bottom"] = (
                leftWidths[leftLabels[i - 1]]["top"] + 0.02 * dataFrame.leftWeight.sum()
            )
            myD["top"] = myD["bottom"] + myD["left"]
            topEdge = myD["top"]
        leftWidths[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths = defaultdict()
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD["right"] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
        if i == 0:
            myD["bottom"] = 0
            myD["top"] = myD["right"]
        else:
            myD["bottom"] = (
                rightWidths[rightLabels[i - 1]]["top"]
                + 0.02 * dataFrame.rightWeight.sum()
            )
            myD["top"] = myD["bottom"] + myD["right"]
            topEdge = myD["top"]
        rightWidths[rightLabel] = myD

    # Total vertical extent of diagram
    xMax = topEdge / aspect

    # Draw vertical bars on left and right of each  label's section & print label
    for leftLabel in leftLabels:
        ax.fill_between(
            [-0.02 * xMax, 0],
            2 * [leftWidths[leftLabel]["bottom"]],
            2 * [leftWidths[leftLabel]["bottom"] + leftWidths[leftLabel]["left"]],
            color=colorDict[leftLabel],
            alpha=0.99,
        )
        ax.text(
            -0.05 * xMax,
            leftWidths[leftLabel]["bottom"] + 0.5 * leftWidths[leftLabel]["left"],
            leftLabel,
            {"ha": "right", "va": "center"},
            fontsize=fontsize,
        )
    for rightLabel in rightLabels:
        ax.fill_between(
            [xMax, 1.02 * xMax],
            2 * [rightWidths[rightLabel]["bottom"]],
            2 * [rightWidths[rightLabel]["bottom"] + rightWidths[rightLabel]["right"]],
            color=colorDict[rightLabel],
            alpha=0.99,
        )
        ax.text(
            1.05 * xMax,
            rightWidths[rightLabel]["bottom"] + 0.5 * rightWidths[rightLabel]["right"],
            rightLabel,
            {"ha": "left", "va": "center"},
            fontsize=fontsize,
        )

    # Plot strips
    for leftLabel in leftLabels:
        for rightLabel in rightLabels:
            labelColor = leftLabel
            if rightColor:
                labelColor = rightLabel
            if (
                len(
                    dataFrame[
                        (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                    ]
                )
                > 0
            ):
                # Create array of y values for each strip, half at left value,
                # half at right, convolve
                ys_d = np.array(
                    50 * [leftWidths[leftLabel]["bottom"]]
                    + 50 * [rightWidths[rightLabel]["bottom"]]
                )
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                ys_u = np.array(
                    50 * [leftWidths[leftLabel]["bottom"] + ns_l[leftLabel][rightLabel]]
                    + 50
                    * [rightWidths[rightLabel]["bottom"] + ns_r[leftLabel][rightLabel]]
                )
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")

                # Update bottom edges at each label, next strip starts at right place
                leftWidths[leftLabel]["bottom"] += ns_l[leftLabel][rightLabel]
                rightWidths[rightLabel]["bottom"] += ns_r[leftLabel][rightLabel]
                ax.fill_between(
                    np.linspace(0, xMax, len(ys_d)),
                    ys_d,
                    ys_u,
                    alpha=0.65,
                    color=colorDict[labelColor],
                )
    # plt.gca().axis("off")
    # plt.gcf().set_size_inches(6, 6)
    if figureName is not None:
        plt.savefig("{}.png".format(figureName), bbox_inches="tight", dpi=150)
    if closePlot:
        plt.close()


def hierplot(
    distance_matrix,
    dendrogram_size="45%",
    figsize=(10, 10),
    dendrogram_kws={},
    heatmap_kws={},
):
    dendrogram_kws["distance_sort"] = "descending"
    dendrogram_kws["color_threshold"] = 0
    dendrogram_kws["above_threshold_color"] = "k"

    inds = np.triu_indices_from(distance_matrix, k=1)
    condensed_distances = distance_matrix[inds]
    linkage_mat = linkage(condensed_distances, method="average")
    R = dendrogram(linkage_mat, no_plot=True, distance_sort="descending")
    # R = plot_dendrogram(model, no_plot=True, distance_sort="descending")
    inds = R["leaves"]
    distance_matrix = distance_matrix[np.ix_(inds, inds)]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        distance_matrix,
        cbar=True,
        square=True,
        xticklabels=True,
        yticklabels=False,
        cbar_kws=dict(shrink=0.5),
        **heatmap_kws,
    )

    divider = make_axes_locatable(ax)

    ax_x = divider.new_vertical(size=dendrogram_size, pack_start=False, pad=0.1)
    ax.figure.add_axes(ax_x)
    # plot_dendrogram(model, ax=ax_x, **dendrogram_kws)
    dendrogram(linkage_mat, ax=ax_x, **dendrogram_kws)
    ax_x.axis("off")

    ax_y = divider.new_horizontal(size=dendrogram_size, pack_start=True, pad=0.1)
    ax.figure.add_axes(ax_y)
    # R = plot_dendrogram(model, ax=ax_y, orientation="left", **dendrogram_kws)
    dendrogram(linkage_mat, ax=ax_y, orientation="left", **dendrogram_kws)
    ax_y.axis("off")
    ax_y.invert_yaxis()
    return ax


def probplot(
    prob_df,
    ax=None,
    title=None,
    log_scale=False,
    cmap="Purples",
    vmin=None,
    vmax=None,
    figsize=(10, 10),
    fmt=".0f",
    font_scale=1,
):
    cbar_kws = {"fraction": 0.08, "shrink": 0.8, "pad": 0.03}

    data = prob_df.values

    if log_scale:
        data = data + 0.001

        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        cbar_ticks = [
            math.pow(10, i)
            for i in range(
                math.floor(math.log10(data.min().min())),
                1 + math.ceil(math.log10(data.max().max())),
            )
        ]
        cbar_kws["ticks"] = cbar_ticks

    if ax is None:
        _ = plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.set_title(title, pad=30, fontsize=30)

    sns.set_context("talk", font_scale=font_scale)

    heatmap_kws = dict(
        cbar_kws=cbar_kws,
        annot=True,
        square=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fmt=fmt,
    )
    if log_scale:
        heatmap_kws["norm"] = log_norm
    if ax is not None:
        heatmap_kws["ax"] = ax

    ax = sns.heatmap(prob_df, **heatmap_kws)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return ax


def _get_block_indices(y):
    """
    y is a length n_verts vector of labels
    returns a length n_verts vector in the same order as the input
    indicates which block each node is
    """
    block_labels, block_inv, block_sizes = np.unique(
        y, return_inverse=True, return_counts=True
    )

    n_blocks = len(block_labels)
    block_inds = range(n_blocks)

    block_vert_inds = []
    for i in block_inds:
        # get the inds from the original graph
        inds = np.where(block_inv == i)[0]
        block_vert_inds.append(inds)
    return block_vert_inds, block_inds, block_inv


def _calculate_block_edgesum(graph, block_inds, block_vert_inds):
    """
    graph : input n x n graph 
    block_inds : list of length n_communities
    block_vert_inds : list of list, for each block index, gives every node in that block
    return_counts : whether to calculate counts rather than proportions
    """

    n_blocks = len(block_inds)
    block_pairs = cartprod(block_inds, block_inds)
    block_p = np.zeros((n_blocks, n_blocks))

    for p in block_pairs:
        from_block = p[0]
        to_block = p[1]
        from_inds = block_vert_inds[from_block]
        to_inds = block_vert_inds[to_block]
        block = graph[from_inds, :][:, to_inds]
        p = np.sum(block)
        p = p / block.size
        block_p[from_block, to_block] = p

    return block_p


def get_colors_hacky(true_labels, pred_labels):
    color_dict = {}
    classes = np.unique(true_labels)
    unk_ind = np.where(classes == "Unk")[0]  # hacky but it looks nice!
    purp_ind = 4
    in_purp_class = classes[purp_ind]
    classes[unk_ind] = in_purp_class
    classes[purp_ind] = "Unk"
    known_palette = sns.color_palette("tab10", n_colors=len(classes))
    for i, true_label in enumerate(classes):
        color = known_palette[i]
        color_dict[true_label] = color

    classes = np.unique(pred_labels)
    known_palette = sns.color_palette("gray", n_colors=len(classes))
    for i, pred_label in enumerate(classes):
        color = known_palette[i]
        color_dict[pred_label] = color
    return color_dict


def clustergram(
    adj,
    true_labels,
    pred_labels,
    figsize=(20, 20),
    title=None,
    color_dict=None,
    font_scale=1,
):
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    ax = ax.ravel()
    sns.set_context("talk", font_scale=font_scale)
    if color_dict is None:
        color_dict = get_colors_hacky(true_labels, pred_labels)
    sankey(
        ax[0], true_labels, pred_labels, aspect=20, fontsize=16, colorDict=color_dict
    )
    ax[0].axis("off")
    ax[0].set_title("Known class sorting", fontsize=30, pad=45)

    ax[1] = heatmap(
        adj,
        transform="simple-all",
        inner_hier_labels=pred_labels,
        cbar=False,
        sort_nodes=True,
        ax=ax[1],
        cmap="PRGn_r",
        hier_label_fontsize=16,
    )
    ax[1].set_title("Sorted heatmap", fontsize=30, pad=70)

    prob_df = get_sbm_prob(adj, pred_labels)
    block_sum_df = get_block_edgesums(adj, pred_labels, prob_df.columns.values)

    probplot(100 * prob_df, ax=ax[2], title="Connection percentage")

    probplot(block_sum_df, ax=ax[3], title="Average synapses")
    plt.suptitle(title, fontsize=40)
    return ax


def get_block_edgesums(adj, pred_labels, sort_blocks):
    block_vert_inds, block_inds, block_inv = _get_block_indices(pred_labels)
    block_sums = _calculate_block_edgesum(adj, block_inds, block_vert_inds)
    block_sums = block_sums[np.ix_(sort_blocks, sort_blocks)]
    block_sum_df = pd.DataFrame(data=block_sums, columns=sort_blocks, index=sort_blocks)
    return block_sum_df


def palplot(k, cmap="viridis", figsize=(1, 10), ax=None, start=0, stop=None):
    if isinstance(k, int):
        pal = sns.color_palette(palette=cmap, n_colors=k)
    else:
        pal = k
    k = len(pal)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    pal = np.array(pal)
    pal = pal.reshape((k, 1, 3))
    ax.imshow(pal)
    ax.xaxis.set_major_locator(plt.NullLocator())
    if stop is None:
        stop = len(pal) + start
    ax.yaxis.set_major_formatter(plt.FixedFormatter(np.arange(start, stop, dtype=int)))
    ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(k)))
    return ax


def stacked_barplot(
    category,
    subcategory,
    category_order=None,
    subcategory_order=None,
    ax=None,
    plot_proportions=False,
    palette="tab10",
    legend_ncol=6,
    bar_height=0.7,
    norm_bar_width=True,
    label_pos=None,
    horizontal_pad=0.02,
    return_data=False,
    color_dict=None,
    hatch_dict=None,
    return_order=False,
    plot_names=False,
    text_color="black",
):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    if isinstance(category, pd.Series):
        category = category.values
    if isinstance(subcategory, pd.Series):
        subcategory = subcategory.values

    # Counts the nunmber of unique category within each category, plotting as bar plot
    if category_order is None:
        uni_cat = np.unique(category)
    else:
        uni_cat = np.array(category_order)
    if subcategory_order is None:
        uni_subcat = np.unique(subcategory)
    else:
        uni_subcat = np.array(subcategory_order)
    if color_dict == "class":
        color_dict = CLASS_COLOR_DICT

    # stolen from mpl docs
    # HACK this could be one line in pandas
    counts_by_label = []
    for label in uni_cat:
        inds = np.where(category == label)
        subcat_in_cat = subcategory[inds]
        counts_by_class = []
        for c in uni_subcat:
            num_class_in_cluster = len(np.where(subcat_in_cat == c)[0])
            counts_by_class.append(num_class_in_cluster)
        counts_by_label.append(counts_by_class)
    results = dict(zip(uni_cat, counts_by_label))
    data = np.array(list(results.values()))

    # order things sensibly
    if category_order is None:
        simdata = data / data.sum(axis=0)[np.newaxis, :]  # normalize counts per class
        # maybe the cos dist is redundant here
        Z = linkage(simdata, method="average", metric="cosine")
        R = dendrogram(Z, truncate_mode=None, get_leaves=True, no_plot=True)
        order = R["leaves"]
        uni_cat = uni_cat[order]
        data = data[order, :]
    else:
        order = category_order
    labels = uni_cat

    # find the width of the bars
    sums = data.sum(axis=1)
    if norm_bar_width:
        norm_data = data / data.sum(axis=1)[:, np.newaxis]
    else:
        norm_data = data.copy()
    data_cum = norm_data.cumsum(axis=1)

    if color_dict is not None:
        subcategory_colors = []
        for sc in uni_subcat:
            subcategory_colors.append(color_dict[sc])
    else:
        if isinstance(palette, str):
            subcategory_colors = sns.color_palette(palette, n_colors=len(uni_subcat))
        else:
            subcategory_colors = palette

    if hatch_dict is None:
        hatch_dict = dict(zip(uni_subcat, len(uni_subcat) * [""]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_ylim(-1, len(labels))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    max_size = np.sum(norm_data, axis=1).max()

    # add just a little bit of space on either side of end of bars
    ax.set_xlim(0 - horizontal_pad * max_size, max_size * (1 + horizontal_pad))

    new_labels = []
    for i, l in enumerate(labels):
        new_labels.append(str(labels[i]) + f" ({sums[i]})")
    labels = np.array(new_labels)

    # ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(0, len(labels))))
    # ax.yaxis.set_major_formatter(plt.FixedFormatter(labels))
    ax.set_yticklabels(labels, color=text_color)
    # # print(labels)
    # ax.set_yticks())
    # print(ax.get_yticklabels(which="both"))
    # for t in ax.get_yticklabels():
    #     print(t)

    for i, (colname, color) in enumerate(zip(uni_subcat, subcategory_colors)):
        widths = norm_data[:, i]
        starts = data_cum[:, i] - widths

        if label_pos is None:
            label_pos = labels

        hatch = hatch_dict[colname]
        if hatch != "":
            alpha = 0.3
        else:
            alpha = 1
        ax.barh(
            label_pos,
            widths,
            tick_label=labels,
            left=starts,
            height=bar_height,
            label=colname,
            color=color,
            hatch=hatch,
            alpha=alpha,
        )

        # this puts small proportion numbers above bar segments
        # tries to offset them so they are readable
        # doesn't work great
        if plot_proportions:
            xcenters = starts + widths / 2
            # r, g, b = color
            text_color = "black"
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if c != 0:
                    ax.text(
                        x,
                        y - bar_height / 2 - i % 2 * bar_height / 4,
                        f"{c:.2f}",
                        ha="center",
                        va="bottom",
                        color=text_color,
                    )

        if plot_names:
            xcenters = starts + widths / 2
            # r, g, b = color
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if c != 0:
                    ax.text(
                        x,
                        y + bar_height * 0.6,
                        # - bar_height / 10,  # - i % 2 * bar_height / 4,
                        colname,
                        ha="center",
                        va="bottom",
                        color=text_color,
                        rotation=90,
                        fontsize=10,
                    )

    # this places the legend outside of the axes and in the lower left corner
    ax.legend(
        ncol=legend_ncol, bbox_to_anchor=(0, 0), loc="upper left", fontsize="small"
    )
    remove_spines(ax)
    if return_data:
        return ax, data, uni_subcat, subcategory_colors, order
    else:
        return ax


def barplot_text(
    category,
    subcategory,
    category_order=None,
    subcategory_order=None,
    ax=None,
    plot_proportions=False,
    legend_ncol=6,
    bar_height=0.7,
    norm_bar_width=True,
    label_pos=None,
    horizontal_pad=0.02,
    return_data=False,
    show_props=True,
    print_props=True,
    text_pad=0.01,
    inverse_memberships=True,
    figsize=(24, 24),
    title=None,
    palette=cc.glasbey_light,
    color_dict=None,
    hatch_dict=None,
    return_order=False,
    **kws,
):
    uni_class_labels, uni_class_counts = np.unique(subcategory, return_counts=True)
    uni_pred_labels, uni_pred_counts = np.unique(category, return_counts=True)

    # set up the figure
    fig, axs = plt.subplots(
        1, 2, figsize=figsize, sharey=True, gridspec_kw={"wspace": 0.01}
    )
    r = fig.canvas.get_renderer()

    # title the plot
    plt.suptitle(title, y=0.92, fontsize=30, x=0.5)

    # plot the barplot (and ticks to the right of them)
    ax = axs[0]
    ax, prop_data, uni_class, subcategory_colors, order = stacked_barplot(
        category,
        subcategory,
        category_order=category_order,
        subcategory_order=subcategory_order,
        ax=ax,
        plot_proportions=plot_proportions,
        palette=palette,
        legend_ncol=legend_ncol,
        bar_height=0.9,
        norm_bar_width=norm_bar_width,
        label_pos=label_pos,
        horizontal_pad=0,
        return_data=True,
        color_dict=color_dict,
        hatch_dict=hatch_dict,
        **kws,
    )
    ax.set_frame_on(False)
    k = len(uni_pred_labels)
    if norm_bar_width:
        ax1_title = f"Cluster proportion of known cell types (k={k})"
    else:
        ax1_title = f"Cluster counts by known cell types (k={k})"

    ax1_title = ax.set_title(ax1_title, pad=0)
    transformer = ax.transData.inverted()
    bbox = ax1_title.get_window_extent(renderer=r)
    bbox_points = bbox.get_points()
    out_points = transformer.transform(bbox_points)
    xlim = ax.get_xlim()
    ax.text(
        xlim[0],
        out_points[0][1],
        "Cluster name (size)",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ticks = ax.get_yticks()

    # plot the cluster compositions as text to the right of the bars
    # gs0.update(right=0.4)
    # ax2 = fig.add_subplot(gs1[0], sharey=ax0)
    ax = axs[1]
    ax.axis("off")
    # gs1.update(left=0.48)

    text_kws = {
        "verticalalignment": "center",
        "horizontalalignment": "left",
        "fontsize": 12,
        "alpha": 1,
        "weight": "bold",
    }

    ax.set_xlim((0, 1))
    transformer = ax.transData.inverted()

    cluster_sizes = prop_data.sum(axis=1)
    for i, y in enumerate(ticks):
        x = 0
        for j, (colname, color) in enumerate(zip(uni_class, subcategory_colors)):
            prop = prop_data[i, j]
            if prop > 0:
                if inverse_memberships:
                    prop = prop / uni_class_counts[j]
                    name = f"{colname} ({prop:3.0%})"
                else:
                    if print_props:
                        name = f"{colname} ({prop / cluster_sizes[i]:3.0%})"
                    else:
                        name = f"{colname} ({prop})"
                text = ax.text(x, y, name, color=color, **text_kws)
                bbox = text.get_window_extent(renderer=r)
                bbox_points = bbox.get_points()
                out_points = transformer.transform(bbox_points)
                width = out_points[1][0] - out_points[0][0]
                x += width + text_pad

    # deal with title for the last plot column based on options
    if inverse_memberships:
        ax2_title = "Known cell type (percentage of cell type in cluster)"
    else:
        if print_props:
            ax2_title = "Known cell type (percentage of cluster)"
        else:
            ax2_title = "Known cell type (count in cluster)"
    ax.set_title(ax2_title, loc="left", pad=-100)
    if return_order:
        return fig, axs, order
    else:
        return fig, axs


def bartreeplot(
    dc,
    class_labels,
    pred_labels,
    show_props=True,
    print_props=True,
    text_pad=0.01,
    inverse_memberships=True,
    figsize=(24, 23),
    title=None,
    palette=cc.glasbey_light,
    color_dict=None,
):
    # gather necessary info from model
    linkage, labels = dc.build_linkage(bic_distance=False)  # hackily built like scipy's
    uni_class_labels, uni_class_counts = np.unique(class_labels, return_counts=True)
    uni_pred_labels, uni_pred_counts = np.unique(pred_labels, return_counts=True)

    # set up the figure
    fig = plt.figure(figsize=figsize)
    r = fig.canvas.get_renderer()
    gs0 = plt.GridSpec(1, 2, figure=fig, width_ratios=[0.2, 0.8], wspace=0)
    gs1 = plt.GridSpec(1, 1, figure=fig, width_ratios=[0.2], wspace=0.1)

    # title the plot
    plt.suptitle(title, y=0.92, fontsize=30, x=0.5)

    # plot the dendrogram
    ax0 = fig.add_subplot(gs0[0])

    dendr_data = dendrogram(
        linkage,
        orientation="left",
        labels=labels,
        color_threshold=0,
        above_threshold_color="k",
        ax=ax0,
    )
    ax0.axis("off")
    ax0.set_title("Dendrogram", loc="left")

    # get the ticks from the dendrogram to apply to the bar plot
    ticks = ax0.get_yticks()

    # plot the barplot (and ticks to the right of them)
    leaf_names = np.array(dendr_data["ivl"])[::-1]
    ax1 = fig.add_subplot(gs0[1], sharey=ax0)
    ax1, prop_data, uni_class, subcategory_colors = stacked_barplot(
        pred_labels,
        class_labels,
        label_pos=ticks,
        category_order=leaf_names,
        ax=ax1,
        bar_height=5,
        horizontal_pad=0,
        palette=palette,
        norm_bar_width=show_props,
        return_data=True,
        color_dict=color_dict,
    )
    ax1.set_frame_on(False)
    ax1.yaxis.tick_right()

    if show_props:
        ax1_title = "Cluster proportion of known cell types"
    else:
        ax1_title = "Cluster counts by known cell types"

    ax1_title = ax1.set_title(ax1_title, loc="left")
    transformer = ax1.transData.inverted()
    bbox = ax1_title.get_window_extent(renderer=r)
    bbox_points = bbox.get_points()
    out_points = transformer.transform(bbox_points)
    xlim = ax1.get_xlim()
    ax1.text(
        xlim[1], out_points[0][1], "Cluster name (size)", verticalalignment="bottom"
    )

    # plot the cluster compositions as text to the right of the bars
    gs0.update(right=0.4)
    ax2 = fig.add_subplot(gs1[0], sharey=ax0)
    ax2.axis("off")
    gs1.update(left=0.48)

    text_kws = {
        "verticalalignment": "center",
        "horizontalalignment": "left",
        "fontsize": 12,
        "alpha": 1,
        "weight": "bold",
    }

    ax2.set_xlim((0, 1))
    transformer = ax2.transData.inverted()

    cluster_sizes = prop_data.sum(axis=1)
    for i, y in enumerate(ticks):
        x = 0
        for j, (colname, color) in enumerate(zip(uni_class, subcategory_colors)):
            prop = prop_data[i, j]
            if prop > 0:
                if inverse_memberships:
                    prop = prop / uni_class_counts[j]
                    name = f"{colname} ({prop:3.0%})"
                else:
                    if print_props:
                        name = f"{colname} ({prop / cluster_sizes[i]:3.0%})"
                    else:
                        name = f"{colname} ({prop})"
                text = ax2.text(x, y, name, color=color, **text_kws)
                bbox = text.get_window_extent(renderer=r)
                bbox_points = bbox.get_points()
                out_points = transformer.transform(bbox_points)
                width = out_points[1][0] - out_points[0][0]
                x += width + text_pad

    # deal with title for the last plot column based on options
    if inverse_memberships:
        ax2_title = "Known cell type (percentage of cell type in cluster)"
    else:
        if print_props:
            ax2_title = "Known cell type (percentage of cluster)"
        else:
            ax2_title = "Known cell type (count in cluster)"
    ax2.set_title(ax2_title, loc="left")
    axs = (ax0, ax1, ax2)
    return fig, axs, leaf_names


def get_colors(labels, pal=cc.glasbey_light, to_int=False, color_dict=None):
    uni_labels = np.unique(labels)
    if to_int:
        uni_labels = [int(i) for i in uni_labels]
    if color_dict is None:
        color_dict = get_color_dict(labels, pal=pal, to_int=to_int)
    colors = np.array(itemgetter(*labels)(color_dict))
    return colors


def get_color_dict(labels, pal="tab10", to_int=False):
    uni_labels = np.unique(labels)
    if to_int:
        uni_labels = [int(i) for i in uni_labels]
    if isinstance(pal, str):
        pal = sns.color_palette(pal, n_colors=len(uni_labels))
    color_dict = dict(zip(uni_labels, pal))
    return color_dict


def gridmap(A, ax=None, legend=False, sizes=(5, 10), spines=False, border=True, **kws):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 20))
    n_verts = A.shape[0]
    inds = np.nonzero(A)
    edges = A[inds]
    scatter_df = pd.DataFrame()
    scatter_df["weight"] = edges
    scatter_df["x"] = inds[1]
    scatter_df["y"] = inds[0]
    ax = sns.scatterplot(
        data=scatter_df,
        x="x",
        y="y",
        size="weight",
        legend=legend,
        sizes=sizes,
        ax=ax,
        linewidth=0,
        **kws,
    )
    # ax.axis("image")
    ax.set_xlim((-1, n_verts + 1))
    ax.set_ylim((n_verts + 1, -1))
    if not spines:
        remove_spines(ax)
    if border:
        linestyle_kws = {
            "linestyle": "--",
            "alpha": 0.5,
            "linewidth": 0.5,
            "color": "grey",
        }
        ax.axvline(0, **linestyle_kws)
        ax.axvline(n_verts, **linestyle_kws)
        ax.axhline(0, **linestyle_kws)
        ax.axhline(n_verts, **linestyle_kws)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.axis("off")
    return ax


def remove_spines(ax, keep_corner=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_corner:
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


def distplot(
    data,
    labels=None,
    direction="out",
    title="",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    palette="Set1",
    xlabel="",
    ylabel="Density",
):

    plt.figure(figsize=figsize)
    ax = plt.gca()
    palette = sns.color_palette(palette)
    plt_kws = {"cumulative": True}
    with sns.plotting_context(context=context, font_scale=font_scale):
        if labels is not None:
            categories, counts = np.unique(labels, return_counts=True)
            for i, cat in enumerate(categories):
                cat_data = data[np.where(labels == cat)]
                if counts[i] > 1 and cat_data.min() != cat_data.max():
                    x = np.sort(cat_data)
                    y = np.arange(len(x)) / float(len(x))
                    plt.plot(x, y, label=cat, color=palette[i])
                else:
                    ax.axvline(cat_data[0], label=cat, color=palette[i])
            plt.legend()
        else:
            if data.min() != data.max():
                sns.distplot(data, hist=False, kde_kws=plt_kws)
            else:
                ax.axvline(data[0])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return ax


def draw_networkx_nice(
    g,
    x_pos,
    y_pos,
    sizes=None,
    colors=None,
    nodelist=None,
    cmap="Blues",
    ax=None,
    x_boost=0,
    y_boost=0,
    draw_axes_arrows=False,
    vmin=None,
    vmax=None,
    weight_scale=1,
    size_scale=1,
    draw_labels=True,
    font_size=14,
):
    if nodelist is None:
        nodelist = g.nodes()
    weights = nx.get_edge_attributes(g, "weight")

    x_attr_dict = nx.get_node_attributes(g, x_pos)
    y_attr_dict = nx.get_node_attributes(g, y_pos)

    pos = {}
    label_pos = {}
    for n in nodelist:
        pos[n] = (x_attr_dict[n], y_attr_dict[n])
        label_pos[n] = (x_attr_dict[n] + x_boost, y_attr_dict[n] + y_boost)

    if sizes is not None:
        size_attr_dict = nx.get_node_attributes(g, sizes)
        node_size = []
        for n in nodelist:
            node_size.append(size_scale * size_attr_dict[n])

    if colors is not None:
        color_attr_dict = nx.get_node_attributes(g, colors)
        node_color = []
        for n in nodelist:
            node_color.append(color_attr_dict[n])

    weight_array = np.array(list(weights.values()))
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    # norm = mplc.Normalize(vmin=0, vmax=weight_array.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cmap = sm.to_rgba(weight_array)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), frameon=False)

    node_collection = nx.draw_networkx_nodes(
        g, pos, node_color=node_color, node_size=node_size, with_labels=False, ax=ax
    )
    n_squared = len(nodelist) ** 2  # maximum z-order so far
    node_collection.set_zorder(n_squared)

    edgelist = list(g.edges(data=True))
    weights = []
    for edge in edgelist:
        weight = edge[2]["weight"]
        weights.append(weight)
    weights = np.array(weights)

    lc = nx.draw_networkx_edges(
        g,
        pos,
        edgelist=edgelist,
        edge_color=cmap,
        width=weight_scale * weights,
        connectionstyle="arc3,rad=0.2",
        arrows=True,
        # width=1.5,
        ax=ax,
    )

    # set z-order by weight
    weight_inds = np.argsort(weights)
    weight_rank = np.argsort(weight_inds)
    for i, l in enumerate(lc):
        l.set_zorder(weight_rank[i])

    if draw_labels:
        text_items = nx.draw_networkx_labels(g, label_pos, ax=ax, font_size=font_size)

        # make sure the labels are above all in z order
        for _, t in text_items.items():
            t.set_zorder(n_squared + 1)

    ax.set_xlabel(x_pos)
    ax.set_ylabel(y_pos)
    return ax


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


def add_connections(x1, x2, y1, y2, color="black", alpha=0.3, linewidth=0.3, ax=None):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    if ax is None:
        ax = plt.gca()
    for i in range(len(x1)):
        ax.plot(
            [x1[i], x2[i]],
            [y1[i], y2[i]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def remove_axis(ax):
    remove_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])


def set_style(
    spine_right=False,
    spine_top=False,
    spine_left=True,
    spine_bottom=True,
    axes_edgecolor="lightgrey",
    tick_color="grey",
    axes_labelcolor="dimgrey",
    text_color="dimgrey",
    context="talk",
    tick_size=0,
    font_scale=1,
):
    # "axes.formatter.limits": (-3, 3),
    # "figure.figsize": (6, 3),
    # "figure.dpi": 100,
    rc_dict = {
        "axes.spines.right": spine_right,
        "axes.spines.top": spine_top,
        "axes.spines.left": spine_left,
        "axes.spines.bottom": spine_bottom,
        "axes.edgecolor": axes_edgecolor,
        "ytick.color": tick_color,
        "xtick.color": tick_color,
        "axes.labelcolor": axes_labelcolor,
        "text.color": text_color,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "xtick.major.size": tick_size,
        "ytick.major.size": tick_size,
    }

    for key, val in rc_dict.items():
        mpl.rcParams[key] = val
    context = sns.plotting_context(context=context, font_scale=font_scale, rc=rc_dict)
    sns.set_context(context)
