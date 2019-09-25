#%% Load data
import networkx as nx
import numpy as np
import pandas as pd

from graspy.plot import degreeplot, edgeplot, gridplot, heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.data import load_networkx
from src.utils import meta_to_array

#%%
graph_type = "Gn"

graph = load_networkx(graph_type)

df_adj = nx.to_pandas_adjacency(graph)
adj = df_adj.values

classes = meta_to_array(graph, "Class")
print(np.unique(classes))

nx_ids = np.array(list(graph.nodes()), dtype=int)
df_ids = df_adj.index.values.astype(int)
df_adj.index = df_ids
df_adj.columns = df_ids
np.array_equal(nx_ids, df_ids)
cell_ids = df_ids


#%% Map MW classes to the indices of cells belonging to them
unique_classes, inverse_classes = np.unique(classes, return_inverse=True)
class_ind_map = {}
class_ids_map = {}
for i, class_name in enumerate(unique_classes):
    inds = np.where(inverse_classes == i)[0]
    ids = cell_ids[inds]
    class_ind_map[class_name] = inds
    class_ids_map[class_name] = ids
class_ind_map


#%%
plt.figure(figsize=(10, 5))
sns.distplot(adj.sum(axis=0))
plt.xlabel("Proportion of input w/in current graph")
plt.ylabel("Frequency")


degreeplot(adj)


#%%
# node = str(df_ids[class_ids_map["ORN mPNs"][0]])
# neighbor_graph = nx.ego_graph(graph, node)
# neighbor_names = meta_to_array(neighbor_graph, "Class")
# neighbor_nodes = list(neighbor_graph.nodes())
# labels = dict(zip(neighbor_nodes, neighbor_names))
# plt.figure(figsize=(20, 20))
# nx.draw_networkx(neighbor_graph, labels=labels)
# plt.show()

#%%
def proportional_search(adj, class_ind_map, or_classes, ids, thresh):
    """finds the cell ids of neurons who receive a certain proportion of their 
    input from one of the cells in or_classes 
    
    Parameters
    ----------
    adj : np.array
        adjacency matrix, assumed to be normalized so that columns sum to 1
    class_map : dict
        keys are class names, values are arrays of indices describing where that class
        can be found in the adjacency matrix
    or_classes : list 
        which classes to consider for the input thresholding. Neurons will be selected 
        which satisfy ANY of the input threshold criteria
    ids : np.array
        names of each cell 
    """

    pred_cell_ids = []
    for i, class_name in enumerate(or_classes):
        inds = class_ind_map[class_name]  # indices for neurons of that class
        from_class_adj = adj[inds, :]  # select the rows corresponding to that class
        prop_input = from_class_adj.sum(axis=0)  # sum input from that class
        # prop_input /= adj.sum(axis=0)
        flag_inds = np.where(prop_input >= thresh[i])[0]  # inds above threshold
        pred_cell_ids += list(ids[flag_inds])  # append to cells which satisfied

    pred_cell_ids = np.unique(pred_cell_ids)

    return pred_cell_ids


pn_types = ["ORN mPNs", "ORN uPNs", "tPNs", "vPNs"]
lhn_thresh = [0.05, 0.05, 0.05, 0.05]

pred_lhn_ids = proportional_search(adj, class_ind_map, pn_types, df_ids, lhn_thresh)

true_lhn_inds = np.concatenate((class_ind_map["LHN"], class_ind_map["LHN; CN"]))
true_lhn_ids = df_ids[true_lhn_inds]

print("LHN")
print("Recall:")
print(np.isin(true_lhn_ids, pred_lhn_ids).mean())  # how many of the og lhn i got
print("Precision:")
print(np.isin(pred_lhn_ids, true_lhn_ids).mean())  # this is how many of mine are in og
print(len(pred_lhn_ids))

my_wrong_lhn_ids = np.setdiff1d(pred_lhn_ids, true_lhn_ids)
#%%


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
    # # side inner curves
    # ax_y = divider.new_horizontal(size="5%", pad=0.0, pack_start=True)
    # ax.figure.add_axes(ax_y)
    # _plot_brackets(
    #     ax_y,
    #     np.tile(inner_unique, len(outer_unique)),
    #     inner_tick_loc,
    #     inner_tick_width,
    #     curve,
    #     "inner",
    #     "y",
    #     n_verts,
    #     fontsize,
    # )

    # if plot_outer:
    #     # top outer curves
    #     pad_scalar = 0.35 / 30 * fontsize
    #     ax_x2 = divider.new_vertical(size="5%", pad=pad_scalar, pack_start=False)
    #     ax.figure.add_axes(ax_x2)
    #     _plot_brackets(
    #         ax_x2,
    #         outer_unique,
    #         outer_tick_loc,
    #         outer_tick_width,
    #         curve,
    #         "outer",
    #         "x",
    #         n_verts,
    #         fontsize,
    #     )
    #     # side outer curves
    #     ax_y2 = divider.new_horizontal(size="5%", pad=pad_scalar, pack_start=True)
    #     ax.figure.add_axes(ax_y2)
    #     _plot_brackets(
    #         ax_y2,
    #         outer_unique,
    #         outer_tick_loc,
    #         outer_tick_width,
    #         curve,
    #         "outer",
    #         "y",
    #         n_verts,
    #         fontsize,
    #     )
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


# proj_class = "ORN mPNs"
pn_types = ["ORN mPNs", "ORN uPNs", "tPNs", "vPNs"]
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import savefig

for proj_class in pn_types:
    sort_inds = _sort_inds(adj, classes, np.ones_like(classes), True)
    sort_adj = _sort_graph(adj, classes, np.ones_like(classes), True)
    sort_classes = classes[sort_inds]
    proj_inds = np.where(sort_classes == proj_class)[0]
    clipped_adj = sort_adj[proj_inds, :]

    plt.figure(figsize=(30, 10))
    # pn_graph = df_adj.loc[class_ids_map["vPNs"], :].values
    xs, ys = np.meshgrid(
        range(1, clipped_adj.shape[1] + 1), range(1, clipped_adj.shape[0] + 1)
    )
    nonzero_inds = np.nonzero(clipped_adj.ravel())
    x = xs.ravel()[nonzero_inds]
    y = ys.ravel()[nonzero_inds]
    weights = clipped_adj.ravel()[nonzero_inds]
    ax = sns.scatterplot(x=x, y=y, size=weights, legend=False)

    plt.ylabel(proj_class)
    plt.title(proj_class, pad=100)

    divider = make_axes_locatable(ax)
    ax_top = divider.new_vertical(size="25%", pad=0.0, pack_start=False)
    ax.figure.add_axes(ax_top)
    sums = clipped_adj.sum(axis=0)
    # sums /= sums.max()
    # sums = sums[sort_inds]
    ax_top.bar(range(1, clipped_adj.shape[1] + 1), sums, width=10)
    ax_top.set_xlim((0, clipped_adj.shape[1]))
    ax_top.axis("off")
    ax_top.hlines(0.05, 0, clipped_adj.shape[1] + 1, color="r", linestyle="--")

    ax = _plot_groups(
        ax, divider, clipped_adj, sort_inds, classes, outer_labels=None, fontsize=10
    )
    ax.set_xlim((0, clipped_adj.shape[1]))
    ax.set_ylim((0, clipped_adj.shape[0]))
    ax.axis("off")
    savefig(proj_class + "_to_all_marginals")

# #%%
# my_classes = classes.copy()
# wrong_inds = np.isin(cell_ids, my_wrong_lhn_ids)
# my_classes[wrong_inds] = "LHN"

# for proj_class in pn_types:
#     sort_inds = _sort_inds(adj, my_classes, np.ones_like(my_classes), True)
#     sort_adj = _sort_graph(adj, my_classes, np.ones_like(my_classes), True)
#     sort_classes = my_classes[sort_inds]
#     proj_inds = np.where(sort_classes == proj_class)[0]
#     clipped_adj = sort_adj[proj_inds, :]

#     plt.figure(figsize=(30, 10))
#     # pn_graph = df_adj.loc[class_ids_map["vPNs"], :].values
#     xs, ys = np.meshgrid(
#         range(1, clipped_adj.shape[1] + 1), range(1, clipped_adj.shape[0] + 1)
#     )
#     nonzero_inds = np.nonzero(clipped_adj.ravel())
#     x = xs.ravel()[nonzero_inds]
#     y = ys.ravel()[nonzero_inds]
#     weights = clipped_adj.ravel()[nonzero_inds]
#     ax = sns.scatterplot(x=x, y=y, size=weights, legend=False)

#     plt.ylabel(proj_class)
#     plt.title(proj_class, pad=100)

#     divider = make_axes_locatable(ax)
#     ax_top = divider.new_vertical(size="25%", pad=0.0, pack_start=False)
#     ax.figure.add_axes(ax_top)
#     sums = clipped_adj.sum(axis=0)
#     # sums /= sums.max()
#     # sums = sums[sort_inds]
#     ax_top.bar(range(1, clipped_adj.shape[1] + 1), sums, width=10)
#     ax_top.set_xlim((0, clipped_adj.shape[1]))
#     ax_top.axis("off")
#     ax_top.hlines(0.05, 0, clipped_adj.shape[1] + 1, color="r", linestyle="--")

#     ax = _plot_groups(
#         ax, divider, clipped_adj, sort_inds, my_classes, outer_labels=None, fontsize=10
#     )
#     ax.set_xlim((0, clipped_adj.shape[1]))
#     ax.set_ylim((0, clipped_adj.shape[0]))
#     ax.axis("off")


#%%

