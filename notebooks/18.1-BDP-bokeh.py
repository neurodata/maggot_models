#%%
from operator import itemgetter

import networkx as nx
import numpy as np

import pandas as pd
from bokeh.embed import file_html
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, FactorRange, Legend, Span

from bokeh.palettes import Spectral4, all_palettes
from bokeh.plotting import curdoc, figure, output_file, show
from bokeh.resources import CDN
from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT
from graspy.plot import degreeplot, edgeplot, heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.data import load_everything, load_networkx
from src.utils import meta_to_array, savefig


# hover.tooltips = [
#     ("index", "$index"),
#     ("(x,y)", "($x, $y)"),
#     ("radius", "@radius"),
#     ("fill color", "$color[hex, swatch]:fill_color"),
#     ("foo", "@foo"),
#     ("bar", "@bar"),
# ]
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


def adj_to_points(adj):
    xs, ys = np.meshgrid(range(adj.shape[1]), range(adj.shape[0]))
    nonzero_inds = np.nonzero(adj.ravel())
    x = xs.ravel()[nonzero_inds]
    y = ys.ravel()[nonzero_inds]
    weights = adj.ravel()[nonzero_inds]
    return list(x), list(y), list(weights)


#%%
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_VERSION = "mb_2019-09-23"
# Sort the graph appropriately first
graph_type = "G"

G_adj, class_labels, side_labels, pair_labels, id_labels = load_everything(
    graph_type,
    return_keys=["Class", "Hemisphere", "Pair"],
    return_ids=True,
    version=GRAPH_VERSION,
)

name_map = {
    "APL": "APL",
    "Gustatory PN": "PN",
    "KC 1 claw": "KC",
    "KC 2 claw": "KC",
    "KC 3 claw": "KC",
    "KC 4 claw": "KC",
    "KC 5 claw": "KC",
    "KC 6 claw": "KC",
    "KC young": "KC",
    "MBIN": "MBIN",
    "MBON": "MBON",
    "ORN mPN": "PN",
    "ORN uPN": "PN",
    "Unknown PN": "PN",
    "tPN": "PN",
    "vPN": "PN",
}
class_labels = np.array(itemgetter(*class_labels)(name_map))

side_class_labels = []
for s, c in zip(side_labels, class_labels):
    side_class_labels.append(str(s + c))
side_class_labels = np.array(side_class_labels)

sort_labels = class_labels  # the labels on which to sort the graph into blocks
sort_inds = _sort_inds(G_adj, sort_labels, side_labels, True)
G_adj = _sort_graph(G_adj, sort_labels, side_labels, True)
sort_labels = sort_labels[sort_inds]
id_labels = id_labels[sort_inds]
pair_labels = pair_labels[sort_inds]

# Turn the graph into points

x = []
y = []
weights = []
edge_type = []

for graph_type in GRAPH_TYPES:
    g = load_everything(graph_type, version=GRAPH_VERSION)
    g = g[np.ix_(sort_inds, sort_inds)]
    temp_x, temp_y, temp_weights = adj_to_points(g)
    x += temp_x
    y += temp_y
    weights += temp_weights
    edge_type += len(temp_x) * [graph_type]

x_names = id_labels[x]
y_names = id_labels[y]
x_pair_names = pair_labels[x]
y_pair_names = pair_labels[y]
x_cell_type = sort_labels[x]
y_cell_type = sort_labels[y]

x = np.array(x)
y = np.array(y)
weights = np.array(weights)
edge_type = np.array(edge_type)

inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum = _get_freqs(
    sort_labels, side_labels
)
inner_unique, _ = _unique_like(sort_labels)


borders = []
for b in inner_freq_cumsum[0:-1]:
    borders.append(b)
    # ax.vlines(x, 0, graph.shape[0] + 1)


TOOLS = "pan,reset,zoom_in,zoom_out,box_zoom"
WEIGHT_SCALE = 0.005

pal = all_palettes["Set1"][4]
colormap = dict(zip(GRAPH_TYPES, pal))
colors = np.array(itemgetter(*edge_type)(colormap))

scatter_tooltips = [
    ("from", "@from"),
    ("to", "@to"),
    ("from_pair", "@from_pair"),
    ("to_pair", "@to_pair"),
    ("weight", "@weight"),
]
scatter_data = {
    "x": x,
    "y": y,
    "weight": weights,
    "from": y_names,
    "to": x_names,
    "from_pair": y_pair_names,
    "to_pair": x_pair_names,
    "radius": WEIGHT_SCALE * weights + 0.5,
    "colors": colors,
    "label": edge_type,
}
scatter_df = pd.DataFrame(scatter_data)
scatter_source = ColumnDataSource(scatter_df)


xmin = x.min() - 1
xmax = x.max() + 1
ymin = y.min() - 1
ymax = y.max() + 1

scatter_fig = figure(
    tools=TOOLS,
    plot_width=800,
    plot_height=800,
    min_border=10,
    min_border_left=50,
    toolbar_location="above",
    x_axis_location=None,
    y_axis_location=None,
    tooltips=scatter_tooltips,
    x_range=(xmin, xmax),
    y_range=(ymax, ymin),
)

scatters = []
legend_it = []
for t in GRAPH_TYPES:
    print(t)
    temp_scatter_df = scatter_df[scatter_df["label"] == t]
    scatter_source = ColumnDataSource(temp_scatter_df)
    print(temp_scatter_df.head())
    out = scatter_fig.circle(
        x="x",
        y="y",
        radius="radius",
        source=scatter_source,
        fill_color="colors",
        alpha=0.6,
        line_color=None,
    )
    legend_it.append((t, [out]))
    scatters.append(out)

legend = Legend(items=legend_it, location=(0, ymax))
legend.click_policy = "hide"
scatter_fig.add_layout(legend, "right")

scatter_fig.xgrid.visible = False
scatter_fig.ygrid.visible = False

#


span_kws = dict(line_color="black", line_dash="dashed", line_width=0.5)

for b in borders:
    s = Span(location=b, dimension="height", **span_kws)
    scatter_fig.add_layout(s)
    s = Span(location=b, dimension="width", **span_kws)
    scatter_fig.add_layout(s)


# legend = Legend(items=edge_type, location=(0, -60))
# legend.click_policy = "mute"

# scatter_fig.add_layout(legend, "right")
show(scatter_fig)

from bokeh.resources import CDN

html = file_html(scatter_fig, CDN, "mb_adj")
with open("mb_adj.html", "w") as f:
    f.write(html)
#%%
##

input_sums = G_adj.sum(axis=0)

top_bar_tooltips = [("name", "@name"), ("input_sum", "@input_sum")]

top_bar_data = {
    "x": range(G_adj.shape[1]),
    "y": y,
    "weight": weights,
    "from": y_names,
    "to": x_names,
    "radius": WEIGHT_SCALE * weights,
    "input_sum": input_sums,
    "name": id_labels,
}
top_bar_source = ColumnDataSource(top_bar_data)
top_bar_fig = figure(
    tools=TOOLS,
    plot_width=800,
    plot_height=100,
    min_border=10,
    min_border_left=50,
    toolbar_location="above",
    x_axis_location=None,
    x_range=scatter_fig.x_range,
    tooltips=top_bar_tooltips,
)
top_bar_fig.vbar(source=top_bar_source, x="x", top="input_sum", width=0.5, bottom=0)


##
output_sums = G_adj.sum(axis=1)

side_bar_tooltips = [("name", "@name"), ("output_sum", "@output_sum")]

side_bar_data = {
    "y": range(G_adj.shape[1]),
    "output_sum": output_sums,
    "name": id_labels,
}
side_bar_source = ColumnDataSource(side_bar_data)
side_bar_fig = figure(
    tools=TOOLS,
    plot_width=100,
    plot_height=800,
    min_border=10,
    min_border_left=10,
    toolbar_location="above",
    y_axis_location=None,
    y_range=scatter_fig.y_range,
    tooltips=side_bar_tooltips,
)
side_bar_fig.hbar(source=side_bar_source, y="y", right="output_sum", height=0.5)


layout = gridplot([[top_bar_fig, None], [scatter_fig, side_bar_fig]])
show(layout)

html = file_html(layout, CDN, "test_adj")
with open("test_adj.html", "w") as f:
    f.write(html)

#%%
# output_notebook()

p = figure(
    plot_width=800, plot_height=250, x_axis_type="datetime", toolbar_location="above"
)
p.title.text = "Click on legend entries to mute the corresponding lines"

legend_it = []

for data, name, color in zip(
    [AAPL, IBM, MSFT, GOOG], ["AAPL", "IBM", "MSFT", "GOOG"], Spectral4
):
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    c = p.line(
        df["date"],
        df["close"],
        line_width=2,
        color=color,
        alpha=0.8,
        muted_color=color,
        muted_alpha=0.2,
    )
    legend_it.append((name, [c]))


legend = Legend(items=legend_it, location=(0, -60))
legend.click_policy = "mute"

p.add_layout(legend, "right")

show(p)


# %%
