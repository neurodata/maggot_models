#%%
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import curdoc, figure, show
from mpl_toolkits.axes_grid1 import make_axes_locatable

from graspy.plot import degreeplot, edgeplot, heatmap
from src.data import load_networkx
from src.utils import meta_to_array, savefig
from bokeh.embed import file_html


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

sort_inds = _sort_inds(adj, classes, np.ones_like(classes), True)
adj = _sort_graph(adj, classes, np.ones_like(classes), True)
classes = classes[sort_inds]
cell_ids = cell_ids[sort_inds]

xs, ys = np.meshgrid(range(adj.shape[1]), range(adj.shape[0]))
nonzero_inds = np.nonzero(adj.ravel())
x = xs.ravel()[nonzero_inds]
y = ys.ravel()[nonzero_inds]
weights = adj.ravel()[nonzero_inds]

names = meta_to_array(graph, "Name")
x_names = names[x]
y_names = names[y]
x_cell_type = classes[x]
y_cell_type = classes[y]

# x_grouped = list(zip(x_cell_type, x))

inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum = _get_freqs(
    classes, np.ones_like(classes)
)
inner_unique, _ = _unique_like(classes)

from bokeh.models import Span

borders = []
for b in inner_freq_cumsum[0:-1]:
    borders.append(b)
    # ax.vlines(x, 0, graph.shape[0] + 1)


TOOLS = "pan,reset,zoom_in,zoom_out,box_zoom"
WEIGHT_SCALE = 16
# hover.tooltips = [
#     ("index", "$index"),
#     ("(x,y)", "($x, $y)"),
#     ("radius", "@radius"),
#     ("fill color", "$color[hex, swatch]:fill_color"),
#     ("foo", "@foo"),
#     ("bar", "@bar"),
# ]
scatter_tooltips = [("from", "@from"), ("to", "@to"), ("weight", "@weight")]
scatter_data = {
    "x": x,
    "y": y,
    "weight": weights,
    "from": y_names,
    "to": x_names,
    "radius": WEIGHT_SCALE * weights,
}
scatter_source = ColumnDataSource(scatter_data)

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
    # x_range=FactorRange(*x_grouped),
)


scatter_fig.scatter(
    x="x", y="y", radius="radius", source=scatter_source, color="#3A5785", alpha=0.6
)
scatter_fig.xgrid.visible = False
scatter_fig.ygrid.visible = False

span_kws = dict(line_color="red", line_dash="dashed", line_width=0.5)

for b in borders:
    s = Span(location=b, dimension="height", **span_kws)
    scatter_fig.add_layout(s)
    s = Span(location=b, dimension="width", **span_kws)
    scatter_fig.add_layout(s)
##

input_sums = adj.sum(axis=0)

top_bar_tooltips = [("name", "@name"), ("input_sum", "@input_sum")]

top_bar_data = {
    "x": range(adj.shape[1]),
    "y": y,
    "weight": weights,
    "from": y_names,
    "to": x_names,
    "radius": WEIGHT_SCALE * weights,
    "input_sum": input_sums,
    "name": names,
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
output_sums = adj.sum(axis=1)

side_bar_tooltips = [("name", "@name"), ("output_sum", "@output_sum")]

side_bar_data = {"y": range(adj.shape[1]), "output_sum": output_sums, "name": names}
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
from bokeh.resources import CDN

html = file_html(layout, CDN, "test_adj")
with open("test_adj.html", "w") as f:
    f.write(html)

# output_file("bars.html")

# fruits = ["Apples", "Pears", "Nectarines", "Plums", "Grapes", "Strawberries"]
# years = ["2015", "2016", "2017"]

# data = {
#     "fruits": fruits,
#     "2015": [2, 1, 4, 3, 2, 4],
#     "2016": [5, 3, 3, 2, 4, 6],
#     "2017": [3, 2, 4, 4, 5, 3],
# }

# # this creates [ ("Apples", "2015"), ("Apples", "2016"), ("Apples", "2017"), ("Pears", "2015), ... ]
# x = [(fruit, year) for fruit in fruits for year in years]
# counts = sum(zip(data["2015"], data["2016"], data["2017"]), ())  # like an hstack

# source = ColumnDataSource(data=dict(x=x, counts=counts))

# p = figure(
#     x_range=FactorRange(*x),
#     plot_height=250,
#     title="Fruit Counts by Year",
#     toolbar_location=None,
#     tools="",
# )

# p.vbar(x="x", top="counts", width=0.9, source=source)

# p.y_range.start = 0
# p.x_range.range_padding = 0.1
# p.xaxis.major_label_orientation = 1
# p.xgrid.grid_line_color = None


#%%
