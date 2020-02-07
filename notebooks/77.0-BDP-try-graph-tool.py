# %% [markdown]
# #
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from graph_tool import load_graph
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool.draw import graph_draw

from scipy.cluster.hierarchy import dendrogram, linkage

from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.visualization import barplot_text

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

# mpl.use("TkAgg")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


graph_type = "Gad"
version = "2020-01-29"
mg = load_metagraph(graph_type, version=version)
temp_loc = f"maggot_models/data/interim/{graph_type}.graphml"

thresholds = np.arange(3, 4, 1)

for threshold in thresholds:
    # simple threshold
    adj = mg.adj.copy()
    adj[adj <= 0] = 0.0
    meta = mg.meta.copy()
    meta = pd.DataFrame(mg.meta[" neuron_name"])

    mg = MetaGraph(adj, meta)

    # save to temp
    nx.write_graphml(mg.g, temp_loc)

    # load into graph-tool from temp
    g = load_graph(temp_loc, fmt="graphml")
    total_degrees = g.get_total_degrees(g.get_vertices())
    remove_verts = np.where(total_degrees == 0)[0]
    g.remove_vertex(remove_verts)
    props = g.vertex_properties
    min_state = minimize_blockmodel_dl(g, verbose=True)

    blocks = list(min_state.get_blocks())
    verts = g.get_vertices()

    block_map = {}

    for v, b in zip(verts, blocks):
        cell_id = int(g.vertex_properties["_graphml_vertex_id"][v])
        block_map[cell_id] = b

    block_series = pd.Series(block_map)
    block_series.name = "block_label"

    mg = load_metagraph(graph_type, version=version)
    mg.meta = pd.concat((mg.meta, block_series), axis=1)
    mg.meta["Original index"] = range(len(mg.meta))
    keep_inds = mg.meta[~mg.meta["block_label"].isna()]["Original index"].values
    mg.reindex(keep_inds)
    mg.verify(10000, graph_type=graph_type, version=version)

    category = mg["block_label"]
    subcategory = mg["Merge Class"]
    uni_cat = np.unique(category)
    uni_subcat = np.unique(subcategory)

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
    labels = list(results.keys())
    data = np.array(list(results.values()))

    data = data / data.sum(axis=0)[np.newaxis, :]  # normalize counts per class

    # maybe the cos dist is redundant here
    Z = linkage(data, method="average", metric="cosine", optimal_ordering=True)

    R = dendrogram(Z, truncate_mode=None, get_leaves=True, no_plot=True)

    savename = f"{graph_type}-mcmc-t{threshold}-counts"
    title = f"{graph_type}, threshold = {hash(str(threshold))}"

    barplot_text(
        mg["block_label"],
        mg["Merge Class"],
        category_order=R["leaves"],
        norm_bar_width=False,
        title=title,
    )
    stashfig(savename)

    barplot_text(
        mg["block_label"],
        mg["Merge Class"],
        category_order=R["leaves"],
        norm_bar_width=True,
        title=title,
    )
    stashfig(savename)

# %% [markdown]
# #

from graph_tool.draw import fruchterman_reingold_layout, arf_layout, radial_tree_layout

# graph_draw(g)

# pos = radial_tree_layout(g)

# min_state.draw(pos=pos)

# %% [markdown]
# #

from graph_tool.inference import minimize_nested_blockmodel_dl

min_state = minimize_nested_blockmodel_dl(g, verbose=True)

# %% [markdown]
# #
# from graph_tool.draw import draw_hierarchy

min_state.draw()
