# %% [markdown]
# #
import itertools
import os
import time
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import textdistance
from joblib import Parallel, delayed

from graspy.embed import AdjacencySpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import get_lcc, symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels
from src.traverse import (
    generate_random_cascade,
    generate_random_walks,
    path_to_visits,
    to_markov_matrix,
    to_path_graph,
)
from src.visualization import (
    CLASS_COLOR_DICT,
    barplot_text,
    draw_networkx_nice,
    remove_spines,
    screeplot,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


#%% Load and preprocess the data

VERSION = "2020-01-29"
print(f"Using version {VERSION}")

graph_type = "G"
threshold = 0
weight = "weight"
all_out = False
mg = load_metagraph(graph_type, VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=True,
    remove_pdiff=False,
    binarize=False,
    weight=weight,
)
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")

if all_out:
    out_classes = [
        "O_dVNC",
        "O_dSEZ",
        "O_IPC",
        "O_ITP",
        "O_dSEZ;FFN",
        "O_CA-LP",
        "O_dSEZ;FB2N",
    ]
else:
    out_classes = ["O_dVNC"]

class_key = "Merge Class"
sens_classes = ["sens-ORN"]

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
n_verts = len(adj)
meta = mg.meta.copy()
g = mg.g.copy()
meta["idx"] = range(len(meta))

from_inds = meta[meta[class_key].isin(sens_classes)]["idx"].values
out_inds = meta[meta[class_key].isin(out_classes)]["idx"].values
ind_map = dict(zip(meta.index, meta["idx"]))
g = nx.relabel_nodes(g, ind_map, copy=True)
out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# # Try the propogation thing

p = 0.01
not_probs = (1 - p) ** adj  # probability of none of the synapses causing postsynaptic
probs = 1 - not_probs  # probability of ANY of the synapses firing onto next

# %% [markdown]
# ## generate random "waves"
currtime = time.time()

n_sims = 1
paths = []
for f in from_inds:
    for i in range(n_sims):
        temp_paths = generate_random_cascade(
            f, probs, 0, stop_inds=out_inds, max_depth=20
        )
        paths += temp_paths

print(len(paths))
print(f"{time.time() - currtime} elapsed")

# %% [markdown]
# #

meta["median_visit"] = -1
meta["n_visits"] = 0

visit_orders = path_to_visits(paths, n_verts)

for node_ind, visits in visit_orders.items():
    median_order = np.median(visits)
    meta.iloc[node_ind, meta.columns.get_loc("median_visit")] = median_order
    meta.iloc[node_ind, meta.columns.get_loc("n_visits")] = len(visits)


# %% [markdown]
# #


path_graph = to_path_graph(paths)


visit_map = dict(zip(meta["idx"].values, meta["median_visit"].values))
nx.set_node_attributes(path_graph, visit_map, name="median_visit")

class_map = dict(zip(meta["idx"].values, meta["Merge Class"].values))
nx.set_node_attributes(path_graph, class_map, name="cell_class")


def add_attributes(
    g,
    drop_neg=True,
    remove_diag=True,
    size_scaler=1,
    use_counts=False,
    use_weights=True,
    color_map=None,
):
    nodelist = list(g.nodes())

    # add spectral properties
    sym_adj = symmetrize(nx.to_numpy_array(g, nodelist=nodelist))
    n_components = 10
    latent = AdjacencySpectralEmbed(n_components=n_components).fit_transform(sym_adj)
    for i in range(n_components):
        latent_dim = latent[:, i]
        lap_map = dict(zip(nodelist, latent_dim))
        nx.set_node_attributes(g, lap_map, name=f"AdjEvec-{i}")

    # add spring layout properties
    pos = nx.spring_layout(g)
    spring_x = {}
    spring_y = {}
    for key, val in pos.items():
        spring_x[key] = val[0]
        spring_y[key] = val[1]
    nx.set_node_attributes(g, spring_x, name="Spring-x")
    nx.set_node_attributes(g, spring_y, name="Spring-y")

    # add colors
    # nx.set_node_attributes(g, color_map, name="Color")
    for node, data in g.nodes(data=True):
        c = data["cell_class"]
        color = CLASS_COLOR_DICT[c]
        data["color"] = color

    # add size attribute base on number of edges
    size_map = dict(path_graph.degree(weight="weight"))
    nx.set_node_attributes(g, size_map, name="Size")

    return g


path_graph = add_attributes(path_graph)


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
draw_networkx_nice(
    path_graph,
    "AdjEvec-1",
    "median_visit",
    sizes="Size",
    colors="color",
    weight_scale=0.0001,
    draw_labels=False,
    ax=ax,
    size_scale=0.005,
)
ax.invert_yaxis()
stashfig(f"propogate-graph-map")
plt.close()


# %% [markdown]
# #


latent = ase(probs, 5, ptr=True)
pairplot(latent, labels=meta["Merge Class"].values, palette=CLASS_COLOR_DICT)

latent = ase(probs, 5, ptr=False)
pairplot(latent, labels=meta["Merge Class"].values, palette=CLASS_COLOR_DICT)
