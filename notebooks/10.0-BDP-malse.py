#%%
from operator import itemgetter
from pathlib import Path
from warnings import filterwarnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from graspy.cluster import GaussianCluster
from graspy.embed import (
    AdjacencySpectralEmbed,
    LaplacianSpectralEmbed,
    OmnibusEmbed,
    MultipleASE,
)
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import (
    binarize,
    get_lcc,
    import_graph,
    is_fully_connected,
    pass_to_ranks,
    remove_loops,
    to_laplace,
)

from src.data import load_june
from src.utils import to_simple_class, get_subgraph

plt.style.use("seaborn-white")


def meta_to_array(graph, key):
    data = [meta[key] for node, meta in graph.nodes(data=True)]
    return np.array(data)


def get_simple(graph):
    classes = meta_to_array(graph, "Class")
    simple_classes = to_simple_class(classes)
    return simple_classes


def preprocess(graph):
    graph = binarize(graph)
    return graph


def save(name, fmt="pdf"):
    path = Path("/Users/bpedigo/JHU_code/maggot_models/maggot_models/notebooks/outs")
    plt.savefig(path / str(name + "." + fmt), fmt=fmt, facecolor="w")


base_title = " o binarized right"
side = "right"

full_graph = load_june("G")
full_graph = get_subgraph(full_graph, "Hemisphere", side)
simple_classes = get_simple(full_graph)
split_graph_types = ["Gaa", "Gad", "Gda", "Gdd"]

# load the graphs
split_graph_dict = {}
split_graph_list = []
for i, graph_type in enumerate(split_graph_types):
    # load
    graph = load_june(graph_type)
    graph = get_subgraph(graph, "Hemisphere", side)

    # save for later
    split_graph_dict[graph_type] = graph
    split_graph_list.append(graph)


#%% embedding parameters
n_components = 3

#%% ASE
embed_graph = get_lcc(full_graph)
labels = get_simple(embed_graph)
embed_graph = preprocess(embed_graph)

ase = AdjacencySpectralEmbed(n_components=n_components)
latent = ase.fit_transform(embed_graph)
latent = np.concatenate(latent, axis=-1)
pairplot(latent, labels=labels, title="ASE" + base_title)
save("ASE")
#%% LSE
regularizer = 1
embed_graph = get_lcc(full_graph)
labels = get_simple(embed_graph)
embed_graph = preprocess(embed_graph)
lse = LaplacianSpectralEmbed(
    form="R-DAD", n_components=n_components, regularizer=regularizer
)
latent = lse.fit_transform(embed_graph)
latent = np.concatenate(latent, axis=-1)
pairplot(latent, labels=labels, title="LSE" + base_title)
save("LSE")
#%% MASE
embed_graphs = []
for graph in split_graph_list:
    graph = preprocess(graph)
    embed_graphs.append(graph)

mase = MultipleASE(n_components=n_components, scaled=True)
shared_latent = mase.fit_transform(embed_graphs)
shared_latent = np.concatenate(shared_latent, axis=-1)
labels = get_simple(split_graph_list[0])
pairplot(shared_latent, labels=labels, title="MASE" + base_title)
save("MASE")
#%% MLSE
regularizer = 1
embed_graphs = []
for graph in split_graph_list:
    graph = preprocess(graph)
    laplacian = to_laplace(graph, "R-DAD", regularizer=regularizer)
    embed_graphs.append(laplacian)

mase = MultipleASE(n_components=n_components, scaled=False)
shared_latent = mase.fit_transform(embed_graphs)
shared_latent = np.concatenate(shared_latent, axis=-1)
labels = get_simple(split_graph_list[0])
pairplot(shared_latent, labels=labels, title="MLSE" + base_title)
save("MLSE")


#%%
