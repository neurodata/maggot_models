# %% [markdown]
# #
from src.data import load_networkx, load_everything
import networkx as nx
from graspy.utils import binarize, symmetrize

graph = load_networkx("G")

nx.algorithms.diameter(graph)

# %% [markdown]
# #
adj = load_everything("G")
adj = symmetrize(adj, "avg")
graph = nx.from_numpy_array(adj)
nx.algorithms.diameter(graph)
# %% [markdown]
# #
adj = load_everything("Gad")
adj = symmetrize(adj, "avg")
graph = nx.from_numpy_array(adj)
nx.algorithms.diameter(graph)
