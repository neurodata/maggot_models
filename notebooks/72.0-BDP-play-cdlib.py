# %% [markdown]
# #

from cdlib import algorithms, viz
import networkx as nx
import matplotlib.pyplot as plt
from graspy.plot import heatmap

g = nx.LFR_benchmark_graph(1000, 3, 1.5, 0.7, min_community=20, average_degree=5)
coms = algorithms.leiden(g)


def nc_to_label(coms):
    nodelist = []
    comlist = []
    com_map = coms.to_node_community_map()
    for node, assignment in com_map.items():
        assignment = assignment[0]
        nodelist.append(node)
        comlist.append(assignment)
    return nodelist, comlist


nodelist, labels = nc_to_label(coms)
adj = nx.to_numpy_array(g, nodelist=nodelist)

heatmap(adj, inner_hier_labels=labels)
