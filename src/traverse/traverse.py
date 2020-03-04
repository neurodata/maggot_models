import numpy as np
import itertools
import networkx as nx


def path_to_visits(paths, n_verts, from_order=True, out_inds=[]):
    visit_orders = {i: [] for i in range(n_verts)}
    for path in paths:
        for i, n in enumerate(path):
            if from_order:
                visit_orders[n].append(i + 1)
            else:
                visit_orders[n].append(len(path) - i)
    return visit_orders


def to_path_graph(paths):
    path_graph = nx.MultiDiGraph()

    all_nodes = list(itertools.chain.from_iterable(paths))
    all_nodes = np.unique(all_nodes)
    path_graph.add_nodes_from(all_nodes)

    for path in paths:
        path_graph.add_edges_from(nx.utils.pairwise(path))

    path_graph = collapse_multigraph(path_graph)
    return path_graph


def collapse_multigraph(multigraph):
    """REF : https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-...
        into-simple-graph-with-weighted-edges
    
    Parameters
    ----------
    multigraph : [type]
        [description]
    """
    G = nx.DiGraph()
    for u, v, data in multigraph.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G
