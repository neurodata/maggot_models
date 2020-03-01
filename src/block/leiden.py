from graspy.utils import symmetrize
from src.graph import MetaGraph
import pandas as pd
import numpy as np
import igraph as ig
import networkx as nx
import leidenalg as la


def run_leiden(mg, temp_loc=None):
    adj = mg.adj
    adj = symmetrize(adj, method="avg")
    mg = MetaGraph(adj, mg.meta)

    if temp_loc is None:
        temp_loc = f"maggot_models/data/interim/temp-{np.random.randint(1e8)}.graphml"
    nx.write_graphml(mg.g, temp_loc)

    g = ig.Graph.Read_GraphML(temp_loc)
    nodes = [int(v["id"]) for v in g.vs]
    vert_part = la.find_partition(g, la.ModularityVertexPartition)
    labels = vert_part.membership
    partition = pd.Series(data=labels, index=nodes)
    return partition


def run_leiden_igraph(mg, temp_loc=None):
    adj = mg.adj
    adj = symmetrize(adj, method="avg")
    mg = MetaGraph(adj, mg.meta)

    if temp_loc is None:
        temp_loc = f"maggot_models/data/interim/temp-{np.random.randint(1e8)}.graphml"
    nx.write_graphml(mg.g, temp_loc)

    g = ig.Graph.Read_GraphML(temp_loc)
    out = g.community_leiden()
    # labels = vert_part.membership
    # partition = pd.Series(data=labels, index=nodes)
