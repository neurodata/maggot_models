from graspy.utils import symmetrize
from src.graph import MetaGraph
import pandas as pd
import numpy as np
import igraph as ig
import networkx as nx
import leidenalg as la


def _process_metagraph(mg, temp_loc):
    adj = mg.adj
    adj = symmetrize(adj, method="avg")
    mg = MetaGraph(adj, mg.meta)
    nx.write_graphml(mg.g, temp_loc)


def run_leiden(mg, temp_loc=None, implementation="igraph", **kws):
    if temp_loc is None:
        temp_loc = f"maggot_models/data/interim/temp-{np.random.randint(1e8)}.graphml"
    else:
        temp_loc = f"maggot_models/data/interim/{temp_loc}.graphml"
    _process_metagraph(mg, temp_loc)
    g = ig.Graph.Read_GraphML(temp_loc)
    nodes = [int(v["id"]) for v in g.vs]
    if implementation == "igraph":
        vert_part = g.community_leiden(**kws)
    elif implementation == "leidenalg":
        vert_part = la.find_partition(g, la.ModularityVertexPartition, **kws)
    labels = vert_part.membership
    partition = pd.Series(data=labels, index=nodes)
    return partition, vert_part.modularity

