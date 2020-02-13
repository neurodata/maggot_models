# TODO write utilities for running MCMC stuff

import networkx as nx
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool import load_graph
import numpy as np
import pandas as pd
import os
from src.graph import MetaGraph


def run_minimize_blockmodel(mg, temp_loc=None, weight_model=None):
    meta = mg.meta.copy()
    meta = pd.DataFrame(mg.meta["neuron_name"])
    mg = MetaGraph(mg.adj, meta)
    if temp_loc is None:
        temp_loc = f"maggot_models/data/interim/temp-{np.random.randint(1e8)}.graphml"
    # save to temp
    nx.write_graphml(mg.g, temp_loc)
    # load into graph-tool from temp
    g = load_graph(temp_loc, fmt="graphml")
    os.remove(temp_loc)

    total_degrees = g.get_total_degrees(g.get_vertices())
    remove_verts = np.where(total_degrees == 0)[0]
    g.remove_vertex(remove_verts)

    if weight_model is not None:
        recs = [g.ep.weight]
        rec_types = [weight_model]
    else:
        recs = []
        rec_types = []
    state_args = dict(recs=recs, rec_types=rec_types)
    min_state = minimize_blockmodel_dl(g, verbose=False, state_args=state_args)

    blocks = list(min_state.get_blocks())
    verts = g.get_vertices()

    block_map = {}

    for v, b in zip(verts, blocks):
        cell_id = int(g.vertex_properties["_graphml_vertex_id"][v])
        block_map[cell_id] = int(b)

    block_series = pd.Series(block_map)
    block_series.name = "block_label"
    return block_series
