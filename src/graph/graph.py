import networkx as nx
import numpy as np
import pandas as pd
from graspy.utils import is_almost_symmetric, get_lcc
from pathlib import Path
from itertools import islice

# helper functions

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome")


def _nx_to_numpy_pandas(g):
    node_data = dict(g.nodes(data=True))
    meta = pd.DataFrame().from_dict(node_data, orient="index")
    meta.index = meta.index.values.astype(int)
    nx_ids = np.array(list(g.nodes()), dtype=int)
    df_adj = nx.to_pandas_adjacency(g)
    df_ids = df_adj.index.values.astype(int)
    if not np.array_equal(nx_ids, df_ids):
        raise ValueError("Networkx indexing is inconsistent with Pandas adjacency")
    adj = nx.to_pandas_adjacency(g).values
    return adj, meta


def _numpy_pandas_to_nx(adj, meta):
    if is_almost_symmetric(adj):
        gtype = nx.Graph
    else:
        gtype = nx.DiGraph
    adj_df = pd.DataFrame(data=adj, index=meta.index.values, columns=meta.index.values)
    g = nx.from_pandas_adjacency(adj_df, create_using=gtype)
    index = meta.index
    for c in meta.columns:
        col_vals = meta[c]
        values = dict(zip(index, col_vals))
        nx.set_node_attributes(g, values, name=c)
    return g


def _verify(source, target, n_checks):
    edgelist = list(source.edges)
    for i in range(n_checks):
        edge_ind = np.random.choice(len(edgelist))
        edge = edgelist[edge_ind]
        if not target.has_edge(*edge):
            ValueError(f"Edge {edge} missing in target graph")


class MetaGraph:
    # init using a nx graph or an adjacency and metadata
    def __init__(self, graph, meta=None):
        if isinstance(graph, (nx.Graph, nx.DiGraph)):
            graph = graph.copy()
            self.g = graph
            adj, meta = _nx_to_numpy_pandas(graph)
            self.adj = adj
            self.meta = meta
        elif isinstance(graph, (np.ndarray)) and isinstance(meta, (pd.DataFrame)):
            graph = graph.copy()
            meta = meta.copy()
            self.adj = graph
            self.meta = meta
            g = _numpy_pandas_to_nx(graph, meta)
            self.g = g
        else:
            raise ValueError("Invalid data source to initialize MetaGraph")
        self.n_verts = self.adj.shape[0]

    def _update_from_nx(self, g):
        self.g = g
        adj, meta = _nx_to_numpy_pandas(g)
        self.adj = adj
        self.meta = meta
        self.n_verts = adj.shape[0]

    def _update_from_numpy_pandas(self, adj, g):
        pass

    def reindex(self, perm_inds):
        self.adj = self.adj[np.ix_(perm_inds, perm_inds)]
        index = self.meta.index
        new_index = index[perm_inds]
        self.meta = self.meta.reindex(new_index)
        return self

    def prune(self):
        # remove pendant nodes
        # update nx
        # update adjacency matrix
        return self

    def make_lcc(self):
        # make whole graph just the lcc
        # update nx
        # update adjacency matrix
        # update metadataframe
        lcc, inds = get_lcc(self.adj, return_inds=True)
        self.adj = lcc
        self.meta = self.meta.iloc[inds, :]
        self.g = _numpy_pandas_to_nx(self.adj, self.meta)
        self.n_verts = self.adj.shape[0]
        return self

    def calculate_degrees(self):
        adj = self.adj
        in_edgesums = adj.sum(axis=0)
        out_edgesums = adj.sum(axis=1)
        in_degree = np.count_nonzero(adj, axis=0)
        out_degree = np.count_nonzero(adj, axis=1)
        degree_df = pd.DataFrame()
        degree_df["In edgesum"] = in_edgesums
        degree_df["Out edgesum"] = out_edgesums
        degree_df["Total edgesum"] = in_edgesums + out_edgesums
        degree_df["In degree"] = in_degree
        degree_df["Out degree"] = out_degree
        degree_df["Total degree"] = in_degree + out_degree
        return degree_df

    def add_metadata(self, meta, name=None):
        # meta is either
        #      array of metadata corresponding to current adj/meta sorting
        #      dict of metadata, mapping node ID to value for new field
        #      dataframe with same indexes as current metadata df
        # name is the name of the field
        return self

    def __getitem__(self, n):
        if n in self.meta.columns:
            return self.meta[n].values
        elif n in self.meta.index:
            return self.meta.loc[n]
        else:
            raise NotImplementedError()

    def __setitem__(self, n, val):
        if n in self.meta.columns:
            self.meta[n] = val
        else:
            raise NotImplementedError()

    def verify(self, n_checks=1000, version="2019-12-18", graph_type="G"):
        from src.data import load_networkx

        raw_g = load_networkx(graph_type, version)
        _verify(self.g, raw_g, n_checks)
        _verify(raw_g, self.g, n_checks)

