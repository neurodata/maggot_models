import networkx as nx
import numpy as np
import pandas as pd
from graspy.utils import is_almost_symmetric, get_lcc
from pathlib import Path
from operator import itemgetter
from tqdm import tqdm

# helper functions

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome")


def _nx_to_numpy_pandas(g, weight="weight"):
    node_data = dict(g.nodes(data=True))
    meta = pd.DataFrame().from_dict(node_data, orient="index")
    df_adj = nx.to_pandas_adjacency(g, weight=weight, nodelist=meta.index)
    adj = df_adj.values
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
        col_vals = meta.loc[index, c]
        values = dict(zip(index, col_vals))
        nx.set_node_attributes(g, values, name=c)
    return g


def _verify_networkx(source, target, n_checks):
    edgelist = list(source.edges)
    for i in range(n_checks):
        edge_ind = np.random.choice(len(edgelist))
        edge = edgelist[edge_ind]
        if not target.has_edge(*edge):
            ValueError(f"Edge {edge} missing in target graph")


class MetaGraph:
    # init using a nx graph or an adjacency and metadata
    def __init__(self, graph, meta=None, weight="weight"):
        if isinstance(graph, (nx.Graph, nx.DiGraph)):
            graph = graph.copy()
            self.g = graph
            adj, meta = _nx_to_numpy_pandas(graph, weight=weight)
            self.adj = adj
            self.meta = meta
            self.weight = weight
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
        adj, meta = _nx_to_numpy_pandas(g, weight=self.weight)
        self.adj = adj
        self.meta = meta
        self.n_verts = adj.shape[0]

    def _update_from_numpy_pandas(self, adj, meta):
        self.adj = adj
        self.meta = meta
        g = _numpy_pandas_to_nx(adj, meta)
        self.g = g
        self.n_verts = adj.shape[0]

    def reindex(self, perm_inds):
        adj = self.adj.copy()
        adj = adj[np.ix_(perm_inds, perm_inds)]
        self.adj = adj
        self.meta = self.meta.iloc[perm_inds, :]
        self.g = _numpy_pandas_to_nx(adj, self.meta)
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
        degree_df["ID"] = self.meta.index
        degree_df = degree_df.set_index("ID")
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
            raise KeyError(f"Key {n} not present as index or column in MetaGraph")

    def __setitem__(self, n, val):
        if n in self.meta.columns:
            self.meta[n] = val
        else:
            raise KeyError(f"Key {n} not present as column in MetaGraph")

    def verify_networkx(self, n_checks=1000, version="2019-12-18", graph_type="G"):
        from src.data import load_networkx

        raw_g = load_networkx(graph_type, version)
        _verify_networkx(self.g, raw_g, n_checks)
        _verify_networkx(raw_g, self.g, n_checks)

    def sort_values(self, sortby, ascending=False):
        self.meta["Original index"] = range(self.meta.shape[0])
        self.meta.sort_values(
            sortby, inplace=True, kind="mergesort", ascending=ascending
        )
        temp_inds = self.meta["Original index"]
        self.adj = self.adj[np.ix_(temp_inds, temp_inds)]
        return self

    def to_edgelist(self, remove_unpaired=False):
        meta = self.meta
        # extract edgelist from the graph
        edgelist_df = nx.to_pandas_edgelist(self.g)
        # get metadata for the nodes and rename them based on source/target
        sources = edgelist_df["source"].values.astype("int64")
        targets = edgelist_df["target"].values.astype("int64")
        source_meta = meta.loc[sources]
        source_meta.index = pd.RangeIndex(len(source_meta))
        target_meta = meta.loc[targets]
        target_meta.index = pd.RangeIndex(len(target_meta))
        source_meta.rename(_source_mapper, axis=1, inplace=True)
        target_meta.rename(_target_mapper, axis=1, inplace=True)
        # append to the columns of edgelist
        edgelist_df = pd.concat(
            (edgelist_df, source_meta, target_meta), axis=1, ignore_index=False
        )
        # add column of which pairs are incident to each edges
        edgelist_df["edge pairs"] = list(
            zip(edgelist_df["source Pair ID"], edgelist_df["target Pair ID"])
        )
        # specify whether this is ipsilateral or contralateral
        edgelist_df["is_ipsi"] = (
            edgelist_df["source Hemisphere"] == edgelist_df["target Hemisphere"]
        )
        # now that we have specified side as well, max # here is 2 (one on each side)
        edgelist_df["edge pairs"] = list(
            zip(edgelist_df["edge pairs"], edgelist_df["is_ipsi"])
        )
        edgelist_df = edgelist_df.sort_values("edge pairs", ascending=False)
        # remove edges incident to an unpaired node
        if remove_unpaired:
            edgelist_df = edgelist_df[edgelist_df["target Pair ID"] != -1]
            edgelist_df = edgelist_df[edgelist_df["source Pair ID"] != -1]
        uni_edge_pairs, uni_edge_counts = np.unique(
            edgelist_df["edge pairs"], return_counts=True
        )
        # give each edge pair an ID
        edge_pair_map = dict(zip(uni_edge_pairs, range(len(uni_edge_pairs))))
        edgelist_df["edge pair ID"] = itemgetter(*edgelist_df["edge pairs"])(
            edge_pair_map
        )
        # count how many times each potential edge pair happens
        edge_pair_count_map = dict(zip(uni_edge_pairs, uni_edge_counts))
        edgelist_df["edge pair counts"] = itemgetter(*edgelist_df["edge pairs"])(
            edge_pair_count_map
        )
        return edgelist_df

    def __len__(self):
        assert self.adj.shape[0] == len(self.g)
        return self.adj.shape[0]

    def verify(self, n_checks=1000, version="2020-01-29", graph_type="G"):
        name_map = {
            "Gaa": "axon-axon",
            "Gad": "axon-dendrite",
            "Gda": "dendrite-axon",
            "Gdd": "dendrite-dendrite",
            "Gaan": "axon-axon",
            "Gadn": "axon-dendrite",
            "Gdan": "dendrite-axon",
            "Gddn": "dendrite-dendrite",
        }
        raw_path = Path(
            "maggot_models/data/raw/Maggot-Brain-Connectome/4-color-matrices_Brain"
        )
        raw_path = raw_path / version
        filename = name_map[graph_type]
        raw_path = raw_path / str(filename + ".csv")
        adj_df = pd.read_csv(raw_path, index_col=0, header=0)
        adj_df.columns = adj_df.index
        nonzero_inds = np.nonzero(self.adj)
        choice_inds = np.random.choice(len(nonzero_inds[0]), size=n_checks)
        index = self.meta.index
        print(f"Verifying {n_checks} edges are present in original graph")
        print()
        for choice in tqdm(choice_inds):
            row_ind = nonzero_inds[0][choice]
            col_ind = nonzero_inds[1][choice]
            row_id = index[row_ind]
            col_id = index[col_ind]
            if adj_df.loc[row_id, col_id] <= 0:
                print(f"Row ID: {row_id}")
                print(f"Column ID: {col_id}")
                print()
                raise ValueError(f"Edge from {row_id} to {col_id} does not exist")


def _source_mapper(name):
    return "source " + name


def _target_mapper(name):
    return "target " + name
