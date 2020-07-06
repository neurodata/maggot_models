import networkx as nx
import numpy as np
import pandas as pd
from graspy.utils import is_almost_symmetric, get_lcc
from pathlib import Path
from operator import itemgetter
from copy import deepcopy

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


# NOTE: something about this breaks when going to Pandas 1, not sure what
class MetaGraph:
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

    def copy(self):
        return deepcopy(self)

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

    def reindex(self, perm_inds, use_ids=False, inplace=True):
        if use_ids:
            meta = self.meta.copy()
            meta["idx"] = range(len(meta))
            meta = meta.reindex(perm_inds)
            idx_perm_inds = meta["idx"]
            return self.reindex(idx_perm_inds, use_ids=False, inplace=inplace)
        else:
            adj = self.adj.copy()
            adj = adj[np.ix_(perm_inds, perm_inds)]
            meta = self.meta.iloc[perm_inds, :].copy()
            if inplace:
                self.adj = adj
                self.meta = meta
                self.g = _numpy_pandas_to_nx(adj, self.meta)
                return self
            else:
                return MetaGraph(adj, meta)

    def prune(self):
        # remove pendant nodes
        # update nx
        # update adjacency matrix
        return self

    def make_lcc(self):
        lcc, inds = get_lcc(self.adj, return_inds=True)
        self.adj = lcc
        self.meta = self.meta.iloc[inds, :]
        self.g = _numpy_pandas_to_nx(self.adj, self.meta)
        self.n_verts = self.adj.shape[0]
        return self

    def calculate_degrees(self, inplace=False):
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
        if inplace:
            self.meta = pd.concat((self.meta, degree_df), ignore_index=False, axis=1)
            return self
        else:
            return degree_df

    def __getitem__(self, n):
        if n in self.meta.columns:
            return self.meta[n]
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
            zip(edgelist_df["source pair_id"], edgelist_df["target pair_id"])
        )
        # specify whether this is ipsilateral or contralateral
        edgelist_df["is_ipsi"] = (
            edgelist_df["source hemisphere"] == edgelist_df["target hemisphere"]
        )
        # now that we have specified side as well, max # here is 2 (one on each side)
        edgelist_df["edge pairs"] = list(
            zip(edgelist_df["edge pairs"], edgelist_df["is_ipsi"])
        )
        edgelist_df = edgelist_df.sort_values("edge pairs", ascending=False)
        # remove edges incident to an unpaired node

        if remove_unpaired:
            edgelist_df = edgelist_df[edgelist_df["target pair_id"] != -1]
            edgelist_df = edgelist_df[edgelist_df["source pair_id"] != -1]

        uni_edge_pairs, uni_edge_counts = np.unique(
            edgelist_df["edge pairs"], return_counts=True
        )

        # give each edge pair an ID
        edge_pair_map = dict(zip(uni_edge_pairs, range(len(uni_edge_pairs))))
        edgelist_df["edge pair_id"] = itemgetter(*edgelist_df["edge pairs"])(
            edge_pair_map
        )

        # edgelist_df[edgelist_df["target Pair ID"] == -1]["edge pair ID"] = -1
        # edgelist_df[edgelist_df["source Pair ID"] == -1]["edge pair ID"] = -1

        # count how many times each potential edge pair happens
        edge_pair_count_map = dict(zip(uni_edge_pairs, uni_edge_counts))
        edgelist_df["edge pair counts"] = itemgetter(*edgelist_df["edge pairs"])(
            edge_pair_count_map
        )

        inds = np.where(
            np.logical_or(
                edgelist_df["target pair_id"] == -1, edgelist_df["source pair_id"] == -1
            )
        )
        inds = edgelist_df.index[inds]
        edgelist_df.loc[inds, ["edge pair_id", "edge pair counts"]] = -1

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
        if graph_type == "G":
            graph_types = ["Gaa", "Gad", "Gda", "Gdd"]
        elif graph_type == "Gn":
            graph_types = ["Gaan", "Gadn", "Gdan", "Gddn"]
        else:
            graph_types = [graph_type]
        dfs = []
        for g in graph_types:
            filename = name_map[g]
            graph_path = raw_path / str(filename + ".csv")
            adj_df = pd.read_csv(graph_path, index_col=0, header=0)
            adj_df.columns = adj_df.index
            dfs.append(adj_df)
        adj_df = sum(dfs)

        nonzero_inds = np.nonzero(self.adj)
        choice_inds = np.random.choice(len(nonzero_inds[0]), size=n_checks)
        index = self.meta.index
        print(f"\nVerifying {n_checks} edges are present in original graph\n")
        for choice in choice_inds:
            row_ind = nonzero_inds[0][choice]
            col_ind = nonzero_inds[1][choice]
            row_id = index[row_ind]
            col_id = index[col_ind]
            if adj_df.loc[row_id, col_id] <= 0:
                print(f"Row ID: {row_id}")
                print(f"Column ID: {col_id}")
                print()
                raise ValueError(f"Edge from {row_id} to {col_id} does not exist")

    def remove_pdiff(self):
        not_pdiff = np.where(~self.meta["partially_differentiated"])[0]
        return self.reindex(not_pdiff)


def add_max_weight(df):
    """Input is an edgelist with `edge pair ID`s

    Uses the 'weight' column
    
    Parameters
    ----------
    df : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    max_pair_edges = df.groupby("edge pair ID", sort=False)["weight"].max()
    # HACK think there is a better way to do this with pandas .map()
    edge_max_weight_map = dict(zip(max_pair_edges.index.values, max_pair_edges.values))
    df["max_weight"] = itemgetter(*df["edge pair ID"])(edge_max_weight_map)
    # need to make sure the unpaired edges don't get a huge max weight
    asym_inds = df[df["edge pair ID"] == -1].index
    df.loc[asym_inds, "max_weight"] = df.loc[asym_inds, "weight"]
    return df


def edgelist_to_mg(edgelist, meta, weight="weight"):
    g = nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=nx.DiGraph)
    nx.set_node_attributes(g, meta.to_dict(orient="index"))
    mg = MetaGraph(g, weight=weight)
    return mg


def preprocess(
    mg,
    threshold=0,
    sym_threshold=True,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
):
    edgelist = mg.to_edgelist()
    if sym_threshold:
        # note that this just removes asymmetric edges
        # another option would be to actually max the edges into mirror images
        edgelist = add_max_weight(edgelist)
        edgelist = edgelist[edgelist["max_weight"] > threshold]
    else:
        edgelist = edgelist[edgelist["weight"] > threshold]
    mg = edgelist_to_mg(edgelist, mg.meta, weight=weight)
    if remove_pdiff:
        mg = mg.remove_pdiff()  # FIXME need a way to deal with pairs
    mg = mg.make_lcc()  # FIXME need a way to deal with pairs
    if binarize:
        # HACK there must be a better way in nx?
        adj = mg.adj
        adj[adj > 0] = 1
        mg = MetaGraph(adj, mg.meta)
    return mg


def _source_mapper(name):
    return "source " + name


def _target_mapper(name):
    return "target " + name
