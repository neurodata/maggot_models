# from graspy.datasets import load_drosophila_left, load_drosophila_right
# from graspy.utils import binarize
# import pandas as pd
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# from src.utils import meta_to_array
from src.graph import MetaGraph

DATA_VERSION = "2021-03-10"
DATA_DIR = "maggot_models/data/processed"
DATA_PATH = Path(DATA_DIR)


def _get_folder(path, version):
    if path is None:
        path = DATA_PATH
    if version is None:
        version = DATA_VERSION
    return path / version


def load_node_meta(path=None, version=None):
    folder = _get_folder(version, path)
    meta = pd.read_csv(folder / "meta_data.csv", index_col=0)
    meta.sort_index(inplace=True)
    return meta


def load_edgelist(graph_type="G", path=None, version=None):
    folder = _get_folder(version, path)
    edgelist = pd.read_csv(
        folder / f"{graph_type}_edgelist.txt",
        delimiter=" ",
        header=None,
        names=["source", "target", "weight"],
    )
    return edgelist


def load_networkx(graph_type="G", node_meta=None, path=None, version=None):
    edgelist = load_edgelist(graph_type)
    g = nx.from_pandas_edgelist(edgelist, edge_attr="weight", create_using=nx.DiGraph())
    for node in node_meta.index:
        if node not in g.nodes:
            g.add_node(node)
    if node_meta is not None:
        meta_data_dict = node_meta.to_dict(orient="index")
        nx.set_node_attributes(g, meta_data_dict)
    return g


def load_adjacency(
    graph_type="G", nodelist=None, output="numpy", path=None, version=None
):
    g = load_networkx(graph_type=graph_type, path=path, version=version)
    if output == "numpy":
        adj = nx.to_numpy_array(g, nodelist=nodelist)
    elif output == "pandas":
        adj = nx.to_pandas_adjacency(g, nodelist=nodelist)
    return adj


# def load_left():
#     """
#     Load the left connectome. Wraps graspy
#     """
#     graph, labels = load_drosophila_left(return_labels=True)
#     graph = binarize(graph)
#     return graph, labels


# def load_right():
#     """
#     Load the right connectome. Wraps graspy
#     """
#     graph, labels = load_drosophila_right(return_labels=True)
#     graph = binarize(graph)
#     return graph, labels


# def load_new_left(return_full_labels=False, return_names=False):
#     data_path = Path("./maggot_models/data/processed/")
#     adj_path = data_path / "BP_20190424mw_left_mb_adj.csv"
#     meta_path = data_path / "BP_20190424mw_left_mb_meta.csv"
#     adj_df = pd.read_csv(adj_path, header=0, index_col=0)
#     meta_df = pd.read_csv(meta_path, header=0, index_col=0)
#     adj = adj_df.values
#     adj = binarize(adj)
#     labels = meta_df["simple_class"].values.astype(str)
#     if return_full_labels:
#         full_labels = meta_df["Class"].values.astype(str)
#         return adj, labels, full_labels
#     elif return_names:
#         names = meta_df["Name"].values.astype(str)
#         return adj, labels, names
#     else:
#         return adj, labels


# def load_new_right(return_full_labels=False, return_names=False):
#     data_path = Path("./maggot_models/data/processed/")
#     adj_path = data_path / "BP_20190424mw_right_mb_adj.csv"
#     meta_path = data_path / "BP_20190424mw_right_mb_meta.csv"
#     adj_df = pd.read_csv(adj_path, header=0, index_col=0)
#     meta_df = pd.read_csv(meta_path, header=0, index_col=0)
#     adj = adj_df.values
#     adj = binarize(adj)
#     labels = meta_df["simple_class"].values.astype(str)
#     if return_full_labels:
#         full_labels = meta_df["Class"].values.astype(str)
#         return adj, labels, full_labels
#     elif return_names:
#         names = meta_df["Name"].values.astype(str)
#         return adj, labels, names
#     else:
#         return adj, labels


def load_june(graph_type):
    data_path = Path("maggot_models/data/raw/20190615_mw")
    base_file_name = "mw_20190615_"
    file_path = data_path / (base_file_name + graph_type + ".graphml")
    graph = nx.read_graphml(file_path)
    return graph


def load_networkx_graphml(graph_type, path=None, version=None):
    data_path = Path(path)
    data_path = data_path / version
    file_path = data_path / (graph_type + ".graphml")
    graph = nx.read_graphml(file_path, node_type=int)
    return graph


def load_metagraph(graph_type, path=None, version=None, nodelist=None):
    node_meta = load_node_meta(path=path, version=version)
    g = load_networkx(
        graph_type=graph_type, path=path, version=version, node_meta=node_meta
    )
    mg = MetaGraph(g)
    # g = load_networkx_graphml(graph_type, path=path, version=version)
    # if nodelist is not None:
    #     for node in nodelist:
    #         if node not in g.nodes:
    #             g.add_node(node)
    # mg = MetaGraph(g)
    # print(np.isin(mg.meta.index, nodelist))
    if nodelist is None:
        nodelist = sorted(mg.meta.index.values)
    mg.reindex(nodelist, use_ids=True, inplace=True)
    return mg


def load_palette(path=None, version=None):
    folder = _get_folder(path, version)
    with open(folder / "simple_color_map.json", "r") as f:
        palette = json.load(f)
    return palette


def load_everything(
    graph_type,
    version="2019-09-18-v2",
    return_keys=None,
    return_df=False,
    return_ids=False,
):

    """Function to load an adjacency matrix and optionally return some associated
    metadata

    Parameters
    ----------
    graph_type : str
        Which version of the graph to load
    version : str, optional
        Date/version descriptor for which dataset to use, by default "2019-09-18-v2"
    return_df : bool, optional
        Whether to return a Pandas DataFrame representation of the adjacency,
        by default False
    return_ids : bool, optional
        Whether to return the cell ids (skeleton ids), by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    graph = load_networkx(graph_type, version=version)
    nx_ids = np.array(list(graph.nodes()), dtype=int)
    df_adj = nx.to_pandas_adjacency(graph)
    df_ids = df_adj.index.values.astype(int)
    if not np.array_equal(nx_ids, df_ids):
        raise ValueError("Networkx indexing is inconsistent with Pandas adjacency")
    adj = nx.to_pandas_adjacency(graph).values
    outs = [adj]

    if return_keys is not None:
        if not isinstance(return_keys, list):
            return_keys = [return_keys]
        for k in return_keys:
            labels = meta_to_array(graph, k)
            outs.append(labels)
    if return_df:
        outs.append(df_adj)
    if return_ids:
        outs.append(df_ids)
    if len(outs) > 1:
        outs = tuple(outs)
        return outs
    else:
        return adj
