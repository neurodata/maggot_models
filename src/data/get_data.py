from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.utils import binarize
import pandas as pd
from pathlib import Path
import networkx as nx
from src.utils import meta_to_array
import numpy as np


def load_left():
    """
    Load the left connectome. Wraps graspy
    """
    graph, labels = load_drosophila_left(return_labels=True)
    graph = binarize(graph)
    return graph, labels


def load_right():
    """
    Load the right connectome. Wraps graspy
    """
    graph, labels = load_drosophila_right(return_labels=True)
    graph = binarize(graph)
    return graph, labels


def load_new_left(return_full_labels=False, return_names=False):
    data_path = Path("./maggot_models/data/processed/")
    adj_path = data_path / "BP_20190424mw_left_mb_adj.csv"
    meta_path = data_path / "BP_20190424mw_left_mb_meta.csv"
    adj_df = pd.read_csv(adj_path, header=0, index_col=0)
    meta_df = pd.read_csv(meta_path, header=0, index_col=0)
    adj = adj_df.values
    adj = binarize(adj)
    labels = meta_df["simple_class"].values.astype(str)
    if return_full_labels:
        full_labels = meta_df["Class"].values.astype(str)
        return adj, labels, full_labels
    elif return_names:
        names = meta_df["Name"].values.astype(str)
        return adj, labels, names
    else:
        return adj, labels


def load_new_right(return_full_labels=False, return_names=False):
    data_path = Path("./maggot_models/data/processed/")
    adj_path = data_path / "BP_20190424mw_right_mb_adj.csv"
    meta_path = data_path / "BP_20190424mw_right_mb_meta.csv"
    adj_df = pd.read_csv(adj_path, header=0, index_col=0)
    meta_df = pd.read_csv(meta_path, header=0, index_col=0)
    adj = adj_df.values
    adj = binarize(adj)
    labels = meta_df["simple_class"].values.astype(str)
    if return_full_labels:
        full_labels = meta_df["Class"].values.astype(str)
        return adj, labels, full_labels
    elif return_names:
        names = meta_df["Name"].values.astype(str)
        return adj, labels, names
    else:
        return adj, labels


def load_june(graph_type):
    data_path = Path("maggot_models/data/raw/20190615_mw")
    base_file_name = "mw_20190615_"
    file_path = data_path / (base_file_name + graph_type + ".graphml")
    graph = nx.read_graphml(file_path)
    return graph


def load_networkx(graph_type, version="2019-09-18-v2"):
    data_path = Path("maggot_models/data/processed")
    data_path = data_path / version
    file_path = data_path / (graph_type + ".graphml")
    graph = nx.read_graphml(file_path)
    return graph


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
