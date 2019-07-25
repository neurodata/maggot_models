from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.utils import binarize
import pandas as pd
from pathlib import Path
import networkx as nx


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
