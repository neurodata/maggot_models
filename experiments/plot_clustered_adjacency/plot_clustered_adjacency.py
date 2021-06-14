#%%
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import NodeMixin
from giskard.plot import MatrixGrid, scattermap
from giskard.utils import get_paired_inds
from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.plot import adjplot
from graspologic.utils import binarize, remove_loops
from scipy.stats import poisson
from src.data import join_node_meta, load_maggot_graph, load_palette
from src.io import savefig
from src.visualization import adjplot, set_theme

t0 = time.time()
set_theme()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/plot_clustered_adjacency/figs",
        **kws,
    )


n_components = 10
min_split = 32
i = 1
CLUSTER_KEY = f"dc_level_{i}_n_components={n_components}_min_split={min_split}"


mg = load_maggot_graph()
mg = mg[mg.nodes["has_embedding"]]
nodes = mg.nodes
adj = mg.sum.adj


HUE_KEY = "simple_group"
palette = load_palette()

HUE_ORDER = "sum_signal_flow"


#%%


meta = nodes.copy()
meta["inds"] = range(len(meta))
lowest_level = 7
level_names = [
    f"dc_level_{i}_n_components={n_components}_min_split={min_split}"
    for i in range(lowest_level + 1)
]
level_names += [HUE_KEY]


def sort_meta(meta, group, group_order=None, item_order=[], ascending=True):
    sort_class = group
    group_order = [group_order]
    total_sort_by = []
    for sc in sort_class:
        for co in group_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    meta = meta.sort_values(total_sort_by, ascending=ascending)
    return meta


sorted_meta = sort_meta(
    meta, level_names, group_order=HUE_ORDER, item_order=[HUE_KEY, HUE_ORDER]
)
sort_inds = sorted_meta["inds"]
sorted_adj = adj[sort_inds][:, sort_inds]


adjplot(
    sorted_adj,
    meta=sorted_meta,
    plot_type="scattermap",
    color=HUE_KEY,
    palette=palette,
    sizes=(1, 1),
    ticks=False,
)

#%%
sorted_meta = meta.copy()
sorted_meta["sort_inds"] = np.arange(len(sorted_meta))
group = level_names + ["merge_class"]
sorted_meta = sort_meta(
    sorted_meta,
    group,
    group_order=HUE_ORDER,
    item_order=[HUE_KEY, HUE_ORDER],
)
sorted_meta["new_inds"] = np.arange(len(sorted_meta))

sort_inds = sorted_meta["sort_inds"]
sorted_adj = adj[np.ix_(sort_inds, sort_inds)]


class MetaNode(NodeMixin):
    def __init__(self, name, parent=None, children=None, meta=None):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children
        self.meta = meta

    def hierarchical_mean(self, key):
        if self.is_leaf:
            meta = self.meta
            var = meta[key]
            return np.mean(var)
        else:
            children = self.children
            child_vars = [child.hierarchical_mean(key) for child in children]
            return np.mean(child_vars)


def get_parent_label(label):
    if len(label) <= 1:
        return None
    elif label[-1] == "-":
        return label[:-1]
    else:  # then ends in a -number
        return label[:-2]


def make_node(label, node_map):
    if label not in node_map:
        node = MetaNode(label)
        node_map[label] = node
    else:
        node = node_map[label]
    return node


meta = sorted_meta
node_map = {}
for i in range(lowest_level, -1, -1):
    level_labels = meta[f"lvl{i}_labels"].unique()
    for label in level_labels:
        node = make_node(label, node_map)
        node.meta = meta[meta[f"lvl{i}_labels"] == label]
        parent_label = get_parent_label(label)
        if parent_label is not None:
            parent = make_node(parent_label, node_map)
            node.parent = parent

root = node


adjplot(
    adj,
    meta=nodes,
    plot_type="scattermap",
    sort_class=CLUSTER_KEY,
    class_order="sum_walk_sort",
    item_order=HUE_KEY,
    ticks=False,
    colors=HUE_KEY,
    palette=palette,
    sizes=(1, 2),
    gridline_kws=dict(linewidth=0.5, linestyle=":", color="grey"),
)
stashfig(f"adjacency-matrix-cluster_key={CLUSTER_KEY}")
