#%%
import time

import matplotlib.pyplot as plt
import numpy as np
from giskard.hierarchy import BaseNetworkTree
from giskard.plot import plot_dendrogram
from src.data import load_maggot_graph, load_palette
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

HUE_ORDER = "sum_walk_sort"


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
sorted_meta["sorted_adjacency_index"] = np.arange(len(sorted_meta))


class MetaTree(BaseNetworkTree):
    def __init__(self):
        super().__init__()

    def build(self, node_data, prefix="", postfix=""):
        if self.is_root and ("adjacency_index" not in node_data.columns):
            node_data = node_data.copy()
            node_data["adjacency_index"] = range(len(node_data))
        self._index = node_data.index
        self._node_data = node_data
        key = prefix + f"{self.depth}" + postfix
        if key in node_data:
            groups = node_data.groupby(key)
            for name, group in groups:
                child = MetaTree()
                child.parent = self
                child.build(group, prefix=prefix, postfix=postfix)


mt = MetaTree()
mt.build(
    sorted_meta,
    prefix="dc_level_",
    postfix=f"_n_components={n_components}_min_split={min_split}",
)


#%%

# class MetaNode(NodeMixin):
#     def __init__(self, name, parent=None, children=None, meta=None):
#         super().__init__()
#         self.name = name
#         self.parent = parent
#         if children:
#             self.children = children
#         self.meta = meta

#     def hierarchical_mean(self, key):
#         if self.is_leaf:
#             meta = self.meta
#             var = meta[key]
#             return np.mean(var)
#         else:
#             children = self.children
#             child_vars = [child.hierarchical_mean(key) for child in children]
#             return np.mean(child_vars)


# def get_parent_label(label):
#     if len(label) <= 1:
#         return None
#     elif label[-1] == "-":
#         return label[:-1]
#     else:  # then ends in a -number
#         return label[:-2]


# def make_node(label, node_map):
#     if label not in node_map:
#         node = MetaNode(label)
#         node_map[label] = node
#     else:
#         node = node_map[label]
#     return node


# meta = sorted_meta
# node_map = {}
# for i in range(lowest_level, -1, -1):
#     level_labels = meta[
#         f"dc_level_{i}_n_components={n_components}_min_split={min_split}"
#     ].unique()
#     for label in level_labels:
#         node = make_node(label, node_map)
#         node.meta = meta[
#             meta[f"dc_level_{i}_n_components={n_components}_min_split={min_split}"]
#             == label
#         ]
#         parent_label = get_parent_label(label)
#         if parent_label is not None:
#             parent = make_node(parent_label, node_map)
#             node.parent = parent

# root = node


#%%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
line_level = 6

ax, divider, top, _ = adjplot(
    sorted_adj,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.5, 0.5),
    sort_class=level_names[:line_level],
    item_order="new_inds",
    class_order=HUE_ORDER,
    meta=meta,
    palette=palette,
    colors=HUE_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),  # 0.2
)

left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, mt, orientation="h")

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, mt, orientation="v")
#%%

stashfig(f"adjacency-matrix-cluster_key={CLUSTER_KEY}")
