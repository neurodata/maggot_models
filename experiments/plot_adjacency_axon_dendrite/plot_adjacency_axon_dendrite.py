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
        pathname="./maggot_models/experiments/plot_adjacency_axon_dendrite/figs",
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

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
line_level = 4

ax, divider, top, _ = adjplot(
    sorted_adj,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.3, 0.3),
    sort_class=level_names[:line_level],
    item_order="sorted_adjacency_index",
    class_order=HUE_ORDER,
    meta=sorted_meta,
    palette=palette,
    colors=HUE_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0, color="grey", linestyle=":"),  # 0.2
)

# stashfig(
#     f"axon-dendrite-adjacency-matrix-cluster_key={CLUSTER_KEY}-hue_key={HUE_KEY}-hue_order={HUE_ORDER}"
# )
#%%

import seaborn as sns

fig, axs = plt.subplots(
    2,
    2,
    figsize=(19.5, 20),
    gridspec_kw=dict(hspace=0, wspace=0),
)
graph_types = ["ad", "aa", "dd", "da"]
edge_type_palette = dict(zip(graph_types, sns.color_palette("deep")))

for i, graph_type in enumerate(graph_types):
    adj = mg.to_edge_type_graph(graph_type).adj
    sorted_adj = adj[sort_inds][:, sort_inds]
    ax = axs.flat[i]
    adjplot(
        sorted_adj,
        ax=ax,
        plot_type="scattermap",
        sizes=(0.75, 0.75),
        sort_class=level_names[:line_level],
        item_order="sorted_adjacency_index",
        class_order=HUE_ORDER,
        meta=sorted_meta,
        # palette=palette,
        # colors=HUE_KEY,
        ticks=False,
        gridline_kws=dict(linewidth=0, color="grey", linestyle=":"),  # 0.2
        color=edge_type_palette[graph_type],
    )

fontsize = "xx-large"
axs[0, 0].set_title("Axon", fontsize=fontsize)
axs[0, 0].set_ylabel("Axon", fontsize=fontsize)
axs[0, 1].set_title("Dendrite", fontsize=fontsize)
axs[1, 0].set_ylabel("Dendrite", fontsize=fontsize)
stashfig(
    f"axon-dendrite-adjacency-matrix-cluster_key={CLUSTER_KEY}-hue_key={HUE_KEY}-hue_order={HUE_ORDER}"
)