#%%
import time

import matplotlib.pyplot as plt
import numpy as np
from giskard.hierarchy import BaseNetworkTree
from giskard.plot import plot_dendrogram
from src.data import load_maggot_graph, load_palette
from src.io import savefig
from src.visualization import adjplot, set_theme, ORDER_KEY, HUE_KEY

print("ORDER_KEY = ", ORDER_KEY)
print("HUE_KEY = ", HUE_KEY)

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


palette = load_palette()


#%%


meta = nodes.copy()
meta["inds"] = range(len(meta))
lowest_level = 7
level_names = [
    f"dc_level_{i}_n_components={n_components}_min_split={min_split}"
    for i in range(lowest_level + 1)
]
meta[level_names] = meta[level_names].astype("Int64")
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
    meta, level_names, group_order=ORDER_KEY, item_order=[HUE_KEY, ORDER_KEY]
)
sort_inds = sorted_meta["inds"]
sorted_adj = adj[sort_inds][:, sort_inds]
sorted_meta["sorted_adjacency_index"] = np.arange(len(sorted_meta))


def is_consecutive(arr):
    return np.all(arr[1:] == arr[:-1] + 1)


class MetaTree(BaseNetworkTree):
    def __init__(self):
        super().__init__()

    def build(self, node_data, prefix="", postfix="", max_depth=7):
        if self.is_root and ("adjacency_index" not in node_data.columns):
            node_data = node_data.copy()
            node_data["adjacency_index"] = range(len(node_data))
        self._index = node_data.index
        self._node_data = node_data
        if self.depth <= max_depth:
            key = prefix + f"{self.depth}" + postfix
            if key in node_data:
                groups = node_data.groupby(key, sort=False, dropna=False)
                for name, group in groups:
                    consec = is_consecutive(group["adjacency_index"].values)
                    if not consec:
                        print(f"splitting on {key}={name}")
                        print(group["adjacency_index"].values)
                        raise ValueError()

                    child = MetaTree()
                    child.parent = self
                    child.build(group, prefix=prefix, postfix=postfix)


mt = MetaTree()
mt.build(
    sorted_meta,
    prefix="dc_level_",
    postfix=f"_n_components={n_components}_min_split={min_split}",
    max_depth=7
)

#%%
root = mt
for node in root.leaves:
    indices = node._node_data["adjacency_index"]
    i_max = indices.max()
    i_min = indices.min()
    arange = np.arange(i_min, i_max + 1)
    if len(arange) != len(indices) or (arange != indices).any():
        print("not sorted")
        print(node._node_data)
        break
    else:
        print("sorted")
#%%
cols = [
    "dc_level_0_n_components=10_min_split=32",
    "dc_level_1_n_components=10_min_split=32",
    "dc_level_2_n_components=10_min_split=32",
    "dc_level_3_n_components=10_min_split=32",
    "dc_level_4_n_components=10_min_split=32",
    "dc_level_5_n_components=10_min_split=32",
    "dc_level_6_n_components=10_min_split=32",
    "dc_level_7_n_components=10_min_split=32",
]
unsorts = sorted_meta[sorted_meta["sorted_adjacency_index"] == 54]
unsorts[cols].values

#%%
sorted_meta[sorted_meta["sorted_adjacency_index"] == 54][cols].values

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
    class_order=ORDER_KEY,
    meta=sorted_meta,
    palette=palette,
    colors=HUE_KEY,
    ticks=False,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),  # 0.2
)

left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, mt, orientation="h")

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, mt, orientation="v")

stashfig(
    f"adjacency-matrix-cluster_key={CLUSTER_KEY}-hue_key={HUE_KEY}-ORDER_KEY={ORDER_KEY}",
    dpi=600,
)

# %%
