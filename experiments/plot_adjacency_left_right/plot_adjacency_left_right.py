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
        pathname="./maggot_models/experiments/plot_adjacency_left_right/figs",
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


meta["hemisphere"] = meta["hemisphere"].map({"L": "Left", "R": "Right"})
meta["hemisphere_num"] = meta["hemisphere"].map({"Left": 0, "Right": 1})
level_names.insert(0, "hemisphere")
level_names.insert(0, "hemisphere_num")
sorted_meta = sort_meta(
    meta, level_names, group_order=HUE_ORDER, item_order=[HUE_KEY, HUE_ORDER]
)
sort_inds = sorted_meta["inds"]
sorted_adj = adj[sort_inds][:, sort_inds]
sorted_meta["sorted_adjacency_index"] = np.arange(len(sorted_meta))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax, divider, top, _ = adjplot(
    sorted_adj,
    ax=ax,
    plot_type="scattermap",
    sizes=(0.3, 0.3),
    item_order="sorted_adjacency_index",
    sort_class=["hemisphere"],
    class_order='hemisphere_num',
    meta=sorted_meta,
    palette=palette,
    ticks=True,
    tick_fontsize=20,
    gridline_kws=dict(linewidth=0.5, color="grey", linestyle=":"),  # 0.2
)
stashfig(f"left-right-adjacency-matrix-cluster_key={CLUSTER_KEY}-hue_key={HUE_KEY}-hue_order={HUE_ORDER}")