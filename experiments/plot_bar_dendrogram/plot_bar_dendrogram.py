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
        pathname="./maggot_models/experiments/plot_bar_dendrogram/figs",
        **kws,
    )
    savefig(
        name,
        pathname="./maggot_models/experiments/plot_bar_dendrogram/figs",
        format="pdf",
        **kws,
    )
    savefig(
        name,
        pathname="./maggot_models/experiments/plot_bar_dendrogram/figs",
        format="svg",
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
lowest_level = 8
level_names = [
    f"dc_level_{i}_n_components={n_components}_min_split={min_split}"
    for i in range(1, lowest_level + 1)
]

#%%

new_level_names = [f"dc_level_{i}" for i in range(lowest_level + 1)]

name_mapping = dict(zip(level_names, new_level_names))

meta = meta.rename(columns=name_mapping)

#%%
from giskard.plot import dendrogram_barplot

ax = dendrogram_barplot(
    meta,
    group="dc_level_",
    max_levels=7,
    hue=HUE_KEY,
    hue_order=HUE_ORDER,
    group_order=HUE_ORDER,
    orient="v",
    figsize=(4, 20),
    palette=palette,
    pad=25,
    linewidth=1,
)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.tick_params(axis="both", which="both", length=0)
ax.set_xlabel("Level")
# stashfig("bar-dendrogram")

#%%

ax = dendrogram_barplot(
    meta,
    group="dc_level_",
    max_levels=7,
    hue=HUE_KEY,
    hue_order=HUE_ORDER,
    group_order=HUE_ORDER,
    orient="h",
    figsize=(20, 5),
    palette=palette,
    pad=25,
    linewidth=1,
)
ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
ax.tick_params(axis="both", which="both", length=0)
ax.set_ylabel("Cluster level", fontsize='x-large')
stashfig("bar-dendrogram-wide", pad_inches=0)
