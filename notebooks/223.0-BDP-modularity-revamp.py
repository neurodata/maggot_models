#%%

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from giskard.plot import stacked_barplot
from graspologic.partition import (
    HierarchicalCluster,
    hierarchical_leiden,
    leiden,
    modularity,
    modularity_components,
)
from graspologic.utils import symmetrize
from src.data import load_metagraph
from src.hierarchy import signal_flow
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import set_theme

set_theme()

#%% [markdown]
# ## Load data
#%%
mg = load_metagraph("G")
meta = mg.meta.copy()
meta = meta[meta["paper_clustered_neurons"]]
mg = mg.reindex(meta.index, use_ids=True)
mg = mg.make_lcc()
adj = mg.adj
mg.meta["sf"] = signal_flow(adj)

#%%
# preprocess the adjacency
sym_adj = mg.adj.copy()
threshold = 0  # smallest to keep
sym_adj[sym_adj < threshold] = 0
sym_adj = symmetrize(sym_adj)

undirected_g = nx.from_numpy_array(sym_adj)
str_arange = [f"{i}" for i in range(len(undirected_g))]
arange = np.arange(len(undirected_g))
str_node_map = dict(zip(arange, str_arange))
nx.relabel_nodes(undirected_g, str_node_map, copy=False)
nodelist = meta.index
resolution = 2.0
randomness = 0.01
n_restarts = 25
currtime = time.time()

best_modularity = -np.inf
best_partition = {}
for i in range(n_restarts):
    partition = leiden(
        undirected_g,
        resolution=resolution,
        randomness=randomness,
        check_directed=False,
        extra_forced_iterations=10,
    )
    modularity_score = modularity(
        undirected_g, partitions=partition, resolution=resolution
    )
    print(modularity_score)
    if modularity_score > best_modularity:
        best_partition = partition
        best_modularity = modularity_score

print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%


FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, format="png", dpi=300, **kws)


partition = best_partition
flat_partition = list(map(partition.get, str_arange))
meta = mg.meta
meta["partition"] = flat_partition

modularity_component_scores = modularity_components(
    undirected_g, partitions=partition, resolution=resolution
)


counts_by_cluster = pd.crosstab(meta["partition"], meta["merge_class"])

partition_order = (
    meta.groupby("partition")["sf"].mean().sort_values(ascending=False).index
)
class_order = meta.groupby("merge_class")["sf"].mean().sort_values(ascending=True).index
counts_by_cluster = counts_by_cluster.reindex(
    index=partition_order, columns=class_order
)

side_by_cluster = pd.crosstab(meta["partition"], meta["hemisphere"])
side_by_cluster = side_by_cluster.reindex(index=partition_order)


colors = sns.color_palette("deep")
side_palette = dict(zip(np.unique(meta["hemisphere"]), colors))
text_pad = 10
thickness = 0.25
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i, (idx, row) in enumerate(counts_by_cluster.iterrows()):

    stacked_barplot(row, center=i - 0.15, ax=ax, palette=palette, thickness=thickness)

    side_row = side_by_cluster.loc[idx]
    stacked_barplot(
        side_row, center=i + 0.15, ax=ax, palette=side_palette, thickness=thickness
    )

    component_score = modularity_component_scores[idx] / best_modularity
    row_height = row.sum()
    ax.text(
        i,
        row_height + text_pad,
        f"{component_score:.2f}",
        va="bottom",
        ha="center",
        size="small",
    )

ax.set_title(f"Modularity = {best_modularity:0.2f}", pad=40)
ax.set(
    xlabel="Module",
    ylabel="Number of neurons",
    xticks=[],
)
stashfig(f"modularity-clustering-resolution={resolution}")

#%%
# confusionmatrix between this clustering and the spectral one.

#%%


resolution = 1
hierarchical_partition = hierarchical_leiden(
    undirected_g, max_cluster_size=150, resolution=resolution
)

#%%


flat_hier_partition = HierarchicalCluster.final_hierarchical_clustering(
    hierarchical_partition
)
flat_partition = list(map(flat_hier_partition.get, str_arange))

#%%

meta["partition"] = flat_partition


counts_by_cluster = pd.crosstab(meta["partition"], meta["merge_class"])

partition_order = (
    meta.groupby("partition")["sf"].mean().sort_values(ascending=False).index
)
class_order = meta.groupby("merge_class")["sf"].mean().sort_values(ascending=True).index
counts_by_cluster = counts_by_cluster.reindex(
    index=partition_order, columns=class_order
)

side_by_cluster = pd.crosstab(meta["partition"], meta["hemisphere"])
side_by_cluster = side_by_cluster.reindex(index=partition_order)


colors = sns.color_palette("deep")
side_palette = dict(zip(np.unique(meta["hemisphere"]), colors))
text_pad = 10
thickness = 0.5
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i, (idx, row) in enumerate(counts_by_cluster.iterrows()):
    stacked_barplot(row, center=i, ax=ax, palette=palette, thickness=thickness)


ax.set(
    xlabel="Module",
    ylabel="Number of neurons",
    xticks=[],
)
stashfig(f"hier-modularity-clustering-resolution={resolution}")
