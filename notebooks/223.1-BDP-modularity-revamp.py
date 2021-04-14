#%%

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from giskard.plot import (
    MatrixGrid,
    axis_on,
    confusionplot,
    crosstabplot,
    soft_axis_off,
    stacked_barplot,
)
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

FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, format="png", dpi=300, **kws)


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

# convert datatype
# NOTE: have to do some weirdness because leiden only wants str keys right now
undirected_g = nx.from_numpy_array(sym_adj)
str_arange = [f"{i}" for i in range(len(undirected_g))]
arange = np.arange(len(undirected_g))
str_node_map = dict(zip(arange, str_arange))
nx.relabel_nodes(undirected_g, str_node_map, copy=False)
nodelist = meta.index


def optimize_leiden(g, n_restarts=25, resolution=1, randomoness=0.001):
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
        if modularity_score > best_modularity:
            best_partition = partition
            best_modularity = modularity_score
    return best_partition, best_modularity


#%%
n_restarts = 25
randomness = 0.01
resolutions = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
rows = []

currtime = time.time()
for resolution in tqdm(resolutions):
    partition, modularity_score = optimize_leiden(
        undirected_g,
        n_restarts=n_restarts,
        resolution=resolution,
        randomoness=randomness,
    )
    modularity_component_scores = modularity_components(
        undirected_g, partitions=partition, resolution=resolution
    )
    flat_partition = list(map(partition.get, str_arange))
    row = {
        "partition": flat_partition,
        "modularity_score": modularity_score,
        "resolution": resolution,
        "modularity_component_scores": modularity_component_scores,
    }
    rows.append(row)

print(f"{time.time() - currtime:.3f} seconds elapsed.")
results = pd.DataFrame(rows)
results

#%%

meta = mg.meta

colors = sns.color_palette("deep")
side_palette = dict(zip(np.unique(meta["hemisphere"]), colors))
text_pad = 10
thickness = 0.25
hatch_map = {"L": "..", "R": None}


def plot_modules(meta, result, hue="merge_class", show_component_score=False):
    partition = result["partition"]
    modularity_score = result["modularity_score"]
    resolution = result["resolution"]
    modularity_component_scores = result["modularity_component_scores"]
    meta = meta.copy()
    meta["partition"] = partition

    counts_by_cluster = pd.crosstab(meta["partition"], meta["merge_class"])
    partition_order = (
        meta.groupby("partition")["sf"].mean().sort_values(ascending=False).index
    )
    class_order = (
        meta.groupby("merge_class")["sf"].mean().sort_values(ascending=True).index
    )
    counts_by_cluster = counts_by_cluster.reindex(
        index=partition_order, columns=class_order
    )

    side_by_cluster = pd.crosstab(meta["partition"], meta["hemisphere"])
    side_by_cluster = side_by_cluster.reindex(index=partition_order)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, (idx, row) in enumerate(counts_by_cluster.iterrows()):

        stacked_barplot(
            row, center=i - 0.15, ax=ax, palette=palette, thickness=thickness
        )

        side_row = side_by_cluster.loc[idx]
        stacked_barplot(
            side_row,
            center=i + 0.15,
            ax=ax,
            thickness=thickness,
            colors="lightgrey",
            hatch_map=hatch_map,
            linewidth=1,
            edgecolor="black",
        )

        if show_component_score:
            component_score = modularity_component_scores[idx] / modularity_score
            row_height = row.sum()
            ax.text(
                i,
                row_height + text_pad,
                f"{component_score:.2f}",
                va="bottom",
                ha="center",
                size="small",
            )

    ax.set_title(
        f"Modularity = {modularity_score:0.2f}, resolution = {resolution}", pad=40
    )
    ax.set(
        xlabel="Module",
        ylabel="Number of neurons",
        xticks=[],
    )


for _, row in results.iterrows():
    plot_modules(meta, row)
    stashfig(f"modularity-clustering-resolution={row['resolution']}")

# #%%
# # confusionmatrix between this clustering and the spectral one.

# #%%
# resolution = 1
# hierarchical_partition = hierarchical_leiden(
#     undirected_g, max_cluster_size=150, resolution=resolution
# )

# #%%


# flat_hier_partition = HierarchicalCluster.final_hierarchical_clustering(
#     hierarchical_partition
# )
# flat_partition = list(map(flat_hier_partition.get, str_arange))
#%%


meta = mg.meta.copy()
meta["partition"] = row["partition"]
crosstabplot(
    meta,
    group="partition",
    hue="merge_class",
    group_order="sf",
    hue_order="sf",
    palette=palette,
)

#%%


cluster_meta_path = "maggot_models/experiments/matched_subgraph_omni_cluster/outs/meta-method=color_iso-d=8-bic_ratio=1-min_split=32.csv"
cluster_meta = pd.read_csv(cluster_meta_path, index_col=0)
level = 6
cluster_meta = cluster_meta.reindex(meta.index)
spectral_labels = cluster_meta[f"lvl{level}_labels"].fillna("na")
meta["spectral"] = spectral_labels
module_labels = meta["partition"].astype(str)

spectral_order = (
    meta.groupby("spectral")["sf"].mean().sort_values(ascending=False).index
)
module_order = meta.groupby("partition")["sf"].mean().sort_values(ascending=False).index
conf_mat = pd.crosstab(module_labels, spectral_labels)
conf_mat = conf_mat.reindex(index=module_order.astype(str), columns=spectral_order)
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
matgrid = MatrixGrid(spines=True)
ax = matgrid.ax
confusionplot(
    conf_mat,
    annot=False,
    title=False,
    xticklabels=False,
    yticklabels=False,
    ax=ax,
    cbar=False,
)
left_ax = matgrid.append_axes("left", size="20%")
crosstabplot(
    meta,
    group="partition",
    group_order=conf_mat.index.astype(int),
    hue="merge_class",
    hue_order="sf",
    ax=left_ax,
    palette=palette,
    orient="h",
    shift=0.5,
    thickness=0.7
)
left_ax.invert_xaxis()
soft_axis_off(left_ax)

top_ax = matgrid.append_axes("top", size="20%")
crosstabplot(
    meta,
    group="spectral",
    group_order=conf_mat.columns,
    hue="merge_class",
    hue_order="sf",
    ax=top_ax,
    palette=palette,
    orient="v",
    shift=0.5,
    thickness=0.7,
    normalize=False,
)

soft_axis_off(top_ax)
axis_on(ax)
stashfig("module-spectral-confusion")
