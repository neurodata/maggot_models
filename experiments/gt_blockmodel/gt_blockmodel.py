#%%
import datetime
import time
from pathlib import Path

import graph_tool as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import crosstabplot, dissimilarity_clustermap
from giskard.utils import get_paired_inds
from graph_tool.collection import data
from graph_tool.draw import draw_hierarchy
from graph_tool.inference import (
    BlockState,
    mcmc_anneal,
    minimize_blockmodel_dl,
    minimize_nested_blockmodel_dl,
)
from graspologic.plot.plot_matrix import scattermap
from graspologic.utils import remap_labels, symmetrize
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_cluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances
from src.data import DATA_PATH, DATA_VERSION, join_node_meta, load_maggot_graph
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import adjplot, set_theme


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/gt_blockmodel/figs",
        **kws,
    )


set_theme()
t0 = time.time()

#%%


def run_minimize_blockmodel(g, n_init=1, use_weights=False, **kwargs):
    if use_weights:
        y = g.ep.weight.copy()
        y.a = np.log(y.a)
        min_state = minimize_blockmodel_dl(
            g,
            deg_corr=True,
            state_args=dict(recs=[y], rec_types=["real-normal"]),
            **kwargs,
        )
    else:
        min_entropy = np.inf
        min_state = None
        for i in range(n_init):
            # NOTE this isn't doing what I'd have expected as the first block state is
            # always the one that is kept
            new_state = minimize_blockmodel_dl(g, deg_corr=True, **kwargs)
            new_state_entropy = new_state.entropy()
            if new_state_entropy < min_entropy:
                min_state = new_state
                min_entropy = new_state_entropy

    block_series = get_block_labels(min_state)
    return block_series


def get_block_labels(state):
    blocks = list(state.get_blocks())
    verts = g.get_vertices()

    block_map = {}

    for v, b in zip(verts, blocks):
        cell_id = int(g.vertex_properties["name"][v])
        block_map[cell_id] = int(b)

    block_series = pd.Series(block_map)
    return block_series


def symmetrize_labels(nodes, key):
    lp_inds, rp_inds = get_paired_inds(
        nodes, pair_key="predicted_pair", pair_id_key="predicted_pair_id"
    )
    left_cluster_labels = nodes.iloc[lp_inds][key]
    right_cluster_labels = nodes.iloc[rp_inds][key]
    right_remapped_labels, cluster_map = remap_labels(
        left_cluster_labels, right_cluster_labels, return_map=True
    )
    nodes.loc[nodes.index[rp_inds], key] = right_remapped_labels


def cluster_crosstabplot(nodes, key="cluster_labels"):
    group_order = (
        nodes.groupby(key)["sum_signal_flow"]
        .apply(np.median)
        .sort_values(ascending=False)
        .index
    )
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    crosstabplot(
        nodes[nodes["hemisphere"] == "L"],
        group=key,
        group_order=group_order,
        hue="merge_class",
        hue_order="sum_signal_flow",
        palette=palette,
        outline=True,
        shift=-0.2,
        thickness=0.25,
        ax=ax,
    )
    crosstabplot(
        nodes[nodes["hemisphere"] == "R"],
        group=key,
        group_order=group_order,
        hue="merge_class",
        hue_order="sum_signal_flow",
        palette=palette,
        outline=True,
        shift=0.2,
        thickness=0.25,
        ax=ax,
    )
    ax.set(xticks=[], xlabel="Cluster")


#%%
data_path = DATA_PATH / DATA_VERSION
g = gt.load_graph_from_csv(
    str(data_path / "G_edgelist.txt"),
    directed=True,
    csv_options=dict(delimiter=" "),
    eprop_types=["float"],
    eprop_names=["weight"],
)
gt_node_names = [int(i) for i in g.vp["name"]]
mg = load_maggot_graph()
nodes = mg.nodes
remove_ids = nodes[~nodes["has_predicted_matching"]].index.values
nodes = nodes[nodes["has_predicted_matching"]].copy()
print(len(nodes))
remove_inds = []
for i, remove_id in enumerate(remove_ids):
    ind = np.where(gt_node_names == remove_id)[0]
    if len(ind) > 0:
        remove_inds.append(ind[0])
g.remove_vertex(remove_inds)
gt_node_names = [int(i) for i in g.vp["name"]]
nodes = nodes.reindex(gt_node_names)


#%%
# lp_inds, rp_inds = get_paired_inds(nodes)
# state = minimize_blockmodel_dl(g, deg_corr=True)
# state_entropy = state.entropy()
# print(state_entropy)
# currtime = time.time()
# dS, nattempts, nmoves = state.multiflip_mcmc_sweep(niter=1000)
# print(f"{time.time() - currtime:.3f} seconds elapsed.")
# print(dS)

#%%
# labels = get_block_labels(state)
# nodes["cluster_labels"] = labels
# symmetrize_labels(nodes, "cluster_labels")
# cluster_crosstabplot(nodes)

#%%
# n_unique_by_pair = nodes.groupby("pair_id")["cluster_labels"].nunique()
# n_unique_by_pair = n_unique_by_pair[n_unique_by_pair.index != -1]
# p_same_cluster = (n_unique_by_pair == 1).mean()
# print(p_same_cluster)

#%%
# print(dir(g.vertex_properties))

# did not like results of discrete-binomial as much as unweighted
# discrete poisson was similar
# discrete geometric maybe the best of the weighted ones
# colors = g.new_vertex_property("string")
# colors.set_2d_array(nodes["merge_class"].map(palette.get).values.astype(str))

# state = minimize_nested_blockmodel_dl(g, deg_corr=True, verbose=True)
# draw_hierarchy(state, vertex_fill_color=colors, output="maggot_nested_mdl.pdf")


#%%

# vprop_int = g.new_vertex_property("int")
# vprop_int.get_array()[:] = nodes["agglom_labels_t=3_n_components=64"].values.astype(int)
# state = BlockState(g, b=vprop_int)
# currtime = time.time()
# out = mcmc_anneal(
#     state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10)
# )
# print(out)
# print(f"{time.time() - currtime:.3f} seconds elapsed.")
# labels = get_block_labels(state)
# labels.name = "mcmc_anneal_from_agglom_labels_t=2.5_n_components=64"
#%%

currtime = time.time()
labels = run_minimize_blockmodel(
    g,
    n_init=1,
)
labels.name = "gt_blockmodel_labels"
print(f"{time.time() - currtime:.3f} seconds elapsed to run minimize_blockmodel.")


#%%
# nodes["cluster_labels"] = labels


symmetrize_labels(nodes, "gt_blockmodel_labels")

labels.name = "gt_blockmodel_labels"
join_node_meta(labels, overwrite=True)


#%%
nodes["cluster_labels"] = nodes["gt_blockmodel_labels"]
group_order = (
    nodes.groupby("cluster_labels")["sum_signal_flow"]
    .apply(np.median)
    .sort_values(ascending=False)
    .index
)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
crosstabplot(
    nodes[nodes["hemisphere"] == "L"],
    group="cluster_labels",
    group_order=group_order,
    hue="merge_class",
    hue_order="sum_signal_flow",
    palette=palette,
    outline=True,
    shift=-0.2,
    thickness=0.25,
    ax=ax,
)
crosstabplot(
    nodes[nodes["hemisphere"] == "R"],
    group="cluster_labels",
    group_order=group_order,
    hue="merge_class",
    hue_order="sum_signal_flow",
    palette=palette,
    outline=True,
    shift=0.2,
    thickness=0.25,
    ax=ax,
)
ax.set(xticks=[], xlabel="Cluster")
stashfig("crosstabplot_gt_blockmodel")
#%%
for B in [40, 50]:  # 60, 70, 80]:
    t = time.time()
    state = BlockState(g, B=B)
    state.multiflip_mcmc_sweep(niter=1000, verbose=True)
    print(state.get_B())
    print(state.get_nonempty_B())
    print(state.get_Be())
    print(time.time() - t)
    print()

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")