#%%
import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from giskard.plot import merge_axes
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import AdjacencySpectralEmbed, MultipleASE, selectSVD
from graspologic.match import GraphMatch
from graspologic.utils import (
    augment_diagonal,
    pass_to_ranks,
    remove_loops,
    to_laplacian,
)
from src.data import DATA_PATH, DATA_VERSION, load_maggot_graph, join_node_meta
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT
from src.visualization import add_connections, adjplot, set_theme
from giskard.utils import get_paired_inds

set_theme()
t0 = time.time()

out_path = Path("./maggot_models/experiments/revamp_embed_agglom")


def stashfig(name, **kws):
    savefig(
        name,
        pathname=out_path / "figs",
        **kws,
    )


#%%
CLASS_KEY = "merge_class"
palette = CLASS_COLOR_DICT
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
mg.to_largest_connected_component(verbose=True)
mg.nodes.sort_values("hemisphere", inplace=True)
mg.nodes["_inds"] = range(len(mg.nodes))
nodes = mg.nodes

#%%
from graspologic.utils import is_fully_connected

adj = mg.sum.adj
print(is_fully_connected(adj))
left_inds = mg.nodes[mg.nodes["hemisphere"] == "L"]["_inds"]
right_inds = mg.nodes[mg.nodes["hemisphere"] == "R"]["_inds"]

left_paired_inds, right_paired_inds = get_paired_inds(mg.nodes)
right_paired_inds_shifted = right_paired_inds - len(left_inds)


def preprocess_for_embed(adjs, method="ase"):
    """Preprocessing necessary prior to embedding a graph, opetates on a list

    Parameters
    ----------
    adjs : list of adjacency matrices
        [description]
    method : str, optional
        [description], by default "ase"

    Returns
    -------
    [type]
        [description]
    """
    if not isinstance(adjs, list):
        adjs = [adjs]
    adjs = [pass_to_ranks(a) for a in adjs]
    adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":  # haven't really used much. a few params to look at here
        adjs = [to_laplacian(a, form="R-DAD") for a in adjs]
    return tuple(adjs)


def split_adj(adj):
    ll_adj = adj[np.ix_(left_inds, left_inds)]
    rr_adj = adj[np.ix_(right_inds, right_inds)]
    lr_adj = adj[np.ix_(left_inds, right_inds)]
    rl_adj = adj[np.ix_(right_inds, left_inds)]
    return ll_adj, rr_adj, lr_adj, rl_adj


def ase(adj, n_components=None):
    U, S, Vt = selectSVD(adj, n_components=n_components, algorithm="full")
    S_sqrt = np.diag(np.sqrt(S))
    X = U @ S_sqrt
    Y = Vt.T @ S_sqrt
    return X, Y


adj = preprocess_for_embed(adj)[0]
ll_adj, rr_adj, lr_adj, rl_adj = split_adj(adj)

#%%
n_components = 32
X_ll, Y_ll = ase(ll_adj, n_components=n_components)
X_rr, Y_rr = ase(rr_adj, n_components=n_components)
X_lr, Y_lr = ase(lr_adj, n_components=n_components)
X_rl, Y_rl = ase(rl_adj, n_components=n_components)

#%%


def joint_procrustes(
    data1, data2, method="orthogonal", paired_inds1=None, paired_inds2=None, swap=False
):
    n = len(data1[0])
    if method == "orthogonal":
        procruster = OrthogonalProcrustes()
    elif method == "seedless":
        procruster = SeedlessProcrustes(init="sign_flips")
    elif method == "seedless-oracle":
        X1_paired = data1[0][paired_inds1, :]
        X2_paired = data2[0][paired_inds2, :]
        if swap:
            Y1_paired = data1[1][paired_inds2, :]
            Y2_paired = data2[1][paired_inds1, :]
        else:
            Y1_paired = data1[1][paired_inds1, :]
            Y2_paired = data2[1][paired_inds2, :]
        data1_paired = np.concatenate((X1_paired, Y1_paired), axis=0)
        data2_paired = np.concatenate((X2_paired, Y2_paired), axis=0)
        op = OrthogonalProcrustes()
        op.fit(data1_paired, data2_paired)
        procruster = SeedlessProcrustes(
            init="custom",
            initial_Q=op.Q_,
            optimal_transport_eps=1.0,
            optimal_transport_num_reps=100,
            iterative_num_reps=10,
        )
    data1 = np.concatenate(data1, axis=0)
    data2 = np.concatenate(data2, axis=0)
    currtime = time.time()
    data1_mapped = procruster.fit_transform(data1, data2)
    print(f"{time.time() - currtime:.3f} seconds elapsed for SeedlessProcrustes.")
    data1 = (data1_mapped[:n], data1_mapped[n:])
    return data1


X_ll, Y_ll = joint_procrustes(
    (X_ll, Y_ll),
    (X_rr, Y_rr),
    method="seedless-oracle",
    paired_inds1=left_paired_inds,
    paired_inds2=right_paired_inds_shifted,
)

#%%
from src.visualization import plot_pairs

composite_ipsi_X = np.concatenate((X_ll, X_rr), axis=0)
plot_pairs(
    composite_ipsi_X,
    mg.nodes[CLASS_KEY].values,
    left_pair_inds=left_paired_inds,
    right_pair_inds=right_paired_inds,
    palette=palette,
    n_show=6,
)

#%%
X_lr, Y_lr = joint_procrustes(
    (X_lr, Y_lr),
    (X_rl, Y_rl),
    method="seedless-oracle",
    paired_inds1=left_paired_inds,
    paired_inds2=right_paired_inds_shifted,
    swap=True,
)

#%%
composite_contra_X = np.concatenate((X_lr, X_rl), axis=0)
plot_pairs(
    composite_contra_X,
    mg.nodes[CLASS_KEY].values,
    left_pair_inds=left_paired_inds,
    right_pair_inds=right_paired_inds,
    palette=palette,
    n_show=6,
)

#%%
composite_ipsi_Y = np.concatenate((Y_ll, Y_rr), axis=0)
composite_contra_Y = np.concatenate((Y_rl, Y_lr), axis=0)

composite_latent = np.concatenate(
    (composite_ipsi_X, composite_ipsi_Y, composite_contra_X, composite_contra_Y), axis=1
)
mg.nodes.iloc[~composite_latent.any(axis=1)]

#%%
n_final_components = 20
joint_latent, _ = ase(composite_latent, n_components=n_final_components)

#%%
plot_pairs(
    joint_latent,
    mg.nodes[CLASS_KEY].values,
    left_pair_inds=left_paired_inds,
    right_pair_inds=right_paired_inds,
    palette=palette,
    n_show=6,
)

#%%
~joint_latent.any(axis=1)

#%%

X = joint_latent
left_unpaired_inds = np.setdiff1d(left_inds, left_paired_inds)
right_unpaired_inds = np.setdiff1d(right_inds, right_paired_inds)
mean_X = (X[left_paired_inds] + X[right_paired_inds]) / 2
left_unpaired_X = X[left_unpaired_inds]
right_unpaired_X = X[right_unpaired_inds]
joint_latent_symmetrized = np.concatenate(
    (mean_X, left_unpaired_X, right_unpaired_X), axis=0
)

plot_pairs(
    joint_latent_symmetrized,
    np.ones(len(joint_latent_symmetrized)),
    n_show=6,
)

#%%
nodes = mg.nodes.copy()
nodes.index.name = "skeleton_id"
nodes = nodes.reset_index()
left_paired_nodes = nodes.iloc[left_paired_inds]
right_paired_nodes = nodes.iloc[right_paired_inds]
left_unpaired_nodes = nodes.iloc[left_unpaired_inds]
right_unpaired_nodes = nodes.iloc[right_unpaired_inds]

n_pairs = len(left_paired_inds)
cols = ["skeleton_id", "name", "merge_class", "pair_id", "sum_signal_flow"]
new_rows = []
max_id = 0
for i in range(n_pairs):
    row_left = left_paired_nodes.iloc[i][cols]
    row_right = right_paired_nodes.iloc[i][cols]
    if row_left["pair_id"] != row_right["pair_id"]:
        raise ValueError("Pairs are not matched.")
    new_row = {}
    new_row["skeleton_ids"] = [row_left["skeleton_id"], row_right["skeleton_id"]]
    new_row["name"] = row_left["name"]
    new_row["merge_class"] = row_left["merge_class"]
    new_row["unit_id"] = row_left["pair_id"]
    new_row["sum_signal_flow"] = (
        row_left["sum_signal_flow"] + row_right["sum_signal_flow"]
    ) / 2
    new_rows.append(new_row)
    max_id = row_left["pair_id"]

n_left_unpaired = len(left_unpaired_nodes)
for i in range(n_left_unpaired):
    max_id += 1
    row = left_unpaired_nodes.iloc[i][cols]
    new_row = {}
    new_row["skeleton_ids"] = [row["skeleton_id"]]
    new_row["name"] = row["name"]
    new_row["merge_class"] = row["merge_class"]
    new_row["unit_id"] = max_id
    new_row["sum_signal_flow"] = row["sum_signal_flow"]
    new_rows.append(new_row)


for i in range(len(right_unpaired_nodes)):
    max_id += 1
    row = right_unpaired_nodes.iloc[i][cols]
    new_row = {}
    new_row["skeleton_ids"] = [row["skeleton_id"]]
    new_row["name"] = row["name"]
    new_row["merge_class"] = row["merge_class"]
    new_row["unit_id"] = max_id
    new_row["sum_signal_flow"] = row["sum_signal_flow"]
    new_rows.append(new_row)

condensed_nodes = pd.DataFrame(new_rows)

#%%
condensed_nodes.iloc[~joint_latent_symmetrized.any(axis=1)]

#%%
from graspologic.plot import pairplot

# pg = pairplot(
#     joint_latent_symmetrized[:, :4],
#
#     palette=palette,
# )
# pg._legend.remove()
plot_pairs(
    joint_latent_symmetrized, labels=condensed_nodes[CLASS_KEY].values, palette=palette
)

#%%
from sklearn.cluster import AgglomerativeClustering
from giskard.plot import crosstabplot


def cluster_crosstabplot(
    nodes,
    group="cluster_labels",
    order="sum_signal_flow",
    hue="merge_class",
    palette=None,
):
    group_order = (
        nodes.groupby(group)[order].agg(np.median).sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    crosstabplot(
        nodes,
        group=group,
        group_order=group_order,
        hue=hue,
        hue_order=order,
        palette=palette,
        outline=True,
        thickness=0.5,
        ax=ax,
    )
    ax.set(xticks=[], xlabel="Cluster")
    return fig, ax


def uncondense_series(condensed_nodes, nodes, key):
    for idx, row in condensed_nodes.iterrows():
        skids = row["skeleton_ids"]
        for skid in skids:
            nodes.loc[skid, key] = row[key]


for n_clusters in [40, 50, 60, 70, 80, 90, 100, 110, 120]:
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="cosine", linkage="average"
    )
    pred_labels = agg.fit_predict(joint_latent_symmetrized)
    key = f"cluster_agglom_K={n_clusters}"
    condensed_nodes[key] = pred_labels
    uncondense_series(condensed_nodes, nodes, key)
    join_node_meta(nodes[key], overwrite=True)
    fig, ax = cluster_crosstabplot(
        condensed_nodes,
        group=key,
        palette=palette,
        hue=CLASS_KEY,
        order="sum_signal_flow",
    )
    ax.set_title(f"# clusters = {n_clusters}")

#%%
n_clusters = 80
plot_pairs(
    joint_latent_symmetrized,
    labels=condensed_nodes[f"cluster_agglom_K={n_clusters}"].values,
)

#%%
from graspologic.utils import symmetrize
from sklearn.metrics import pairwise_distances
from giskard.plot import dissimilarity_clustermap

metric = "cosine"
linkage = "average"
distances = symmetrize(pairwise_distances(joint_latent_symmetrized, metric=metric))
dissimilarity_clustermap(
    distances, colors=condensed_nodes["merge_class"], palette=palette, method=linkage
)

#%%
# X_lr, Y_lr = joint_procrustes(
#     (X_lr, Y_lr),
#     (X_rl, Y_rl),
#     method="seedless-oracle",
#     paired_inds1=left_paired_inds,
#     paired_inds2=right_paired_inds,
# )
