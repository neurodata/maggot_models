#%%
import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import crosstabplot, dissimilarity_clustermap, merge_axes
from giskard.utils import get_paired_inds
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.cluster import AutoGMMCluster, DivisiveCluster
from graspologic.embed import AdjacencySpectralEmbed, MultipleASE, select_svd
from graspologic.plot import pairplot
from graspologic.utils import (
    augment_diagonal,
    binarize,
    is_fully_connected,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplacian,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from src.data import DATA_PATH, DATA_VERSION, join_node_meta, load_maggot_graph
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT, add_connections, adjplot, set_theme
from src.visualization import plot_pairs
from factor_analyzer import Rotator

set_theme()
t0 = time.time()
np.random.seed(8888)

out_path = Path("./maggot_models/experiments/revamp_embed")


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
mg = mg.node_subgraph(mg.nodes[mg.nodes["selected_lcc"]].index)
mg = mg.node_subgraph(mg.nodes[mg.nodes["hemisphere"].isin(["L", "R"])].index)
# mg = mg.node_subgraph(mg.nodes[mg.nodes["predicted_pair_id"] > 1].index)
# mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
# mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
# mg.to_largest_connected_component(verbose=True)
out_degrees = np.count_nonzero(mg.sum.adj, axis=0)
in_degrees = np.count_nonzero(mg.sum.adj, axis=1)
max_in_out_degree = np.maximum(out_degrees, in_degrees)
# TODO ideally we would OOS these back in?
keep_inds = np.arange(len(mg.nodes))[max_in_out_degree > 2]
remove_ids = np.setdiff1d(mg.nodes.index, mg.nodes.index[keep_inds])
print(f"Removed {len(remove_ids)} nodes when removing pendants.")
mg.nodes = mg.nodes.iloc[keep_inds]
mg.g.remove_nodes_from(remove_ids)
mg.to_largest_connected_component(verbose=True)
mg.nodes.sort_values("hemisphere", inplace=True)
mg.nodes["_inds"] = range(len(mg.nodes))
nodes = mg.nodes

#%%

raw_adj = mg.sum.adj.copy()

left_inds = mg.nodes[mg.nodes["hemisphere"] == "L"]["_inds"]
right_inds = mg.nodes[mg.nodes["hemisphere"] == "R"]["_inds"]

left_paired_inds, right_paired_inds = get_paired_inds(
    mg.nodes, pair_key="predicted_pair", pair_id_key="predicted_pair_id"
)
right_paired_inds_shifted = right_paired_inds - len(left_inds)

has_pairing = pd.Series(
    data=np.zeros(len(mg.nodes), dtype=bool),
    index=mg.nodes.index,
    name="has_valid_predicted_pair",
)
has_pairing.iloc[left_paired_inds] = True
has_pairing.iloc[right_paired_inds] = True

join_node_meta(has_pairing, overwrite=True, fillna=False)

print(f"Number of paired nodes {2*len(left_paired_inds)}")
print(f"Number of nodes: {len(mg.nodes)}")

#%%


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
    if not isinstance(adjs, (list, tuple)):
        adjs = [adjs]
    adjs = [pass_to_ranks(a) for a in adjs]
    # adjs = [a + 1 / a.size for a in adjs]
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


def prescale_for_embed(adjs):
    norms = [np.linalg.norm(adj, ord="fro") for adj in adjs]
    mean_norm = np.mean(norms)
    adjs = [adjs[i] * mean_norm / norms[i] for i in range(len(adjs))]
    return adjs


def ase(adj, n_components=None):
    U, S, Vt = select_svd(adj, n_components=n_components, algorithm="full")
    S_sqrt = np.diag(np.sqrt(S))
    X = U @ S_sqrt
    Y = Vt.T @ S_sqrt
    return X, Y


# TODO could fight about the exact details of preprocessing here
adj = preprocess_for_embed(raw_adj)[0]
adjs = split_adj(adj)
ll_adj, rr_adj, lr_adj, rl_adj = adjs
ll_adj, rr_adj = prescale_for_embed([ll_adj, rr_adj])
lr_adj, rl_adj = prescale_for_embed([lr_adj, rl_adj])

#%%
n_components = 24  # 24 looked fine
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

#%%
missing = mg.nodes.iloc[~composite_latent.any(axis=1)][
    [
        "pair_id",
        "axon_output",
        "axon_input",
        "dendrite_input",
        "dendrite_output",
        "hemisphere",
        "_inds",
    ]
]

missing

#%%
n_final_components = 16
joint_latent, _ = ase(composite_latent, n_components=n_final_components)


def varimax(X):
    return Rotator(normalize=False).fit_transform(X)


# joint_latent = varimax(joint_latent)

joint_latent_df = pd.DataFrame(
    index=mg.nodes.index,
    data=joint_latent,
    columns=[f"latent_{i}" for i in range(joint_latent.shape[1])],
)
join_node_meta(joint_latent_df, overwrite=True)

has_embedding = pd.Series(
    index=mg.nodes.index,
    data=np.ones(len(mg.nodes), dtype=bool),
    name="has_embedding",
)
join_node_meta(has_embedding, overwrite=True, fillna=False)

#%%
plot_pairs(
    joint_latent,
    mg.nodes[CLASS_KEY].values,
    left_pair_inds=left_paired_inds,
    right_pair_inds=right_paired_inds,
    palette=palette,
    n_show=10,
)
stashfig("joint-latent")
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
        print(row_left["merge_class"])
        # raise ValueError("Pairs are not matched.")
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
plot_pairs(
    joint_latent_symmetrized,
    labels=condensed_nodes[CLASS_KEY].values,
    palette=palette,
    n_show=10,
)
stashfig("joint-latent-symmetrized")

#%%
latent_symmetrized_df = pd.DataFrame(
    data=joint_latent_symmetrized,
    index=condensed_nodes.index,
    columns=[f"latent_{i}" for i in range(joint_latent_symmetrized.shape[1])],
)
condensed_nodes = pd.concat(
    (condensed_nodes, latent_symmetrized_df), axis=1, ignore_index=False
)
condensed_nodes
condensed_nodes.to_csv(out_path / "outs/condensed_nodes.csv")


# %%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")