# %% [markdown]
# #
import os
from itertools import chain

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from graspy.cluster import AutoGMMCluster
from graspy.utils import pass_to_ranks
from src.data import load_metagraph
from src.graph import MetaGraph, preprocess
from src.io import savecsv, savefig, saveskels
from src.traverse import (
    Cascade,
    RandomWalk,
    TraverseDispatcher,
    cascades_from_node,
    generate_cascade_tree,
    generate_random_walks,
    path_to_visits,
    to_markov_matrix,
    to_path_graph,
    to_transmission_matrix,
)
from src.visualization import (
    CLASS_COLOR_DICT,
    barplot_text,
    draw_networkx_nice,
    draw_separators,
    matrixplot,
    remove_shared_ax,
    remove_spines,
    screeplot,
    sort_meta,
    stacked_barplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

sns.set_context("talk")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


#%% Load and preprocess the data

VERSION = "2020-03-09"
print(f"Using version {VERSION}")

graph_type = "Gad"
threshold = 0
weight = "weight"
mg = load_metagraph(graph_type, VERSION)
mg = preprocess(
    mg,
    threshold=threshold,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight=weight,
)
print(f"Preprocessed graph {graph_type} with threshold={threshold}, weight={weight}")


inout = "sensory_to_out"
if inout == "sensory_to_out":
    out_classes = [
        "O_dSEZ",
        "O_dSEZ;CN",
        "O_dSEZ;LHN",
        "O_dVNC",
        "O_dVNC;O_RG",
        "O_dVNC;CN",
        "O_RG",
        "O_dUnk",
        "O_RG-IPC",
        "O_RG-ITP",
        "O_RG-CA-LP",
    ]
    from_groups = [
        ("sens-ORN",),
        ("sens-photoRh5", "sens-photoRh6"),
        ("sens-MN",),
        ("sens-thermo",),
        ("sens-vtd",),
        ("sens-AN",),
    ]
    from_group_names = ["Odor", "Photo", "MN", "Temp", "VTD", "AN"]

if inout == "out_to_sensory":
    from_groups = [
        ("motor-mAN", "motormVAN", "motor-mPaN"),
        ("O_dSEZ", "O_dVNC;O_dSEZ", "O_dSEZ;CN", "LHN;O_dSEZ"),
        ("O_dVNC", "O_dVNC;CN", "O_RG;O_dVNC", "O_dVNC;O_dSEZ"),
        ("O_RG", "O_RG-IPC", "O_RG-ITP", "O_RG-CA-LP", "O_RG;O_dVNC"),
        ("O_dUnk",),
    ]
    from_group_names = ["Motor", "SEZ", "VNC", "RG", "dUnk"]
    out_classes = [
        "sens-ORN",
        "sens-photoRh5",
        "sens-photoRh6",
        "sens-MN",
        "sens-thermo",
        "sens-vtd",
        "sens-AN",
    ]

from_classes = list(chain.from_iterable(from_groups))  # make this a flat list

class_key = "Merge Class"

adj = nx.to_numpy_array(mg.g, weight=weight, nodelist=mg.meta.index.values)
n_verts = len(adj)
meta = mg.meta.copy()
g = mg.g.copy()
meta["inds"] = range(len(meta))

from_inds = meta[meta[class_key].isin(from_classes)]["inds"].values
out_inds = meta[meta[class_key].isin(out_classes)]["inds"].values
ind_map = dict(zip(meta.index, meta["inds"]))
g = nx.relabel_nodes(g, ind_map, copy=True)
out_ind_map = dict(zip(out_inds, range(len(out_inds))))

# %% [markdown]
# ## Organize metadata for the neurons
ids = pd.Series(index=meta["inds"], data=meta.index, name="id")
to_class = ids.map(meta["Merge Class"])
to_class.name = "to_class"
colors = to_class.map(CLASS_COLOR_DICT)
colors.name = "color"
col_df = pd.concat([ids, to_class, colors], axis=1)

# %% [markdown]
# # Setup to generate cascades/random walks


def hist_from_traversal(
    transition_probs,
    start_nodes,
    traverse=Cascade,
    stop_nodes=[],
    allow_loops=False,
    n_init=100,
    simultaneous=True,
    max_hops=10,
):
    dispatcher = TraverseDispatcher(
        traverse,
        transition_probs,
        allow_loops=allow_loops,
        n_init=n_init,
        max_hops=max_hops,
        stop_nodes=stop_nodes,
    )
    if simultaneous:
        hop_hist = dispatcher.start(start_nodes)
    else:
        n_verts = len(transition_probs)
        hop_hist = np.zeros((n_verts, max_hops))
        for s in start_nodes:
            hop_hist += dispatcher.start(s)
    return hop_hist


from sklearn.neighbors import NearestNeighbors


def compute_neighbors_at_k(X, left_inds, right_inds, k_max=10, metric="euclidean"):
    nn = NearestNeighbors(radius=0, n_neighbors=k_max + 1, metric=metric)
    nn.fit(X)
    neigh_dist, neigh_inds = nn.kneighbors(X)
    is_neighbor_mat = np.zeros((X.shape[0], k_max), dtype=bool)
    for left_ind, right_ind in zip(left_inds, right_inds):
        left_neigh_inds = neigh_inds[left_ind]
        right_neigh_inds = neigh_inds[right_ind]
        for k in range(k_max):
            if right_ind in left_neigh_inds[: k + 2]:
                is_neighbor_mat[left_ind, k] = True
            if left_ind in right_neigh_inds[: k + 2]:
                is_neighbor_mat[right_ind, k] = True

    neighbors_at_k = np.sum(is_neighbor_mat, axis=0) / is_neighbor_mat.shape[0]
    return neighbors_at_k


def get_paired_inds(meta):
    pair_meta = meta[meta["Pair"] != -1].copy()
    pair_group_size = pair_meta.groupby("Pair ID").size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta["Pair ID"].isin(remove_pairs)]
    assert pair_meta.groupby("Pair ID").size().min() == 2
    pair_meta.sort_values(["Pair ID", "Hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["Hemisphere"] == "L"]["inds"]
    rp_inds = pair_meta[pair_meta["Hemisphere"] == "R"]["inds"]
    assert (
        meta.iloc[lp_inds]["Pair ID"].values == meta.iloc[rp_inds]["Pair ID"].values
    ).all()
    return lp_inds, rp_inds


# traversal parameters
transmission_probs = np.linspace(0.02, 0.1, 10)
n_init = 10
seed = 8888
max_hops = 10
simultaneous = False
use_stop_nodes = False
method = "cascade"

if use_stop_nodes:
    stop_nodes = out_inds
else:
    stop_nodes = []
cascade_kws = {
    "n_init": n_init,
    "max_hops": max_hops,
    "stop_nodes": stop_nodes,
    "allow_loops": False,
    "simultaneous": simultaneous,
}

basename = f"-n_init={n_init}-max_hops={max_hops}-simult={simultaneous}-stopnodes={use_stop_nodes}"

np.random.seed(seed)

lp_inds, rp_inds = get_paired_inds(meta)

neighbors = []
for p in transmission_probs:
    print()
    print(p)
    if method == "cascade":
        transition_probs = to_transmission_matrix(adj, p)
        traverse = Cascade
    elif method == "random-walk":
        transition_probs = to_markov_matrix(adj)
        traverse = RandomWalk

    fg_hop_hists = []
    for fg, fg_name in zip(from_groups, from_group_names):
        print(fg_name)

        # running the cascadese
        from_inds = meta[meta[class_key].isin(fg)]["inds"].values
        hop_hist = hist_from_traversal(transition_probs, from_inds, **cascade_kws).T
        fg_hop_hists.append(hop_hist)

    all_hop_hist = np.concatenate(fg_hop_hists, axis=0).T
    neighbors_at_k = compute_neighbors_at_k(all_hop_hist, lp_inds, rp_inds)
    neighbors.append(neighbors_at_k)


# %% [markdown]
# ##
neighbors = np.array(neighbors)
neighbor_df = pd.DataFrame(data=neighbors, columns=np.arange(1, neighbors.shape[1] + 1))
neighbor_df["p"] = transmission_probs

neighbor_df = neighbor_df.melt(id_vars="p", var_name="k")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(data=neighbor_df, x="k", y="value", hue="p")
ax.set_ylabel("P(Pair w/in k)")
stashfig("pair-knn-all")

# %% [markdown]
# ##

nb_df = neighbor_df[neighbor_df["k"] == 10]
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.scatterplot(data=nb_df, x="p", y="value")
ax.set_ylabel("P(Pair w/in k=10)")
stashfig("pair-knn-10")

