#%%

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from giskard.utils import get_paired_inds
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import select_svd
from graspologic.utils import augment_diagonal, pass_to_ranks, to_laplacian
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as normalize_matrix
from src.data import load_maggot_graph, load_palette
from src.io import savefig
from src.visualization import plot_pairs, set_theme

set_theme()
t0 = time.time()

out_path = Path("./maggot_models/experiments/connectivity_pair_rank")


def stashfig(name, **kws):
    savefig(
        name,
        pathname=out_path / "figs",
        **kws,
    )


#%%

CLASS_KEY = "simple_group"
palette = load_palette()
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
mg.to_largest_connected_component(verbose=True)
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


def ase(adj, n_components=None, normalize=False):
    U, S, Vt = select_svd(adj, n_components=n_components, algorithm="full")
    S_sqrt = np.diag(np.sqrt(S))
    X = U @ S_sqrt
    Y = Vt.T @ S_sqrt
    if normalize:
        X = normalize_matrix(X)
        Y = normalize_matrix(Y)
    return X, Y


# TODO could fight about the exact details of preprocessing here
adj = preprocess_for_embed(raw_adj)[0]
adjs = split_adj(adj)
ll_adj, rr_adj, lr_adj, rl_adj = adjs
ll_adj, rr_adj = prescale_for_embed([ll_adj, rr_adj])
lr_adj, rl_adj = prescale_for_embed([lr_adj, rl_adj])

#%%
normalize = False
n_components = 32  # 24 looked fine
X_ll, Y_ll = ase(ll_adj, n_components=n_components, normalize=normalize)
X_rr, Y_rr = ase(rr_adj, n_components=n_components, normalize=normalize)
X_lr, Y_lr = ase(lr_adj, n_components=n_components, normalize=normalize)
X_rl, Y_rl = ase(rl_adj, n_components=n_components, normalize=normalize)

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
n_final_components = 16
joint_latent, _ = ase(composite_latent, n_components=n_final_components)

# %%

left_X = composite_latent[left_paired_inds]
right_X = composite_latent[right_paired_inds]


#%%

# get nearest right neighbor for everyone on the left
def rank_neighbors(source_X, target_X, metric="euclidean"):
    n_target = len(target_X)
    n_source = len(source_X)
    nn = NearestNeighbors(radius=0, n_neighbors=n_target, metric=metric)
    nn.fit(target_X)
    neigh_dist, neigh_inds = nn.kneighbors(source_X)
    source_rank_neighbors = np.empty((n_source, n_target), dtype=int)
    for i in range(n_source):
        source_rank_neighbors[i, neigh_inds[i]] = np.arange(1, n_target + 1, dtype=int)
    return source_rank_neighbors


metric = "cosine"
left_neighbors = rank_neighbors(left_X, right_X, metric=metric)
right_neighbors = rank_neighbors(right_X, left_X, metric=metric)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
left_ranks = np.diag(left_neighbors)
right_ranks = np.diag(right_neighbors)
ranks = np.concatenate((left_ranks, right_ranks))
sns.histplot(
    ranks,
    ax=ax,
    cumulative=False,
    stat="probability",
    element="bars",
    fill=True,
    discrete=True,
)
ax.set(xlim=(0.5, 5.5), ylim=(0, 1))
ax.set(ylabel="Fraction neuron pairs", xlabel="Rank")
