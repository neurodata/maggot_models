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
from graspologic.align import OrthogonalProcrustes
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
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import add_connections, adjplot, set_theme

set_theme()
t0 = time.time()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/embed/figs",
        **kws,
    )


#%%
mg = load_maggot_graph()
mg = mg[mg.nodes["has_predicted_matching"]]

# mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
# mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
# og_nodelist = mg.nodes.index
# og_nodes = mg.nodes.copy()
# mg.to_largest_connected_component(verbose=True)
# new_nodelist = mg.nodes.index
# diff = np.setdiff1d(og_nodelist, new_nodelist)
# og_nodes.loc[diff, ["accessory_neurons", "paper_clustered_neurons", 'merge_class']]


#%%
nodes = mg.nodes

from giskard.utils import get_paired_inds

lp_inds, rp_inds = get_paired_inds(
    nodes, pair_key="predicted_pair", pair_id_key="predicted_pair_id"
)

#%%
nodes["_inds"] = range(len(nodes))
left_nodes = nodes[nodes["hemisphere"] == "L"].copy()
left_paired_nodes = left_nodes[left_nodes["predicted_pair_id"] != -1]
left_unpaired_nodes = left_nodes[left_nodes["predicted_pair_id"] == -1]
right_nodes = nodes[nodes["hemisphere"] == "R"].copy()
right_paired_nodes = right_nodes[right_nodes["predicted_pair_id"] != -1]
right_unpaired_nodes = right_nodes[right_nodes["predicted_pair_id"] == -1]

# HACK this only works because all the duplicate nodes are on the right
lp_inds = left_paired_nodes.loc[right_paired_nodes["predicted_pair"]]["_inds"]
rp_inds = right_paired_nodes["_inds"]
print("Pairs all valid: ")
print((nodes.iloc[lp_inds].index == nodes.iloc[rp_inds]["predicted_pair"]).all())
left_inds = lp_inds
right_inds = rp_inds

n_pairs = len(rp_inds)
adj = mg.sum.adj


def split_adj(adj):
    ll_adj = adj[np.ix_(left_inds, left_inds)]
    rr_adj = adj[np.ix_(right_inds, right_inds)]
    lr_adj = adj[np.ix_(left_inds, right_inds)]
    rl_adj = adj[np.ix_(right_inds, left_inds)]
    return ll_adj, rr_adj, lr_adj, rl_adj


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
    if not isinstance(adjs, list):
        adjs = [adjs]
    adjs = [pass_to_ranks(a) for a in adjs]
    # adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":  # haven't really used much. a few params to look at here
        adjs = [to_laplacian(a, form="R-DAD") for a in adjs]
    return tuple(adjs)


n_initial_components = 64  # TODO maybe this should be even bigger? used to have at 16
n_final_components = 64


def svd(X, n_components=n_final_components):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


embed_edge_types = "sum"
if embed_edge_types == "all":
    edge_types = ["ad", "aa", "dd", "da"]
elif embed_edge_types == "sum":
    edge_types = ["sum"]


def project_graph(U, V, R, scaled=True):
    # This is basically saying
    # \hat{P} = U Z S W^T V^T
    # in the scaled case, then
    # \hat{P} = XY^T
    # where
    # X = UZS^{1/2}
    # Y = VWS^{1/2}
    Z, S, W = selectSVD(R, n_components=len(R), algorithm="full")
    S_sqrt = np.diag(np.sqrt(S))
    X = U @ Z
    Y = V @ W
    if scaled:
        X = X @ S_sqrt
        Y = Y @ S_sqrt

    return X, Y


def prescale_for_embed(adjs):
    norms = [np.linalg.norm(adj, ord="fro") for adj in adjs]
    mean_norm = np.mean(norms)
    adjs = [adjs[i] * mean_norm / norms[i] for i in range(len(adjs))]
    return adjs


def decompose_scores(scores, scaled=True, align=True):
    R1 = scores[0]
    R2 = scores[1]
    Z1, S1, W1 = selectSVD(R1, n_components=len(R1), algorithm="full")
    Z2, S2, W2 = selectSVD(R2, n_components=len(R2), algorithm="full")
    if scaled:
        S1_sqrt = np.diag(np.sqrt(S1))
        S2_sqrt = np.diag(np.sqrt(S2))
        Z1 = Z1 @ S1_sqrt
        Z2 = Z2 @ S2_sqrt
        W1 = W1 @ S1_sqrt
        W2 = W2 @ S2_sqrt
    if align:
        op = OrthogonalProcrustes()
        n = len(Z1)
        U1 = np.concatenate((Z1, W1), axis=0)
        U2 = np.concatenate((Z2, W2), axis=0)
        U1_mapped = op.fit_transform(U1, U2)
        Z1 = U1_mapped[:n]
        W1 = U1_mapped[n:]
    return Z1, W1, Z2, W2


def project_to_node_space(mase, scaled=True, align=True):
    U = mase.latent_left_
    V = mase.latent_right_
    Z1, W1, Z2, W2 = decompose_scores(mase.scores_, scaled=scaled, align=align)
    X1 = U @ Z1
    Y1 = V @ W1
    X2 = U @ Z2
    Y2 = V @ W2
    return X1, Y1, X2, Y2


def align(X1, X2, Y1, Y2):
    # Solve argmin_Q \|X1Q - X2\|_F^2 + \|Y1Q - Y2\|_F^2
    # This is the same as \|U1Q - U2\|_F^2 where U1 = [X1^T Y1^T]^T
    n = len(X1)
    U1 = np.concatenate((X1, Y1), axis=0)
    U2 = np.concatenate((X2, Y2), axis=0)
    op = OrthogonalProcrustes()
    U1_mapped = op.fit_transform(U1, U2)
    X1_mapped = U1_mapped[:n]
    Y1_mapped = U1_mapped[n:]
    return (X1_mapped, Y1_mapped)


# TODO when do we want to scale
project_scaled = True  # whether to scale in the final projection step
mase_scaled = True  # whether to scale during the initial embedding in MASE
prescaled = True  # whether to scale the subgraphs to have same norm prior to embedding
prealign = True
edge_type_adjs = []
stage1_left_embeddings = []
stage1_right_embeddings = []
for et in edge_types:
    edge_type_adj = mg.to_edge_type_graph(et).adj

    # split into subgraphs (left to left, right to right, left to right, right to left)
    split_adjs = split_adj(edge_type_adj)
    # preprocess by adding a small constant, pass to ranks, augmenting the diagonals
    (ll_adj, rr_adj, lr_adj, rl_adj) = preprocess_for_embed(list(split_adjs))
    # potentially scale the two matrices to have the same (mean) Frobenius norm
    adjs_to_embed = [ll_adj, rr_adj]
    if prescaled:
        adjs_to_embed = prescale_for_embed(adjs_to_embed)
    # Run MASE between corresponding subgraphs
    ipsi_mase = MultipleASE(n_components=n_initial_components, scaled=mase_scaled)
    ipsi_mase.fit(adjs_to_embed)

    left_ipsi_out, left_ipsi_in, right_ipsi_out, right_ipsi_in = project_to_node_space(
        ipsi_mase, scaled=project_scaled, align=prealign
    )
    if not prealign:
        left_ipsi_out, left_ipsi_in = align(
            left_ipsi_out, right_ipsi_out, left_ipsi_in, left_ipsi_in
        )

    adjs_to_embed = [lr_adj, rl_adj]
    if prescaled:
        adjs_to_embed = prescale_for_embed(adjs_to_embed)
    contra_mase = MultipleASE(n_components=n_initial_components, scaled=mase_scaled)
    contra_mase.fit(adjs_to_embed)

    (
        left_contra_out,
        right_contra_in,
        right_contra_out,
        left_contra_in,
    ) = project_to_node_space(contra_mase, scaled=project_scaled, align=prealign)

    if not prealign:
        left_contra_out, right_contra_in = align(
            left_contra_out, right_contra_out, right_contra_in, left_contra_in
        )

    stage1_left_embeddings += [
        left_ipsi_out,
        left_ipsi_in,
        left_contra_out,
        left_contra_in,
    ]
    stage1_right_embeddings += [
        right_ipsi_out,
        right_ipsi_in,
        right_contra_out,
        right_contra_in,
    ]

stage1_left_embeddings = np.concatenate(stage1_left_embeddings, axis=1)
stage1_right_embeddings = np.concatenate(stage1_right_embeddings, axis=1)
stage1_embeddings = np.concatenate(
    (stage1_left_embeddings, stage1_right_embeddings), axis=0
)
stage2_embedding = svd(stage1_embeddings)

#%%
fig, axs = plt.subplots(
    2, 2, figsize=(10, 5), gridspec_kw=dict(height_ratios=[0.9, 0.05])
)

cbar_ax = merge_axes(fig, axs, rows=1)
vmax = np.max(ipsi_mase.scores_)
vmin = np.min(ipsi_mase.scores_)
heatmap_kws = dict(
    square=True,
    vmin=vmin,
    vmax=vmax,
    center=0,
    cmap="RdBu_r",
    xticklabels=False,
    yticklabels=False,
)
ax = axs[0, 0]
sns.heatmap(ipsi_mase.scores_[0], ax=ax, cbar=False, **heatmap_kws)
ax.set(title=r"$\hat{R}_{LL}$")
ax = axs[0, 1]
sns.heatmap(
    ipsi_mase.scores_[1],
    ax=ax,
    cbar_ax=cbar_ax,
    cbar_kws={"orientation": "horizontal", "shrink": 0.6},
    **heatmap_kws,
)
ax.set(title=r"$\hat{R}_{RR}$")
stashfig("ipsi-R-matrices")

#%%

Z1, W1, Z2, W2 = decompose_scores(ipsi_mase.scores_, scaled=project_scaled, align=False)
(
    Z1_mapped,
    W1_mapped,
    _,
    _,
) = decompose_scores(ipsi_mase.scores_, scaled=project_scaled, align=True)

n = len(Z1)
Z = np.concatenate((Z1, Z1_mapped, Z2), axis=0)
labels = n * [r"$Z_{LL}$"] + n * [r"$Z_{LL}Q$"] + n * [r"$Z_{RR}$"]
plot_data = pd.DataFrame(data=Z, columns=[str(i) for i in range(Z.shape[1])])
plot_data["labels"] = labels
colors = sns.color_palette("tab20")
r_palette = {r"$Z_{LL}$": colors[7], r"$Z_{LL}Q$": colors[6], r"$Z_{RR}$": colors[0]}
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.scatterplot(data=plot_data, x="1", y="3", ax=ax, hue=labels, palette=r_palette)

#%%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
heatmap_kws = dict(
    vmax=7,
    vmin=-7,
    center=0,
    cmap="RdBu_r",
    square=True,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
)
sns.heatmap(Z1, ax=axs[0], **heatmap_kws)
sns.heatmap(Z1_mapped, ax=axs[1], **heatmap_kws)
sns.heatmap(Z2, ax=axs[2], **heatmap_kws)


#%%


def plot_pairs(
    X,
    labels,
    n_show=8,
    model=None,
    left_pair_inds=None,
    right_pair_inds=None,
    equal=False,
):
    """Plots pairwise dimensional projections, and draws lines between known pair neurons

    Parameters
    ----------
    X : [type]
        [description]
    labels : [type]
        [description]
    model : [type], optional
        [description], by default None
    left_pair_inds : [type], optional
        [description], by default None
    right_pair_inds : [type], optional
        [description], by default None
    equal : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if n_show is not None:
        n_dims = n_show
    else:
        n_dims = X.shape[1]

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X[:, :n_dims], columns=[str(i) for i in range(n_dims)])
    data["label"] = labels

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=str(j),
                    y=str(i),
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=palette,
                )
                if left_pair_inds is not None and right_pair_inds is not None:
                    add_connections(
                        data.iloc[left_pair_inds, j],
                        data.iloc[right_pair_inds, j],
                        data.iloc[left_pair_inds, i],
                        data.iloc[right_pair_inds, i],
                        ax=ax,
                    )

    plt.tight_layout()
    return fig, axs


joint_inds = np.concatenate((left_inds, right_inds), axis=0)
left_pair_inds = np.arange(n_pairs)
right_pair_inds = left_pair_inds.copy() + n_pairs
reindexed_nodes = nodes.iloc[joint_inds]
plot_pairs(
    stage2_embedding,
    labels=reindexed_nodes["merge_class"].values,
    left_pair_inds=left_pair_inds,
    right_pair_inds=right_pair_inds,
)
stashfig("mase_final_embedding")

#%%
from giskard.utils import get_paired_inds
from sklearn.neighbors import NearestNeighbors


def compute_nn_ranks(left_X, right_X, max_n_neighbors=None, metric="cosine"):
    if max_n_neighbors is None:
        max_n_neighbors = len(left_X)

    nn_kwargs = dict(n_neighbors=max_n_neighbors, metric=metric)
    nn_left = NearestNeighbors(**nn_kwargs)
    nn_right = NearestNeighbors(**nn_kwargs)
    nn_left.fit(left_X)
    nn_right.fit(right_X)

    left_neighbors = nn_right.kneighbors(left_X, return_distance=False)
    right_neighbors = nn_left.kneighbors(right_X, return_distance=False)

    arange = np.arange(len(left_X))
    _, left_match_rank = np.where(left_neighbors == arange[:, None])
    _, right_match_rank = np.where(right_neighbors == arange[:, None])
    left_match_rank += 1
    right_match_rank += 1

    rank_data = np.concatenate((left_match_rank, right_match_rank))
    rank_data = pd.Series(rank_data, name="pair_nn_rank")
    rank_data = rank_data.to_frame()
    rank_data["metric"] = metric
    rank_data["side"] = len(left_X) * ["Left"] + len(right_X) * ["Right"]
    rank_data["n_components"] = left_X.shape[1]
    return rank_data


lp_inds, rp_inds = get_paired_inds(reindexed_nodes)

frames = []
for n_components in [16, 32, 48, 64]:
    left_X = stage2_embedding[lp_inds, :n_components]
    right_X = stage2_embedding[rp_inds, :n_components]
    rank_data = compute_nn_ranks(left_X, right_X, metric="cosine")
    frames.append(rank_data)
results = pd.concat(frames, ignore_index=True)
results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.ecdfplot(
    data=results,
    x="pair_nn_rank",
    hue="n_components",
    ax=ax,
)
vals = [1, 5, 10, 15, 20]
ax.set(
    xlim=(0, 20),
    ylim=(0.4, 1),
    xticks=vals,
    xlabel="K (# of nearest neighbors)",
    ylabel="Recall @ K",
)
# for val in vals:
#     r_at_k = (results["pair_nn_rank"] <= val).mean()
#     ax.text(
#         val,
#         r_at_k + 0.005,
#         f"{r_at_k:.2f}",
#         ha="center",
#         va="bottom",
#     )
ax.axhline(1, linewidth=3, linestyle=":", color="lightgrey")
stashfig("pair-nn-recall")

#%%
out_path = Path("maggot_models/experiments/embed/outs")

stage1_embedding_df = pd.DataFrame(index=reindexed_nodes.index, data=stage1_embeddings)
stage1_embedding_df.sort_index(inplace=True)
stage1_embedding_df.to_csv(out_path / "stage1_embedding.csv")

stage2_embedding = pd.DataFrame(index=reindexed_nodes.index, data=stage2_embedding)
stage2_embedding.sort_index(inplace=True)
stage2_embedding.to_csv(out_path / "stage2_embedding.csv")

has_embedding = pd.Series(
    data=np.ones(len(reindexed_nodes.index), dtype=bool),
    index=reindexed_nodes.index,
    name="has_embedding",
)
join_node_meta(has_embedding, overwrite=True, fillna=False)

# #%%
# plot_pairs(
#     np.concatenate((left_ipsi_out, right_ipsi_out), axis=0),
#     labels=reindexed_nodes["merge_class"].values,
#     left_pair_inds=left_pair_inds,
#     right_pair_inds=right_pair_inds,
# )
# stashfig("ipsi-out-mase-projection-w0")
# #%%
# op = OrthogonalProcrustes()
# left_ipsi_out_mapped = op.fit_transform(left_ipsi_out, right_ipsi_out)
# plot_pairs(
#     np.concatenate((left_ipsi_out_mapped, right_ipsi_out), axis=0),
#     labels=reindexed_nodes["merge_class"].values,
#     left_pair_inds=left_pair_inds,
#     right_pair_inds=right_pair_inds,
# )
# stashfig("ipsi-out-mase-projection-w-rotation")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
