#%%
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from graspologic.match import GraphMatch
from src.data import DATA_PATH, DATA_VERSION, load_maggot_graph
from src.io import savefig
from src.visualization import adjplot, set_theme
from graspologic.utils import (
    remove_loops,
    pass_to_ranks,
    augment_diagonal,
    to_laplacian,
)

set_theme()
t0 = time.time()

#%%
mg = load_maggot_graph()
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]]
mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
mg.nodes["_inds"] = range(len(mg.nodes))
nodes = mg.nodes
adj = mg.sum.adj
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
n_pairs = len(rp_inds)

left_inds = lp_inds
right_inds = rp_inds


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
    adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":  # haven't really used much. a few params to look at here
        adjs = [to_laplacian(a) for a in adjs]
    return adjs


from graspologic.embed import AdjacencySpectralEmbed, selectSVD

n_initial_components = 16
n_final_components = 16


def embed_subgraph(adj):
    ase = AdjacencySpectralEmbed(
        n_components=n_initial_components, check_lcc=False, diag_aug=True, concat=False
    )
    return ase.fit_transform(adj)


def svd(X, n_components=n_final_components):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


embed_edge_types = "all"
if embed_edge_types == "all":
    edge_types = ["ad", "aa", "dd", "da"]
elif embed_edge_types == "sum":
    edge_types = ["sum"]

edge_type_adjs = []
stage1_left_embeddings = []
stage1_right_embeddings = []
for et in edge_types:
    edge_type_adj = mg.to_edge_type_graph(et).adj
    edge_type_adj = preprocess_for_embed(edge_type_adj)[0]
    ll_adj, rr_adj, lr_adj, rl_adj = split_adj(edge_type_adj)
    # LL, RR, LR, RL
    left_ipsi_out, left_ipsi_in = embed_subgraph(ll_adj)
    right_ipsi_out, right_ipsi_in = embed_subgraph(rr_adj)
    left_contra_out, right_contra_in = embed_subgraph(lr_adj)
    right_contra_out, left_contra_in = embed_subgraph(rl_adj)
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

from src.visualization import add_connections
import pandas as pd
from src.visualization import CLASS_COLOR_DICT as palette


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
plot_pairs(
    stage2_embedding,
    labels=nodes.iloc[joint_inds]["merge_class"].values,
    left_pair_inds=left_pair_inds,
    right_pair_inds=right_pair_inds,
)
plot_pairs(
    stage2_embedding[:, 8:16],
    labels=nodes.iloc[joint_inds]["merge_class"].values,
    left_pair_inds=left_pair_inds,
    right_pair_inds=right_pair_inds,
)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
