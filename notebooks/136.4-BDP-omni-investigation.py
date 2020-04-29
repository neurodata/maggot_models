# %% [markdown]
# ##
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm
from umap import UMAP

from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    OmnibusEmbed,
    select_dimension,
    selectSVD,
)
from graspy.plot import pairplot
from graspy.simulations import sbm
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    symmetrize,
    to_laplace,
)
from src.align import Procrustes
from src.cluster import MaggotCluster, get_paired_inds
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    screeplot,
    set_axes_equal,
    stacked_barplot,
)

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


graph_type = "G"


def plot_pairs(
    X, labels, model=None, left_pair_inds=None, right_pair_inds=None, equal=False
):

    n_dims = X.shape[1]

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=CLASS_COLOR_DICT,
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


def preprocess_adjs(adjs, method="ase"):
    adjs = [pass_to_ranks(a) for a in adjs]
    adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":
        adjs = [to_laplace(a) for a in adjs]
    return adjs


def omni(
    adjs,
    n_components=4,
    remove_first=None,
    concat_graphs=True,
    concat_directed=True,
    method="ase",
):
    adjs = preprocess_adjs(adjs, method=method)
    omni = OmnibusEmbed(n_components=n_components, check_lcc=False, n_iter=10)
    embed = omni.fit_transform(adjs)
    if concat_directed:
        embed = np.concatenate(
            embed, axis=-1
        )  # this is for left/right latent positions
    if remove_first is not None:
        embed = embed[remove_first:]
    if concat_graphs:
        embed = np.concatenate(embed, axis=0)
    return embed


def ipsi_omni(adj, lp_inds, rp_inds, co_adj=None, n_components=4, method="ase"):
    ll_adj = adj[np.ix_(lp_inds, lp_inds)]
    rr_adj = adj[np.ix_(rp_inds, rp_inds)]
    ipsi_adjs = [ll_adj, rr_adj]
    if co_adj is not None:
        co_ll_adj = co_adj[np.ix_(lp_inds, lp_inds)]
        co_rr_adj = co_adj[np.ix_(rp_inds, rp_inds)]
        ipsi_adjs += [co_ll_adj, co_rr_adj]

    out_ipsi, in_ipsi = omni(
        ipsi_adjs,
        n_components=n_components,
        concat_directed=False,
        concat_graphs=False,
        method=method,
    )
    left_embed = np.concatenate((out_ipsi[0], in_ipsi[0]), axis=1)
    right_embed = np.concatenate((out_ipsi[1], in_ipsi[1]), axis=1)
    ipsi_embed = np.concatenate((left_embed, right_embed), axis=0)
    return ipsi_embed


def contra_omni(adj, lp_inds, rp_inds, co_adj=None, n_components=4, method="ase"):
    lr_adj = adj[np.ix_(lp_inds, rp_inds)]
    rl_adj = adj[np.ix_(rp_inds, lp_inds)]
    contra_adjs = [lr_adj, rl_adj]

    if co_adj is not None:
        co_lr_adj = co_adj[np.ix_(lp_inds, rp_inds)]
        co_rl_adj = co_adj[np.ix_(rp_inds, lp_inds)]
        contra_adjs += [co_lr_adj, co_rl_adj]

    out_contra, in_contra = omni(
        contra_adjs,
        n_components=n_components,
        concat_directed=False,
        concat_graphs=False,
        method=method,
    )

    left_embed = np.concatenate((out_contra[0], in_contra[1]), axis=1)
    right_embed = np.concatenate((out_contra[1], in_contra[0]), axis=1)
    contra_embed = np.concatenate((left_embed, right_embed), axis=0)
    return contra_embed


def lateral_omni(adj, lp_inds, rp_inds, n_components=4, method="ase"):
    ipsi_embed = ipsi_omni(
        adj, lp_inds, rp_inds, n_components=n_components, method=method
    )
    contra_embed = contra_omni(
        adj, lp_inds, rp_inds, n_components=n_components, method=method
    )

    embed = np.concatenate((ipsi_embed, contra_embed), axis=1)
    return embed


def multi_lateral_omni(adjs, lp_inds, rp_inds, n_components=4):
    ipsi_adjs = []
    for a in adjs:
        ll_adj = a[np.ix_(lp_inds, lp_inds)]
        rr_adj = a[np.ix_(rp_inds, rp_inds)]
        ipsi_adjs.append(ll_adj)
        ipsi_adjs.append(rr_adj)

    ipsi_embed = omni(ipsi_adjs, concat_graphs=False, n_components=n_components)
    left = []
    right = []
    for i, e in enumerate(ipsi_embed):
        if i % 2 == 0:
            left.append(e)
        else:
            right.append(e)
    left = np.concatenate(left, axis=1)
    right = np.concatenate(right, axis=1)
    ipsi_embed = np.concatenate((left, right), axis=0)

    contra_adjs = []
    for a in adjs:
        lr_adj = a[np.ix_(lp_inds, rp_inds)]
        rl_adj = a[np.ix_(rp_inds, lp_inds)]
        contra_adjs.append(lr_adj)
        contra_adjs.append(rl_adj)

    contra_embed = omni(contra_adjs, concat_graphs=False, n_components=n_components)
    left = []
    right = []
    for i, e in enumerate(contra_embed):
        if i % 2 == 0:
            left.append(e)
        else:
            right.append(e)
    left = np.concatenate(left, axis=1)
    right = np.concatenate(right, axis=1)
    contra_embed = np.concatenate((left, right), axis=0)

    embed = np.concatenate((ipsi_embed, contra_embed), axis=1)
    return embed


def reg_lateral_omni(adj, base_adj, lp_inds, rp_inds, n_components=4):
    base_ll_adj = base_adj[np.ix_(lp_inds, lp_inds)]
    base_rr_adj = base_adj[np.ix_(rp_inds, rp_inds)]
    ll_adj = adj[np.ix_(lp_inds, lp_inds)]
    rr_adj = adj[np.ix_(rp_inds, rp_inds)]

    ipsi_adjs = [base_ll_adj, base_rr_adj, ll_adj, rr_adj]
    ipsi_embed = omni(ipsi_adjs, remove_first=2, n_components=n_components)

    base_lr_adj = base_adj[np.ix_(lp_inds, rp_inds)]
    base_rl_adj = base_adj[np.ix_(rp_inds, lp_inds)]
    lr_adj = adj[np.ix_(lp_inds, rp_inds)]
    rl_adj = adj[np.ix_(rp_inds, lp_inds)]

    contra_adjs = [base_lr_adj, base_rl_adj, lr_adj, rl_adj]
    contra_embed = omni(contra_adjs, remove_first=2, n_components=n_components)

    embed = np.concatenate((ipsi_embed, contra_embed), axis=1)
    return embed


def quick_embed_viewer(
    embed, labels=None, lp_inds=None, rp_inds=None, left_right_indexing=False
):
    if left_right_indexing:
        lp_inds = np.arange(len(embed) // 2)
        rp_inds = np.arange(len(embed) // 2) + len(embed) // 2

    fig, axs = plt.subplots(3, 2, figsize=(20, 30))

    cmds = ClassicalMDS(n_components=2)
    cmds_euc = cmds.fit_transform(embed)
    plot_df = pd.DataFrame(data=cmds_euc)
    plot_df["labels"] = labels
    plot_kws = dict(
        x=0,
        y=1,
        hue="labels",
        palette=CLASS_COLOR_DICT,
        legend=False,
        s=20,
        linewidth=0.5,
        alpha=0.7,
    )
    ax = axs[0, 0]
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )
    ax.set_title("CMDS o euclidean")

    cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")
    pdist = symmetrize(pairwise_distances(embed, metric="cosine"))
    cmds_cos = cmds.fit_transform(pdist)
    plot_df[0] = cmds_cos[:, 0]
    plot_df[1] = cmds_cos[:, 1]
    ax = axs[0, 1]
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )
    ax.set_title("CMDS o cosine")

    tsne = TSNE(metric="euclidean")
    tsne_euc = tsne.fit_transform(embed)
    plot_df[0] = tsne_euc[:, 0]
    plot_df[1] = tsne_euc[:, 1]
    ax = axs[1, 0]
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )
    ax.set_title("TSNE o euclidean")

    tsne = TSNE(metric="precomputed")
    tsne_cos = tsne.fit_transform(pdist)
    plot_df[0] = tsne_cos[:, 0]
    plot_df[1] = tsne_cos[:, 1]
    ax = axs[1, 1]
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )
    ax.set_title("TSNE o cosine")

    umap = UMAP(metric="euclidean", n_neighbors=30, min_dist=1)
    umap_euc = umap.fit_transform(embed)
    plot_df[0] = umap_euc[:, 0]
    plot_df[1] = umap_euc[:, 1]
    ax = axs[2, 0]
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )
    ax.set_title("UMAP o euclidean")

    umap = UMAP(metric="cosine", n_neighbors=30, min_dist=1)
    umap_cos = umap.fit_transform(embed)
    plot_df[0] = umap_cos[:, 0]
    plot_df[1] = umap_cos[:, 1]
    ax = axs[2, 1]
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )
    ax.set_title("UMAP o cosine")


def umapper(embed, metric="euclidean", n_neighbors=30, min_dist=1, **kws):
    umap = UMAP(metric=metric, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_euc = umap.fit_transform(embed)
    plot_df = pd.DataFrame(data=umap_euc)
    plot_df["labels"] = labels
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_kws = dict(
        x=0,
        y=1,
        hue="labels",
        palette=CLASS_COLOR_DICT,
        legend=False,
        s=20,
        linewidth=0.5,
        alpha=0.7,
    )
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    left_right_indexing = True
    if left_right_indexing:
        tlp_inds = np.arange(len(embed) // 2)
        trp_inds = np.arange(len(embed) // 2) + len(embed) // 2
        add_connections(
            plot_df.iloc[tlp_inds, 0],
            plot_df.iloc[trp_inds, 0],
            plot_df.iloc[tlp_inds, 1],
            plot_df.iloc[trp_inds, 1],
            ax=ax,
        )
    return fig, ax


# %% [markdown]
# ## Load and preprocess data
VERSION = "2020-04-23"
graph_type = "G"
master_mg = load_metagraph(graph_type, version="2020-04-23")
mg = preprocess(
    master_mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

degrees = mg.calculate_degrees()
quant_val = np.quantile(degrees["Total edgesum"], 0.05)

# remove low degree neurons
idx = meta[degrees["Total edgesum"] > quant_val].index
print(quant_val)
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

idx = mg.meta[mg.meta["Pair"].isin(mg.meta.index)].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)

meta = mg.meta
meta["pair_td"] = meta["Pair ID"].map(meta.groupby("Pair ID")["Total degree"].mean())
mg = mg.sort_values(["pair_td", "Pair ID"], ascending=False)
meta["inds"] = range(len(meta))
adj = mg.adj.copy()
lp_inds, rp_inds = get_paired_inds(meta)
left_inds = meta[meta["left"]]["inds"]

print(len(mg))

# %% [markdown]
# ## Plot the ipsilateral connectomes
if meta["pair_td"].max() > 0:
    meta["pair_td"] = -meta["pair_td"]
ll_adj = adj[np.ix_(lp_inds, lp_inds)]
rr_adj = adj[np.ix_(rp_inds, rp_inds)]
left_meta = meta.iloc[lp_inds]
right_meta = meta.iloc[rp_inds]

plot_kws = dict(
    plot_type="scattermap",
    sort_class="merge_class",
    item_order=["pair_td", "Pair ID"],
    colors="merge_class",
    palette=CLASS_COLOR_DICT,
    ticks=False,
    class_order="pair_td",
    sizes=(1, 1),
    gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
)

plot_adjs = False
if plot_adjs:
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    _, _, top, _ = adjplot(ll_adj, ax=axs[0], meta=left_meta, **plot_kws)
    top.set_title(r"L $\to$ L")
    _, _, top, _ = adjplot(rr_adj, ax=axs[1], meta=right_meta, **plot_kws)
    top.set_title(r"R $\to$ R")
    plt.tight_layout()
    stashfig("ipsilateral-adj")

    lr_adj = adj[np.ix_(lp_inds, rp_inds)]
    rl_adj = adj[np.ix_(rp_inds, lp_inds)]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    _, _, top, _ = adjplot(lr_adj, ax=axs[0], meta=left_meta, **plot_kws)
    top.set_title(r"L $\to$ R")
    _, _, top, _ = adjplot(rl_adj, ax=axs[1], meta=right_meta, **plot_kws)
    top.set_title(r"R $\to$ L")
    plt.tight_layout()
stashfig("contralateral-adj")

# %% [markdown]
# ## Load the 4-color graphs

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g, version=VERSION)
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

# %% [markdown]
# ## simple demo of "in" vs "out" latent positions


# blocks 0, 1 differ only in their inputs, not their outputs
B = np.array(
    [
        [0.1, 0.1, 0.2, 0.05],
        [0.1, 0.1, 0.2, 0.05],
        [0.35, 0.15, 0.1, 0.1],
        [0.1, 0.05, 0.3, 0.4],
    ]
)
sns.heatmap(B, square=True, annot=True)
sbm_sample, sbm_labels = sbm([100, 100, 100, 100], B, directed=True, return_labels=True)
ase = AdjacencySpectralEmbed()
out_embed, in_embed = ase.fit_transform(sbm_sample)
pairplot(out_embed, sbm_labels)  # don't see separation between [0, 1]
pairplot(in_embed, sbm_labels)  # do see separation between [0, 1]
# from this we can conclude that the "right" embedding or right singular vectors are the
# ones corresponding to input
# (out, in)

# %% [markdown]
# ## Options for the embedding
# - ASE and procrustes (not shown here)
# - Bilateral OMNI on G, SVD
# - Bilateral OMNI on each of the 4-colors, concatenated, SVD
# - Bilateral OMNI on each of the 4-colors, with regularization, concatenated, SVD
# - Bilateral OMNI jointly with all 4-colors

n_omni_components = 8  # this is used for all of the embedings initially
n_svd_components = 16  # this is for the last step


def svd(X, n_components=n_svd_components):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


# %% [markdown]
# ## only contra
# just_contra_embed = omni(
#     [full_adjs[0], full_adjs[2]],
#     n_components=n_omni_components,
#     remove_first=None,
#     concat_graphs=True,
#     concat_directed=True,
#     method="ase",
# )
# svd_contra_embed = svd(just_contra_embed)


# %% [markdown]
# # Omni of contra/ipsi together

full_adjs = [
    adj[np.ix_(lp_inds, lp_inds)],
    adj[np.ix_(lp_inds, rp_inds)],
    adj[np.ix_(rp_inds, rp_inds)],
    adj[np.ix_(rp_inds, lp_inds)],
]
out_embed, in_embed = omni(
    full_adjs,
    n_components=n_omni_components,
    remove_first=None,
    concat_graphs=False,
    concat_directed=False,
    method="ase",
)

# ipsi out, contra out, ipsi in, contra in
left_embed = np.concatenate(
    (out_embed[0], out_embed[1], in_embed[0], in_embed[3]), axis=1
)
right_embed = np.concatenate(
    (out_embed[2], out_embed[3], in_embed[2], in_embed[1]), axis=1
)
omni_naive_embed = np.concatenate((left_embed, right_embed), axis=0)

ase_naive_embed = svd(omni_naive_embed)

# ##
# out_embed, in_embed = omni(
#     full_adjs,
#     n_components=n_omni_components,
#     remove_first=None,
#     concat_graphs=False,
#     concat_directed=False,
#     method="lse",
# )
# # ipsi out, contra out, ipsi in, contra in
# left_embed = np.concatenate(
#     (out_embed[0], out_embed[1], in_embed[0], in_embed[3]), axis=1
# )
# right_embed = np.concatenate(
#     (out_embed[2], out_embed[3], in_embed[2], in_embed[1]), axis=1
# )
# omni_naive_embed = np.concatenate((left_embed, right_embed), axis=0)

# lse_naive_embed = svd(omni_naive_embed)

# %% [markdown]
# ## Bilateral OMNI on G, SVD

omni_flat_embed = lateral_omni(
    adj, lp_inds, rp_inds, n_components=n_omni_components, method="ase"
)
ase_flat_embed = svd(omni_flat_embed)


# %% [markdown]
# ## just compare


# %% [markdown]
# ## Bilateral OMNI on each of the 4-colors, concatenated, SVD

omni_multi_embed = []
for a in adjs:
    omni_multi_embed.append(
        lateral_omni(a, lp_inds, rp_inds, n_components=n_omni_components)
    )
omni_multi_embed = np.concatenate(omni_multi_embed, axis=1)
ase_multi_embed = svd(omni_multi_embed)


# %% [markdown]
# ## Bilateral OMNI on each of the 4-colors, with regularization, concatenated, SVD

omni_reg_embed = []
for a in adjs:
    omni_reg_embed.append(
        reg_lateral_omni(a, adj, lp_inds, rp_inds, n_components=n_omni_components)
    )
omni_reg_embed = np.concatenate(omni_reg_embed, axis=1)
ase_reg_embed = svd(omni_reg_embed)

# %% [markdown]
# ## Bilateral OMNI on all 4-colors

adjs_and_sum = adjs + [adj]
omni_joint_embed = multi_lateral_omni(
    adjs_and_sum, lp_inds, rp_inds, n_components=n_omni_components
)
ase_joint_embed = svd(omni_joint_embed)

# %% [markdown]
# ## Compute neighbors at K
new_lp_inds = np.arange(len(mg) // 2)
new_rp_inds = np.arange(len(mg) // 2) + len(mg) // 2


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
    neighbors_at_k = pd.Series(data=neighbors_at_k, index=np.arange(1, k_max + 1))
    neighbors_at_k.name = "p_at_k"
    return neighbors_at_k


# names = ["flat", "multi", "joint", "reg", "naive"]
# embeds = [
#     ase_flat_embed,
#     ase_multi_embed,
#     ase_joint_embed,
#     ase_reg_embed,
#     ase_naive_embed,
# ]
names = ["iso", "aniso", "multi"]
embeds = [ase_naive_embed, ase_flat_embed, ase_multi_embed]
dims = np.arange(1, 16)
dfs = []
for d in dims:
    for name, embed in zip(names, embeds):
        p_at_k = compute_neighbors_at_k(embed[:, :d], new_lp_inds, new_rp_inds)
        neighbor_df = p_at_k.to_frame()
        neighbor_df.reset_index(inplace=True)
        neighbor_df.rename(columns={"index": "K"}, inplace=True)
        neighbor_df["method"] = name
        neighbor_df["d"] = d
        dfs.append(neighbor_df)
neighbor_df = pd.concat(dfs, ignore_index=True)

# %% [markdown]
# ## Plot nearest neighbor results
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
k = 5
sns.lineplot(
    data=neighbor_df[neighbor_df["K"] == k],
    x="d",
    y="p_at_k",
    hue="method",
    style="method",
    # style_order=["reg", "joint", "multi", "flat"],
)
ax.set_ylabel(f"P @ K = {k}")
ax.set_xlabel("# dimensions")
stashfig(f"p_at_k={k}_embed-iso-aniso-multi")

# %% [markdown]
# ## Look at the best one! (ish)

new_meta = meta.iloc[np.concatenate((lp_inds, rp_inds), axis=0)].copy()
labels = new_meta["merge_class"].values
plot_pairs(
    ase_flat_embed[:, :8],
    labels,
    left_pair_inds=new_lp_inds,
    right_pair_inds=new_rp_inds,
)
stashfig("ase-flat-pairs")

quick_embed_viewer(
    ase_flat_embed[:, :8], labels=labels, lp_inds=new_lp_inds, rp_inds=new_rp_inds
)
stashfig("ase-flat-manifold")


# %% [markdown]
# ## Now, try to do a similar quantification but for classes
# KC
# MBON
# MBIN
# ORN
# UPN
# some of the antennal lobe stuff


def class_neighbors_at_k(X, labels, target, k_max=10, metric="euclidean"):
    nn = NearestNeighbors(radius=0, n_neighbors=k_max + 1, metric=metric)
    nn.fit(X)
    neigh_dist, neigh_inds = nn.kneighbors(X)

    neigh_inds = neigh_inds[:, 1:]  # remove self as neighbor

    mask = labels == target
    target_inds = np.arange(len(X))[mask]
    target_neigh_inds = neigh_inds[mask]

    p_nearby = []
    neighbors_in_target = np.isin(target_neigh_inds, target_inds)
    for k in np.arange(1, k_max + 1):
        p_nearby_at_k = neighbors_in_target[:, :k].sum() / (k * len(target_inds))
        p_nearby.append(p_nearby_at_k)
    p_nearby = np.array(p_nearby)
    neighbor_df = pd.DataFrame(data=p_nearby, index=np.arange(1, k_max + 1))
    neighbor_df.index.name = "K"
    neighbor_df.rename(columns={0: target}, inplace=True)
    return neighbor_df


new_meta = meta.iloc[np.concatenate((lp_inds, rp_inds), axis=0)].copy()
labels = new_meta["merge_class"].values

k_max = 10

embed_df = []
for name, embed in zip(names, embeds):
    neighbor_df = []
    for d in np.arange(1, 16):
        X = embed[:, :d]

        class1 = new_meta["class1"].values

        neighbors = []
        for target in ["uPN", "sens-ORN"]:
            neighbors.append(class_neighbors_at_k(X, labels, target))

        for target in ["KC", "mPN", "MBON", "MBIN"]:
            neighbors.append(class_neighbors_at_k(X, class1, target))

        neighbors = pd.concat(neighbors, ignore_index=False, axis=1)
        neighbors = neighbors.reset_index()
        neighbors = neighbors.melt(value_name="p_at_k", var_name="class", id_vars=["K"])
        neighbors["d"] = d
        neighbor_df.append(neighbors)
    neighbor_df = pd.concat(neighbor_df, axis=0)
    neighbor_df["method"] = name
    embed_df.append(neighbor_df)

embed_df = pd.concat(embed_df, axis=0)

# k = 5
# temp_df = embed_df[embed_df["K"] == k]
# fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
# axs = axs.ravel()
# for i, name in enumerate(names):
#     ax = axs[i]
#     plot_df = temp_df[temp_df["method"] == name]
#     sns.lineplot(data=plot_df, x="d", y="p_at_k", hue="class", ax=ax)
#     ax.set_title(name)
#     ax.get_legend().remove()
# plt.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

# hard to compare directly on the above

# %% [markdown]
# ##
# fix d
# one plot for each class
# line for each of the embeddings

k = 5

plot_df = embed_df[embed_df["K"] == k]
# plot_df = plot_df[plot_df["d"] == d]

classes = ["uPN", "sens-ORN", "KC", "mPN", "MBON", "MBIN"]
fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
axs = axs.ravel()

for i, cell_class in enumerate(classes):
    ax = axs[i]
    temp_df = plot_df[plot_df["class"] == cell_class]
    sns.lineplot(
        data=temp_df,
        x="d",
        y="p_at_k",
        hue="method",
        ax=ax,
        style="method",
        # style_order=["reg", "joint", "multi", "flat"],
    )
    ax.set_title(cell_class)

axs[0].set_ylabel(f"Prop. @ K = {k}")
axs[3].set_ylabel(f"Prop. @ K = {k}")

plt.tight_layout()

stashfig(f"embed-class-knn-k={k}")


# %%
# # Notes
# I like aniso better than iso
# not sure about reg or not
# for sides, we have {iso, aniso}
# for method, we have {lse, ase}
# for color, we have {flat, multi (separate), joint (omni), reg (multi but with G)}
# there seems to be no single embedding that is winning at everything.

n_levels = 10
metric = "bic"
bic_ratio = 1
d = 10
basename = f"aniso-omni-bic_ratio={bic_ratio}-d={d}"

mc = MaggotCluster(
    "0",
    adj=adj,
    n_init=25,
    meta=new_meta,
    stashfig=stashfig,
    min_clusters=1,
    max_clusters=3,
    X=ase_flat_embed[:, :d],
    bic_ratio=bic_ratio,
    reembed=False,
    min_split=4,
)


for i in range(n_levels):
    for j, node in enumerate(mc.get_lowest_level()):
        node.fit_candidates(show_plot=False)
    for j, node in enumerate(mc.get_lowest_level()):
        node.select_model(k=None, metric=metric)
    mc.collect_labels()

n_levels = mc.height

fig, axs = plt.subplots(1, n_levels, figsize=(10 * n_levels, 40))
for i in range(n_levels):
    ax = axs[i]
    stacked_barplot(
        mc.meta[f"lvl{i}_labels_side"],
        mc.meta["merge_class"],
        category_order=np.unique(mc.meta[f"lvl{i}_labels_side"].values),
        color_dict=CLASS_COLOR_DICT,
        norm_bar_width=False,
        ax=ax,
    )
    ax.set_yticks([])
    ax.get_legend().remove()

plt.tight_layout()

stashfig(f"count-barplot-lvl{i}" + basename)
plt.close()

fig, axs = plt.subplots(1, n_levels, figsize=(10 * n_levels, 40))
for i in range(n_levels):
    ax = axs[i]
    stacked_barplot(
        mc.meta[f"lvl{i}_labels_side"],
        mc.meta["merge_class"],
        category_order=np.unique(mc.meta[f"lvl{i}_labels_side"].values),
        color_dict=CLASS_COLOR_DICT,
        norm_bar_width=True,
        ax=ax,
    )
    ax.set_yticks([])
    ax.get_legend().remove()

plt.tight_layout()

stashfig(f"prop-barplot-lvl{i}" + basename)
plt.close()


inds = np.concatenate((lp_inds, rp_inds))
new_adj = adj[np.ix_(inds, inds)]
new_meta = mc.meta
new_meta["sf"] = -signal_flow(new_adj)

for l in range(n_levels):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    sort_class = [f"lvl{i}_labels" for i in range(l)]
    sort_class += [f"lvl{l}_labels_side"]
    adjplot(
        new_adj,
        meta=new_meta,
        sort_class=sort_class,
        item_order="merge_class",
        plot_type="scattermap",
        class_order="sf",
        sizes=(0.5, 1),
        ticks=False,
        colors="merge_class",
        ax=ax,
        palette=CLASS_COLOR_DICT,
        gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
    )
    stashfig(f"adj-lvl{l}" + basename)
    plt.close()


pairs = np.unique(new_meta["Pair ID"])
p_same_clusters = []
for l in range(n_levels):
    n_same = 0
    for p in pairs:
        if new_meta[new_meta["Pair ID"] == p][f"lvl{l}_labels"].nunique() == 1:
            n_same += 1
    p_same = n_same / len(pairs)
    print(p_same)
    p_same_clusters.append(p_same)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(x=range(n_levels), y=p_same_clusters, ax=ax)
sns.scatterplot(x=range(n_levels), y=p_same_clusters, ax=ax)
ax.set_ylabel("P same cluster")
ax.set_xlabel("Level")
stashfig("p_in_same_cluster" + basename)

n_clusters = []
for l in range(n_levels):
    n_clusters.append(new_meta[f"lvl{l}_labels"].nunique())

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(x=range(n_levels), y=n_clusters, ax=ax)
sns.scatterplot(x=range(n_levels), y=n_clusters, ax=ax)
ax.set_ylabel("Clusters per side")
ax.set_xlabel("Level")
stashfig("n_cluster" + basename)

size_dfs = []
for l in range(n_levels):
    sizes = new_meta.groupby(f"lvl{l}_labels_side").size().values
    sizes = pd.DataFrame(data=sizes, columns=["Size"])
    sizes["Level"] = l
    size_dfs.append(sizes)

size_df = pd.concat(size_dfs)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=size_df, x="Level", y="Size", ax=ax, jitter=0.45, alpha=0.5)
ax.set_yscale("log")
stashfig("log-sizes" + basename)

# %% [markdown]
# ## try a dcsbm

from graspy.models import DCSBMEstimator, SBMEstimator
from scipy.stats import poisson
from graspy.utils import remove_loops

X = ase_flat_embed[:, :d]
n_pairs = len(X) // 2
new_lp_inds = np.arange(n_pairs)
new_rp_inds = np.arange(n_pairs).copy() + n_pairs

rows = []
for l in range(n_levels):
    labels = new_meta[f"lvl{l}_labels"].values
    left_adj = binarize(new_adj[np.ix_(new_lp_inds, new_lp_inds)])
    left_adj = remove_loops(left_adj)
    right_adj = binarize(new_adj[np.ix_(new_rp_inds, new_rp_inds)])
    right_adj = remove_loops(right_adj)

    dcsbm = DCSBMEstimator(directed=True, loops=False)
    uni_labels, inv = np.unique(labels, return_inverse=True)
    dcsbm.fit(left_adj, inv[new_lp_inds])
    train_left_p = dcsbm.p_mat_
    train_left_p[train_left_p == 0] = 1 / train_left_p.size

    score = poisson.logpmf(left_adj, train_left_p).sum()
    rows.append(
        dict(train_side="left", test="same", test_side="left", score=score, level=l)
    )
    score = poisson.logpmf(right_adj, train_left_p).sum()
    rows.append(
        dict(
            train_side="left", test="opposite", test_side="right", score=score, level=l
        )
    )

    dcsbm = DCSBMEstimator(directed=True, loops=False)
    dcsbm.fit(right_adj, inv[new_rp_inds])
    train_right_p = dcsbm.p_mat_
    train_right_p[train_right_p == 0] = 1 / train_right_p.size

    score = poisson.logpmf(left_adj, train_right_p).sum()
    rows.append(
        dict(
            train_side="right", test="opposite", test_side="left", score=score, level=l
        )
    )
    score = poisson.logpmf(right_adj, train_right_p).sum()
    rows.append(
        dict(train_side="right", test="same", test_side="right", score=score, level=l)
    )
    # right_train_lik_sums.append(right_train_lik.sum())


# %% [markdown]
# ##
plot_df = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(data=plot_df, hue="test", x="level", y="score", style="train_side")
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
stashfig("dcsbm-lik-curves")
# sns.lineplot(x=range(n_levels), y=left_train_lik_sums)
# sns.lineplot(x=range(n_levels), y=right_train_lik_sums)




# %%
from src.visualization import plot_neurons
from src.pymaid import start_instance

lvl = 4

uni_labels = np.unique(new_meta[f"lvl{lvl}_labels"])
start_instance()

for label in uni_labels:
    plot_neurons(new_meta, f"lvl{lvl}_labels", label=label, barplot=True)
    stashfig(f"label{label}_lvl{lvl}" + basename)

# %% [markdown]
# ## next 
# could plot "stop points" in the embedded space
#