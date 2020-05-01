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
from scipy.stats import poisson
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
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import pairplot
from graspy.simulations import sbm
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    symmetrize,
    to_laplace,
)
from src.align import Procrustes
from src.cluster import BinaryCluster, MaggotCluster, get_paired_inds
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.pymaid import start_instance

from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    plot_neurons,
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


def preprocess_adjs(adjs, method="ase", internal_edge=10):
    internal_inds = np.where(adjs[0] == 1000)
    adjs = [pass_to_ranks(a) for a in adjs]
    for a in adjs:
        a[internal_inds] = internal_edge
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
    internal_edge=10,
):
    adjs = preprocess_adjs(adjs, method=method, internal_edge=internal_edge)
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
plot_adjs = False
if plot_adjs:
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
# ## Combine them into the 2N graph...
n_verts = len(adjs[0])
axon_inds = np.arange(n_verts)
dend_inds = axon_inds.copy() + n_verts
double_adj = np.empty((2 * n_verts, 2 * n_verts))
double_adj[np.ix_(axon_inds, axon_inds)] = adjs[1]  # Gaa
double_adj[np.ix_(axon_inds, dend_inds)] = adjs[0]  # Gad
double_adj[np.ix_(dend_inds, axon_inds)] = adjs[3]  # Gda
double_adj[np.ix_(dend_inds, dend_inds)] = adjs[2]  # Gdd
double_adj[axon_inds, dend_inds] = 1000  # make internal edges, make em big
double_adj[dend_inds, axon_inds] = 1000

axon_meta = mg.meta.rename(index=lambda x: str(x) + "_axon")
axon_meta["compartment"] = "axon"
dend_meta = mg.meta.rename(index=lambda x: str(x) + "_dend")
dend_meta["compartment"] = "dend"


double_meta = pd.concat((axon_meta, dend_meta), axis=0)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(
    double_adj,
    plot_type="scattermap",
    sizes=(1, 1),
    ax=ax,
    meta=double_meta,
    sort_class=["hemisphere", "compartment"],
    item_order=["merge_class", "Pair ID"],
    colors=["merge_class"],
    palette=CLASS_COLOR_DICT,
)
stashfig("double-adj")

# %% [markdown]
# ##
from src.graph import MetaGraph

double_mg = MetaGraph(double_adj, double_meta)
double_mg.sort_values(["left", "compartment", "Pair ID"])
n_pairs = double_mg["Pair ID"].nunique()
(
    double_mg.meta.iloc[: 2 * n_pairs]["Pair ID"].values
    == double_mg.meta.iloc[2 * n_pairs :]["Pair ID"].values
).all()

left_inds = np.arange(2 * n_pairs)
right_inds = np.arange(2 * n_pairs) + 2 * n_pairs
adj = double_mg.adj

ll_adj = adj[np.ix_(left_inds, left_inds)]
rr_adj = adj[np.ix_(right_inds, right_inds)]
lr_adj = adj[np.ix_(left_inds, right_inds)]
rl_adj = adj[np.ix_(right_inds, left_inds)]

all_adjs = [ll_adj, lr_adj, rr_adj, rl_adj]

n_omni_components = 8  # this is used for all of the embedings initially
n_svd_components = 16  # this is for the last step
method = "lse"


def svd(X, n_components=n_svd_components):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


# %% [markdown]
# ##


def _get_omni_matrix(graphs):
    """
    Helper function for creating the omnibus matrix.
    Parameters
    ----------
    graphs : list
        List of array-like with shapes (n_vertices, n_vertices).
    Returns
    -------
    out : 2d-array
        Array of shape (n_vertices * n_graphs, n_vertices * n_graphs)
    """
    shape = graphs[0].shape
    n = shape[0]  # number of vertices
    m = len(graphs)  # number of graphs

    A = np.array(graphs, copy=False, ndmin=3)

    # Do some numpy broadcasting magic.
    # We do sum in 4d arrays and reduce to 2d array.
    # Super fast and efficient
    out = (A[:, :, None, :] + A.transpose(1, 0, 2)[None, :, :, :]).reshape(n * m, -1)

    # Averaging
    out /= 2

    return out


lp_inds = left_inds[:n_pairs]
rp_inds = right_inds[:n_pairs]
new_lp_inds = np.arange(n_pairs)
new_rp_inds = np.arange(n_pairs) + n_pairs
new_meta = double_mg.meta.iloc[np.concatenate((lp_inds, rp_inds))].copy()


def preprocess_adjs(adjs, method="ase", internal_edge=10):
    internal_inds = np.where(adjs[0] == 1000)
    adjs = [pass_to_ranks(a) for a in adjs]
    for a in adjs:
        a[internal_inds] = internal_edge
    # adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":
        adjs = [to_laplace(a) for a in adjs]
    return adjs


pre_adjs = preprocess_adjs(all_adjs, internal_edge=1)
omni_mat = _get_omni_matrix(pre_adjs)

omni_metas = []
names = ["LL", "LR", "RR", "RL"]
for i in range(len(all_adjs)):
    om = new_meta.copy()
    om["graph"] = names[i]
    omni_metas.append(om)

omni_meta = pd.concat(omni_metas, ignore_index=True)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
adjplot(omni_mat, plot_type="scattermap", sizes=(1, 1), ax=ax)
stashfig("omni-mat")
# %% [markdown]
# ##

A = np.array([[1, 2], [3, 4]]).astype(float)
B = np.array([[5, 6], [7, 8]]).astype(float)
omni_mat = _get_omni_matrix([A, B])
adjplot(
    omni_mat,
    plot_type="scattermap",
    sizes=(100, 100),
    hue="weight",
    # palette=pal,
)


# %% [markdown]
# ##


for internal_edge in np.linspace(0, 1, 4):
    out_embed, in_embed = omni(
        all_adjs,
        n_components=n_omni_components,
        remove_first=None,
        concat_graphs=False,
        concat_directed=False,
        method=method,
        internal_edge=internal_edge,
    )

    # ipsi out, contra out, ipsi in, contra in
    left_embed = np.concatenate(
        (out_embed[0], out_embed[1], in_embed[0], in_embed[3]), axis=1
    )
    right_embed = np.concatenate(
        (out_embed[2], out_embed[3], in_embed[2], in_embed[1]), axis=1
    )

    left_dend_embed = left_embed[:n_pairs]
    left_axon_embed = left_embed[n_pairs:]
    left_embed = np.concatenate((left_dend_embed, left_axon_embed), axis=1)
    right_dend_embed = right_embed[:n_pairs]
    right_axon_embed = right_embed[n_pairs:]
    right_embed = np.concatenate((right_dend_embed, right_axon_embed), axis=1)

    omni_iso_embed = np.concatenate((left_embed, right_embed), axis=0)

    svd_iso_embed = svd(omni_iso_embed)

    labels = new_meta["merge_class"].values
    plot_pairs(
        svd_iso_embed[:, :8],
        labels,
        left_pair_inds=new_lp_inds,
        right_pair_inds=new_rp_inds,
    )
    stashfig(f"double-embed-flat-pairs-ie={internal_edge}-method={method}")
    plt.close()

    quick_embed_viewer(
        svd_iso_embed[:, :16], labels=labels, lp_inds=new_lp_inds, rp_inds=new_rp_inds
    )
    stashfig(f"double-embed-flat-manifold-ie={internal_edge}-method={method}")
    plt.close()

# # %% [markdown]
# # ##
# sf = signal_flow(pass_to_ranks(double_adj))
# double_meta["sf"] = -sf


# fig, ax = plt.subplots(1, 1, figsize=(20, 20))
# adjplot(
#     double_adj,
#     plot_type="scattermap",
#     sizes=(1, 1),
#     ax=ax,
#     meta=double_meta,
#     item_order=["sf"],  # ,"merge_class", "Pair ID"],
#     colors=["merge_class"],
#     palette=CLASS_COLOR_DICT,
# )
# stashfig("double-adj-sf")

# # %% [markdown]
# # ##

# embedder = AdjacencySpectralEmbed(n_components=None, diag_aug=True)
# embed = embedder.fit_transform(pass_to_ranks(double_adj))
# embed = np.concatenate(embed, axis=-1)
# top_embed = embed[axon_inds]
# bottom_embed = embed[dend_inds]
# embed = np.concatenate((top_embed, bottom_embed), axis=-1)
# U, _, _ = selectSVD(embed)

# pairplot(U, labels=mg.meta["merge_class"].values, palette=CLASS_COLOR_DICT)
# stashfig("ase-double-pairs")


# %%
