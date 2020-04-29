# %% [markdown]
# ## The goal of this notebook:
# investigate regularization approaches, for now, just on the full graph
# these include
#     - truncate high degree
#     - truncate low degree
#     - plus c
#     - levina paper on row normalization
#     - others?

# %% [markdown]
# ##
import os
import time
import warnings
from itertools import chain

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter, NodeMixin
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.utils.testing import ignore_warnings
from tqdm.autonotebook import tqdm

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import (
    AdjacencySpectralEmbed,
    ClassicalMDS,
    LaplacianSpectralEmbed,
    select_dimension,
    selectSVD,
)
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import rdpg
from graspy.utils import augment_diagonal, binarize, pass_to_ranks
from src.align import Procrustes
from src.cluster import get_paired_inds
from src.data import load_metagraph
from src.graph import preprocess
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.traverse import (
    Cascade,
    RandomWalk,
    TraverseDispatcher,
    to_markov_matrix,
    to_path_graph,
    to_transmission_matrix,
)
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

from graspy.embed import OmnibusEmbed
from umap import UMAP

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
    savecsv(df, name)


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


def lateral_omni(adj, lp_inds, rp_inds, n_components=4):
    left_left_adj = pass_to_ranks(adj[np.ix_(lp_inds, lp_inds)])
    right_right_adj = pass_to_ranks(adj[np.ix_(rp_inds, rp_inds)])
    omni = OmnibusEmbed(
        n_components=n_components, n_elbows=2, check_lcc=False, n_iter=10
    )
    ipsi_embed = omni.fit_transform([left_left_adj, right_right_adj])
    ipsi_embed = np.concatenate(ipsi_embed, axis=-1)
    ipsi_embed = np.concatenate(ipsi_embed, axis=0)

    left_right_adj = pass_to_ranks(adj[np.ix_(lp_inds, rp_inds)])
    right_left_adj = pass_to_ranks(adj[np.ix_(rp_inds, lp_inds)])
    omni = OmnibusEmbed(
        n_components=n_components, n_elbows=2, check_lcc=False, n_iter=10
    )
    contra_embed = omni.fit_transform([left_right_adj, right_left_adj])
    contra_embed = np.concatenate(contra_embed, axis=-1)
    contra_embed = np.concatenate(contra_embed, axis=0)

    embed = np.concatenate((ipsi_embed, contra_embed), axis=1)
    return embed


# %% [markdown]
# ##
graph_type = "G"
master_mg = load_metagraph(graph_type, version="2020-04-01")
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
# ##

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g, version="2020-04-01")
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

# embed_adjs = [pass_to_ranks(a) for a in adjs]

# %% [markdown]
# ## just omni on the 4 colors for the right right subgraph

right_embed_adjs = [pass_to_ranks(a[np.ix_(rp_inds, rp_inds)]) for a in adjs]
omni = OmnibusEmbed(check_lcc=False)
embeds = omni.fit_transform(right_embed_adjs)
embeds = np.concatenate(embeds, axis=-1)
embeds = np.concatenate(embeds, axis=-1)
print(embeds.shape)

U, S, V = selectSVD(embeds, n_components=8)


labels = meta["merge_class"].values[rp_inds]

plot_pairs(U, labels)
stashfig(f"simple-omni-right-reduced-4-color")

# %% [markdown]
# ## Look at what each edge color type looks like when regularized by g
# only the right right subgraph
right_full_adj = pass_to_ranks(adj[np.ix_(rp_inds, rp_inds)])
labels = meta["merge_class"].values[rp_inds]
all_reg_embeds = []
for a in right_embed_adjs:
    omni = OmnibusEmbed(check_lcc=False)
    embeds = omni.fit_transform([right_full_adj, a])
    embeds = np.concatenate(embeds, axis=-1)
    embeds = embeds[1]
    all_reg_embeds.append(embeds)
    # plot_pairs(embeds, labels)
all_reg_embeds = np.concatenate(all_reg_embeds, axis=1)
U, S, V = selectSVD(all_reg_embeds, n_components=8)
plot_pairs(U, labels)
stashfig(f"omni-regularized-right-colors")

# %% [markdown]
# ## embed all of the right right subgraphs for each color separately
all_ase_embeds = []
for a in right_embed_adjs:
    ase = AdjacencySpectralEmbed(check_lcc=False)
    embeds = ase.fit_transform(a)
    embeds = np.concatenate(embeds, axis=-1)
    # embeds = embeds[1]
    all_ase_embeds.append(embeds)
    # plot_pairs(embeds, labels)
all_ase_embeds = np.concatenate(all_ase_embeds, axis=1)
U, S, V = selectSVD(all_ase_embeds, n_components=8)
plot_pairs(U, labels)
stashfig(f"ase-right-colors")


# %% [markdown]
# ## do lateral omni on each separately, then concatenates

color_embeds = []
for a in adjs:
    embed = lateral_omni(pass_to_ranks(a), lp_inds, rp_inds)
    color_embeds.append(embed)

color_embeds = np.concatenate(color_embeds, axis=1)
U, S, V = selectSVD(color_embeds, n_components=6)

labels = np.concatenate(
    (meta["merge_class"].values[lp_inds], meta["merge_class"].values[rp_inds])
)

plot_pairs(
    U,
    labels,
    left_pair_inds=np.arange(len(lp_inds)),
    right_pair_inds=np.arange(len(lp_inds)) + len(lp_inds),
)

# %% [markdown]
# ## Try bilateral, regularized-color omni


def reg_omni(adjs):
    adjs = [a + 1 / (len(lp_inds) ** 2) for a in adjs]
    adjs = [augment_diagonal(a) for a in adjs]
    omni = OmnibusEmbed(n_components=4, check_lcc=False, n_iter=10)
    embed = omni.fit_transform(adjs)
    embed = np.concatenate(embed, axis=-1)
    embed = embed[2:]  # TODO
    embed = np.concatenate(embed, axis=0)
    return embed


def reg_lateral_omni(adj, base_adj, lp_inds, rp_inds):
    base_ll_adj = pass_to_ranks(base_adj[np.ix_(lp_inds, lp_inds)])
    base_rr_adj = pass_to_ranks(base_adj[np.ix_(rp_inds, rp_inds)])
    ll_adj = pass_to_ranks(adj[np.ix_(lp_inds, lp_inds)])
    rr_adj = pass_to_ranks(adj[np.ix_(rp_inds, rp_inds)])

    ipsi_adjs = [base_ll_adj, base_rr_adj, ll_adj, rr_adj]
    ipsi_embed = reg_omni(ipsi_adjs)

    base_lr_adj = pass_to_ranks(base_adj[np.ix_(lp_inds, rp_inds)])
    base_rl_adj = pass_to_ranks(base_adj[np.ix_(rp_inds, lp_inds)])
    lr_adj = pass_to_ranks(adj[np.ix_(lp_inds, rp_inds)])
    rl_adj = pass_to_ranks(adj[np.ix_(rp_inds, lp_inds)])

    contra_adjs = [base_lr_adj, base_rl_adj, lr_adj, rl_adj]
    contra_embed = reg_omni(contra_adjs)

    embed = np.concatenate((ipsi_embed, contra_embed), axis=1)
    return embed


reg_color_embeds = []
for a in adjs:
    embed = reg_lateral_omni(a, adj, lp_inds, rp_inds)
    reg_color_embeds.append(embed)

reg_color_embeds = np.concatenate(reg_color_embeds, axis=1)


U, S, V = selectSVD(reg_color_embeds, n_components=16)
# %% [markdown]
# ##
from sklearn.decomposition import PCA

# U = PCA(n_components=8).fit_transform(reg_color_embeds)

labels = np.concatenate(
    (meta["merge_class"].values[lp_inds], meta["merge_class"].values[rp_inds])
)

plot_pairs(
    U,
    labels,
    left_pair_inds=np.arange(len(lp_inds)),
    right_pair_inds=np.arange(len(lp_inds)) + len(lp_inds),
)
stashfig("regularized-bilateral-omni")

# %% [markdown]
# ##

from graspy.utils import symmetrize


def quick_embed_viewer(
    embed, labels=labels, lp_inds=None, rp_inds=None, left_right_indexing=False
):
    if left_right_indexing:
        lp_inds = np.arange(len(embed) // 2)
        rp_inds = np.arange(len(embed) // 2) + len(embed) // 2

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
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )

    cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")
    pdist = symmetrize(pairwise_distances(embed, metric="cosine"))
    cmds_cos = cmds.fit_transform(pdist)
    plot_df[0] = cmds_cos[:, 0]
    plot_df[1] = cmds_cos[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )

    tsne = TSNE(metric="euclidean")
    tsne_euc = tsne.fit_transform(embed)
    plot_df[0] = tsne_euc[:, 0]
    plot_df[1] = tsne_euc[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )

    tsne = TSNE(metric="precomputed")
    tsne_cos = tsne.fit_transform(pdist)
    plot_df[0] = tsne_cos[:, 0]
    plot_df[1] = tsne_cos[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )

    umap = UMAP(metric="euclidean")
    umap_euc = umap.fit_transform(embed)
    plot_df[0] = umap_euc[:, 0]
    plot_df[1] = umap_euc[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )

    umap = UMAP(metric="cosine")
    umap_cos = umap.fit_transform(embed)
    plot_df[0] = umap_cos[:, 0]
    plot_df[1] = umap_cos[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(data=plot_df, ax=ax, **plot_kws)
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )


quick_embed_viewer(reg_color_embeds, labels, left_right_indexing=True)
quick_embed_viewer(U, labels, left_right_indexing=True)

# %% [markdown]
# ##


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


# %%

# monotone_embed = lateral_omni(adj, lp_inds, rp_inds)
# umapper(monotone_embed, min_dist=0.7)
umapper(color_embeds, min_dist=0.7)

# how can i quantify goodness here? we care about pairs and what things are being clustered together
# esp kc, uPN, MBON, sensory, antennal lobe stuff.
from sklearn.neighbors import NearestNeighbors

tlp_inds = np.arange(len(embed) // 2)
trp_inds = np.arange(len(embed) // 2) + len(embed) // 2


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


# %% [markdown]
# ##
embed = lateral_omni(adj, lp_inds, rp_inds, n_components=20)
dims = np.arange(1, 20)
neibs = []
for d in dims:
    neibs.append(compute_neighbors_at_k(embed[:, :d], tlp_inds, trp_inds))

neibs = np.array(neibs)

neibs_df = pd.DataFrame(data=neibs)
neibs_df["d"] = dims
neibs_df = neibs_df.melt(id_vars="d", value_name="P @ K", var_name="K")
neibs_df["K"] = neibs_df["K"] + 1

sns.lineplot(data=neibs_df, x="K", y="P @ K", hue="d")

# %% [markdown]
# ##
sns.lineplot(data=neibs_df[neibs_df["K"] == 10], x="d", y="P @ K")
sns.lineplot(data=neibs_df[neibs_df["K"] == 5], x="d", y="P @ K")
sns.lineplot(data=neibs_df[neibs_df["K"] == 1], x="d", y="P @ K")

# %% [markdown]
# ##
# embed = lateral_omni(adj, lp_inds, rp_inds, n_components=20)
dims = np.arange(1, 16)
neibs = []
for d in dims:
    neibs.append(compute_neighbors_at_k(U[:, :d], tlp_inds, trp_inds))

neibs = np.array(neibs)

color_neibs_df = pd.DataFrame(data=neibs)
color_neibs_df["d"] = dims
color_neibs_df = color_neibs_df.melt(id_vars="d", value_name="P @ K", var_name="K")
color_neibs_df["K"] = color_neibs_df["K"] + 1

# sns.lineplot(data=color_neibs_df, x="K", y="P @ K", hue="d")

# %% [markdown]
# ##
sns.lineplot(data=neibs_df[neibs_df["K"] == 1], x="d", y="P @ K")
sns.lineplot(data=color_neibs_df[color_neibs_df["K"] == 1], x="d", y="P @ K")
stashfig("p@k-prelim")

# %% [markdown]
# ##
d = 5
print(np.linalg.norm(U[lp_inds, :d] - U[rp_inds, :d]) / np.linalg.norm(U[:, d]))
print(
    np.linalg.norm(embed[lp_inds, :d] - embed[rp_inds, :d])
    / np.linalg.norm(embed[:, d])
)

# %% [markdown]
# ##

inds = np.concatenate((lp_inds.values, rp_inds.values))
pair_meta = meta.iloc[inds]
pair_adj = pass_to_ranks(adj[np.ix_(inds, inds)])

from src.cluster import MaggotCluster

np.random.seed(888)
n_levels = 8
metric = "bic"
mc = MaggotCluster(
    "bilateral-reg-c-omni-0",
    meta=pair_meta,
    adj=pair_adj,
    n_init=25,
    stashfig=stashfig,
    min_clusters=1,
    max_clusters=3,
    X=U,
)
basename = "bilateral-regulatized-color-omni"

for i in range(n_levels):
    for j, node in enumerate(mc.get_lowest_level()):
        node.fit_candidates(show_plot=False)
    for j, node in enumerate(mc.get_lowest_level()):
        node.select_model(2, metric=metric)
    mc.collect_labels()

fig, axs = plt.subplots(1, n_levels, figsize=(10 * n_levels, 30))
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

stashfig(f"count-barplot-lvl{i}" + basename)
plt.close()

fig, axs = plt.subplots(1, n_levels, figsize=(10 * n_levels, 30))
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

stashfig(f"prop-barplot-lvl{i}" + basename)
plt.close()

for i in range(n_levels):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    adjplot(
        mc.adj,
        meta=mc.meta,
        sort_class=f"lvl{i}_labels_side",
        item_order="merge_class",
        plot_type="scattermap",
        sizes=(0.5, 1),
        ticks=False,
        colors="merge_class",
        ax=ax,
        palette=CLASS_COLOR_DICT,
        gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
    )
    stashfig(f"adj-lvl{i}" + basename)

# %% [markdown]
# ##
uni, counts = np.unique(mc.meta["lvl6_labels"], return_counts=True)
max_ind = np.argmax(counts)
uni[max_ind]

# %% [markdown]
# ##
big_guy_meta = mc.meta[mc.meta["lvl6_labels"] == uni[max_ind]]

# %% [markdown]
# ##
sns.distplot(big_guy_meta["Total edgesum"])

# %% [markdown]
# ##
big_inds = big_guy_meta["inds"]
adjplot(
    pass_to_ranks(adj[np.ix_(big_inds, big_inds)]),
    plot_type="heatmap",
    meta=big_guy_meta,
    sort_class="merge_class",
    item_order="Total edgesum",
)

# %% [markdown]
# ##
plot_pairs(U[big_inds, :] * 1000, labels=big_guy_meta["merge_class"].values)

# %% [markdown]
# ## conclusions
# looked like the low degree nodes were getting "trapped" in a small cluster, numerically
# adjusted maggot cluster code to rescale when things get too small

# %% [markdown]
# ## redo the regularization investigations, but with omni

