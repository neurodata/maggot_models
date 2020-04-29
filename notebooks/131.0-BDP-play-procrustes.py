# %% [markdown]
# # THE MIND OF A MAGGOT

# %% [markdown]
# ## Imports
import os
import warnings

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import orthogonal_procrustes
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
    adjplot,
    barplot_text,
    draw_networkx_nice,
    gridmap,
    matrixplot,
    palplot,
    screeplot,
    set_axes_equal,
    stacked_barplot,
    remove_spines,
    add_connections,
)

import n_sphere

# from tqdm import tqdm


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


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    """
    p = np.asarray(p)
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


graph_type = "G"
mg = load_metagraph(graph_type, version="2020-04-01")
mg = preprocess(
    mg,
    threshold=0,
    sym_threshold=False,
    remove_pdiff=True,
    binarize=False,
    weight="weight",
)
meta = mg.meta

# plot where we are cutting out nodes based on degree
degrees = mg.calculate_degrees()
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
sns.distplot(np.log10(degrees["Total edgesum"]), ax=ax)
q = np.quantile(degrees["Total edgesum"], 0.05)
ax.axvline(np.log10(q), linestyle="--", color="r")
ax.set_xlabel("log10(total synapses)")

# remove low degree neurons
idx = meta[degrees["Total edgesum"] > q].index
mg = mg.reindex(idx, use_ids=True)

# remove center neurons # FIXME
idx = mg.meta[mg.meta["hemisphere"].isin(["L", "R"])].index
mg = mg.reindex(idx, use_ids=True)

mg = mg.make_lcc()
mg.calculate_degrees(inplace=True)
meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj


# %% [markdown]
# ## Embed for a dissimilarity measure


class Procrustes:
    def __init__(self, method="ortho"):
        self.method = method

    def fit(self, X, Y=None, x_seeds=None, y_seeds=None):
        if Y is None and (x_seeds is not None and y_seeds is not None):
            Y = X[y_seeds]
            X = X[x_seeds]
        elif Y is not None and (x_seeds is not None or y_seeds is not None):
            ValueError("May only use one of \{Y, \{x_seeds, y_seeds\}\}")

        X = X.copy()
        Y = Y.copy()

        if self.method == "ortho":
            R = orthogonal_procrustes(X, Y)[0]
        elif self.method == "diag-ortho":
            norm_X = np.linalg.norm(X, axis=1)
            norm_Y = np.linalg.norm(Y, axis=1)
            norm_X[norm_X <= 1e-15] = 1
            norm_Y[norm_Y <= 1e-15] = 1
            X = X / norm_X[:, None]
            Y = Y / norm_Y[:, None]
            R = orthogonal_procrustes(X, Y)[0]
        else:
            raise ValueError("Invalid `method` parameter")

        self.R_ = R
        return self

    def transform(self, X, map_inds=None):
        if map_inds is not None:
            X_transform = X.copy()
            X_transform[map_inds] = X_transform[map_inds] @ self.R_
        else:
            X_transform = X @ self.R_
        return X_transform


lp_inds, rp_inds = get_paired_inds(meta)
left_inds = meta[meta["left"]]["inds"]
right_inds = meta[meta["right"]]["inds"]


def remove_axis(ax):
    remove_spines(ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])


method = "ortho"

print("Embedding graph...")
embedder = AdjacencySpectralEmbed(n_components=None, n_elbows=2)
in_embed, out_embed = embedder.fit_transform(pass_to_ranks(adj))
procrust = Procrustes(method=method)
# procrust.fit(in_embed, x_seeds=lp_inds, y_seeds=rp_inds)
embed = np.concatenate((in_embed, out_embed), axis=-1)


dim1 = 0
dim2 = 4

fig, axs = plt.subplots(2, 2, figsize=(20, 20))
plot_df = pd.DataFrame(data=embed[:, [0, 1]])
plot_df["merge_class"] = meta["merge_class"].values
ax = axs[0, 0]
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
remove_axis(ax)
ax.set_ylabel("Cartesian")
ax.spines["right"].set_visible(True)
ax.set_title("Before Procrustes")
add_connections(
    plot_df.iloc[lp_inds, 0],
    plot_df.iloc[rp_inds, 0],
    plot_df.iloc[lp_inds, 1],
    plot_df.iloc[rp_inds, 1],
    ax=ax,
)

##
ax = axs[1, 0]

norm_embed = n_sphere.convert_spherical(embed)
norm_embed = norm_embed[:, 1:]  # chop off R dimension
plot_df[0] = norm_embed[:, 0]
plot_df[1] = norm_embed[:, 1]

sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
add_connections(
    plot_df.iloc[lp_inds, 0],
    plot_df.iloc[rp_inds, 0],
    plot_df.iloc[lp_inds, 1],
    plot_df.iloc[rp_inds, 1],
    ax=ax,
)
remove_axis(ax)
ax.set_ylabel("Angles")
ax.spines["right"].set_visible(True)

##
ax = axs[0, 1]
procrust.fit(in_embed, x_seeds=lp_inds, y_seeds=rp_inds)
in_embed = procrust.transform(in_embed, map_inds=left_inds)

procrust.fit(out_embed, x_seeds=lp_inds, y_seeds=rp_inds)
out_embed = procrust.transform(out_embed, map_inds=left_inds)
embed = np.concatenate((in_embed, out_embed), axis=-1)

plot_df[0] = embed[:, 0]
plot_df[1] = embed[:, 1]

sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
ax.axis("off")
add_connections(
    plot_df.iloc[lp_inds, 0],
    plot_df.iloc[rp_inds, 0],
    plot_df.iloc[lp_inds, 1],
    plot_df.iloc[rp_inds, 1],
    ax=ax,
)
ax.set_title("After Procrustes")

##
ax = axs[1, 1]

norm_embed = n_sphere.convert_spherical(embed)
norm_embed = norm_embed[:, 1:]  # chop off R dimension
plot_df[0] = norm_embed[:, 0]
plot_df[1] = norm_embed[:, 1]

sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="merge_class",
    palette=CLASS_COLOR_DICT,
    legend=False,
    ax=ax,
    s=20,
    linewidth=0.5,
    alpha=0.7,
)
ax.axis("off")
add_connections(
    plot_df.iloc[lp_inds, 0],
    plot_df.iloc[rp_inds, 0],
    plot_df.iloc[lp_inds, 1],
    plot_df.iloc[rp_inds, 1],
    ax=ax,
)

plt.tight_layout()

fig.suptitle(f"Method = {method}", y=1)
stashfig(f"procrustes-ase-{method}")


# %% [markdown]
# ## Try ranking pairs

pg = pairplot(
    norm_embed,
    labels=meta["merge_class"].values,
    palette=CLASS_COLOR_DICT,
    diag_kind="hist",
)
pg._legend.remove()

# %% [markdown]
# ##


def embedplot(embed):
    plot_df = pd.DataFrame(data=embed[:, [0, 1]])
    plot_df["merge_class"] = meta["merge_class"].values
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(
        data=plot_df,
        x=0,
        y=1,
        hue="merge_class",
        palette=CLASS_COLOR_DICT,
        legend=False,
        ax=ax,
        s=20,
        linewidth=0.5,
        alpha=0.7,
    )
    ax.axis("off")
    add_connections(
        plot_df.iloc[lp_inds, 0],
        plot_df.iloc[rp_inds, 0],
        plot_df.iloc[lp_inds, 1],
        plot_df.iloc[rp_inds, 1],
        ax=ax,
    )


from sklearn.manifold import MDS, Isomap, TSNE
from graspy.embed import ClassicalMDS
from graspy.utils import symmetrize

euc_pdist = pairwise_distances(embed, metric="euclidean")
euc_pdist = symmetrize(euc_pdist)

cos_pdist = pairwise_distances(embed, metric="cosine")
cos_pdist = symmetrize(cos_pdist)

for Manifold, name in zip((ClassicalMDS,), ("cmds",)):  # MDS, Isomap, TSNE):
    print(name)
    embedder = Manifold(n_components=2, dissimilarity="precomputed")

    euc_embed = embedder.fit_transform(euc_pdist)
    embedplot(euc_embed)
    stashfig(f"euc-embed-{name}")

    cos_embed = embedder.fit_transform(cos_pdist)
    embedplot(cos_embed)
    stashfig(f"cos-embed-{name}")


for Manifold, name in zip((Isomap, TSNE), ("iso", "tsne")):
    print(name)
    embedder = Manifold(n_components=2, metric="precomputed")

    euc_embed = embedder.fit_transform(euc_pdist)
    embedplot(euc_embed)
    stashfig(f"euc-embed-{name}")

    cos_embed = embedder.fit_transform(cos_pdist)
    embedplot(cos_embed)
    stashfig(f"cos-embed-{name}")

# %%
