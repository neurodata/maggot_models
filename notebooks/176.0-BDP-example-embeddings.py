# %% [markdown]
# #
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from graspy.embed import AdjacencySpectralEmbed, selectSVD, OmnibusEmbed
from graspy.utils import augment_diagonal, to_laplace
from graspy.plot import pairplot
from graspy.utils import pass_to_ranks
from src.data import load_metagraph
from src.io import savecsv, savefig
from src.utils import get_paired_inds
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
)
from topologic.embedding import node2vec_embedding

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

sns.set_context("talk")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


graph_type = "G"

mg = load_metagraph(graph_type)
mg = mg.make_lcc()
meta = mg.meta
meta["inds"] = range(len(meta))
lp_inds, rp_inds = get_paired_inds(meta)
inds = np.concatenate((lp_inds, rp_inds))
ids = meta.index[inds]
mg = mg.reindex(ids, use_ids=True)
meta = mg.meta.sort_index()
ids = meta.index
mg = mg.reindex(ids, use_ids=True)
g = mg.g.copy()
meta = mg.meta
meta["inds"] = range(len(meta))
class_labels = meta["merge_class"].values
lp_inds, rp_inds = get_paired_inds(meta)

# %% [markdown]
# ## Embed using Node2vec


n2v_params = dict(
    num_walks=100,
    walk_length=8,
    dimensions=16,
    window_size=2,
    iterations=3,
    inout_hyperparameter=1,
    return_hyperparameter=1,
)
embed = node2vec_embedding(g, **n2v_params)
latent = embed[0]
node_labels = embed[1]
node_labels = np.vectorize(int)(node_labels)
sort_inds = np.argsort(node_labels)
n2v_latent = latent[sort_inds]

# %% [markdown]
# ## Embed using ASE
adj = mg.adj.copy()
ptr_adj = pass_to_ranks(adj)

ase = AdjacencySpectralEmbed(n_components=16)
ase_latent = ase.fit_transform(ptr_adj)
ase_latent = np.concatenate(ase_latent, axis=-1)

# %% [markdown]
# ## Embed using Omni

n_omni_components = 8  # this is used for all of the embedings initially
n_svd_components = 16  # this is for the last step
method = "ase"  # one could also do LSE


def preprocess_adjs(adjs, method="ase"):
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
    adjs = [pass_to_ranks(a) for a in adjs]
    adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":  # haven't really used much. a few params to look at here
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
    """Omni with a few extra (optional) bells and whistles for concatenation post embed

    Parameters
    ----------
    adjs : [type]
        [description]
    n_components : int, optional
        [description], by default 4
    remove_first : [type], optional
        [description], by default None
    concat_graphs : bool, optional
        [description], by default True
    concat_directed : bool, optional
        [description], by default True
    method : str, optional
        [description], by default "ase"

    Returns
    -------
    [type]
        [description]
    """
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


def svd(X, n_components=n_svd_components):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


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
    method=method,
)

# this step is weird to explain - for the omnibus embedding I've jointly embedded
# the left ipsilateral, left-to-right contralateral, right ipsilateral, and
# right-to-left contralateral subgraphs (so taken together, it's everything).
# ipsi out, contra out, ipsi in, contra in
left_embed = np.concatenate(
    (out_embed[0], out_embed[1], in_embed[0], in_embed[3]), axis=1
)
right_embed = np.concatenate(
    (out_embed[2], out_embed[3], in_embed[2], in_embed[1]), axis=1
)
omni_iso_embed = np.concatenate((left_embed, right_embed), axis=0)

svd_embed = svd(omni_iso_embed)

# %% [markdown]
# ##

from src.visualization import remove_spines

# import matplotlib as mpl

# # plotting settings
# rc_dict = {
#     "axes.spines.right": False,
#     "axes.spines.top": False,
#     "axes.formatter.limits": (-3, 3),
#     "figure.figsize": (6, 3),
#     "figure.dpi": 100,
#     # "axes.edgecolor": "lightgrey",
#     # "ytick.color": "grey",
#     # "xtick.color": "grey",
#     # "axes.labelcolor": "dimgrey",
#     # "text.color": "dimgrey",
# }
# for key, val in rc_dict.items():
#     mpl.rcParams[key] = val


def plot_pairs(
    X,
    labels,
    n_components=8,
    model=None,
    left_pair_inds=None,
    right_pair_inds=None,
    equal=False,
    plot_tsne=False,
    ax=None,
    s=12,
):
    """ Plots pairwise dimensional projections, and draws lines between known pair neurons

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

    if n_components is None:
        n_components = X.shape[1]

    data = pd.DataFrame(data=X.copy())
    data["label"] = labels

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    tsne = TSNE(metric="euclidean")
    tsne_euc = tsne.fit_transform(X)
    data[0] = tsne_euc[:, 0]
    data[1] = tsne_euc[:, 1]

    sns.scatterplot(
        data=data,
        x=0,
        y=1,
        alpha=0.7,
        linewidth=0,
        s=s,
        legend=False,
        hue="label",
        palette=CLASS_COLOR_DICT,
        ax=ax,
    )
    remove_spines(ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # ax.axis("off")
    if left_pair_inds is not None and right_pair_inds is not None:
        add_connections(
            data.iloc[left_pair_inds, 0],
            data.iloc[right_pair_inds, 0],
            data.iloc[left_pair_inds, 1],
            data.iloc[right_pair_inds, 1],
            ax=ax,
        )
    ax.set_title("TSNE o euclidean")

    plt.tight_layout()
    return ax


# %% [markdown]
# ##

sns.set_context("talk", font_scale=1.5)

embeddings = [n2v_latent, ase_latent, svd_embed]
names = ["Node2Vec", "ASE", "Omni-ASE"]
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
for i, (embed, name) in enumerate(zip(embeddings, names)):
    if name == "Omni-ASE":
        plot_pairs(
            embed,
            class_labels[np.concatenate((lp_inds, rp_inds))],
            n_components=16,
            left_pair_inds=np.arange((len(lp_inds))),
            right_pair_inds=np.arange((len(rp_inds))) + len(rp_inds),
            ax=axs[i],
        )
    else:
        plot_pairs(
            embed,
            class_labels,
            n_components=16,
            left_pair_inds=lp_inds,
            right_pair_inds=rp_inds,
            ax=axs[i],
        )
    axs[i].set_title(name)

axs[1].spines["left"].set_visible(True)
axs[1].spines["left"].set_color("grey")
axs[2].spines["left"].set_visible(True)
axs[2].spines["left"].set_color("grey")
stashfig("embed-comparison")
