# %% [markdown]
# #
import os

import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from graspy.plot import pairplot
from src.data import load_metagraph
from src.io import savecsv, savefig
from src.visualization import CLASS_COLOR_DICT
from topologic.embedding import node2vec_embedding

import matplotlib.pyplot as plt

from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
)
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

sns.set_context("talk")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, save_on=True, **kws)


def plot_pairs(
    X,
    labels,
    n_components=8,
    model=None,
    left_pair_inds=None,
    right_pair_inds=None,
    equal=False,
    plot_tsne=False,
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

    fig, axs = plt.subplots(
        n_components, n_components, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X.copy())
    data["label"] = labels

    for i in range(n_components):
        for j in range(n_components):
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

    if plot_tsne:
        mini_axs = axs[-4:, :4]
        gs = mini_axs[0, 0].get_gridspec()
        for ax in mini_axs.ravel():
            ax.remove()
        ax = fig.add_subplot(gs[-4:, :4])
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
            s=12,
            legend=False,
            hue="label",
            palette=CLASS_COLOR_DICT,
            ax=ax,
        )
        ax.axis("off")
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
    return fig, axs


#%% Load and preprocess the data

graph_type = "G"

mg = load_metagraph(graph_type)
mg = mg.make_lcc()

g = mg.g.copy()
meta = mg.meta

# %% [markdown]
# #
from src.utils import get_paired_inds


n2v_params = dict(
    num_walks=[100],
    walk_length=[4, 8, 16],
    dimensions=[16],
    window_size=[1, 2, 4],
    iterations=[1, 3],
    inout_hyperparameter=[0.5, 1, 2],
    return_hyperparameter=[0.5, 1, 2],
)


# meta = meta.loc[node_labels]
meta = meta.sort_index()
meta["inds"] = range(len(meta))
class_labels = meta["merge_class"].values
lp_inds, rp_inds = get_paired_inds(meta)

param_grid = list(ParameterGrid(n2v_params))
np.random.shuffle(param_grid)
for p in tqdm(param_grid):
    embed = node2vec_embedding(g, **p)
    latent = embed[0]
    node_labels = embed[1]
    node_labels = np.vectorize(int)(node_labels)
    sort_inds = np.argsort(node_labels)
    latent = latent[sort_inds]
    fig, _ = plot_pairs(latent, class_labels, plot_tsne=True)
    fig.suptitle("Node2Vec embedding (first 8 dimensions)", y=1.03)
    stashfig(f"n2v_embed_params={p}")
    if np.random.uniform() < 0.05:
        fig, _ = plot_pairs(
            latent,
            class_labels,
            left_pair_inds=lp_inds,
            right_pair_inds=rp_inds,
            plot_tsne=False,
        )
        fig.suptitle("Node2Vec embedding (first 8 dimensions)", y=1.03)
        stashfig(f"n2v_pairs_params={p}")

