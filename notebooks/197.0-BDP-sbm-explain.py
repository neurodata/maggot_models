#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter
from sklearn.metrics import adjusted_rand_score

from graspologic.cluster import AutoGMMCluster
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.plot import heatmap, pairplot
from graspologic.simulations import sbm
from src.io import savefig
from src.visualization import set_theme

np.random.seed(88888)
set_theme()


FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


#%%

n_per_comm = 100
B = np.array([[0.4, 0.01], [0.01, 0.4]])
n_comm = len(B)
comm_sizes = np.array(n_comm * [n_per_comm])

adj, labels = sbm(comm_sizes, B, return_labels=True)

palette = dict(zip(np.unique(labels), sns.color_palette("deep", 10)))

from src.visualization import adjplot

adjplot(
    adj,
    sort_class=labels,
    colors=labels,
    ticks=False,
    cbar=False,
    cmap="binary",
    vmin=0,
    center=None,
    vmax=1,
    palette=palette,
)
stashfig("sbm-adj")
#%%
ase = AdjacencySpectralEmbed(n_components=2)
ase_embedding = ase.fit_transform(adj)

#%%


#%%

from src.visualization import remove_spines


def soft_axis_off(ax, spines=True):
    ax.set(xticks=[], xlabel="", yticks=[], ylabel="")
    if not spines:
        remove_spines(ax)


def draw_edges(
    adj,
    data,
    ax,
    x="ase_0",
    y="ase_1",
    hue="labels",
    palette=None,
    linewidth=0.1,
    alpha=0.1,
    subsample=None,
):
    labels = data["labels"]
    if palette is None:
        palette = dict(
            zip(np.arange(len(np.unique(labels))), sns.color_palette("deep", 10))
        )

    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    node_to_label_map = dict(zip(np.arange(len(adj)), labels))

    rows = []
    for i, (pre, post) in enumerate(g.edges):
        rows.append({"pre": pre, "post": post, "edge_idx": i})
    edgelist = pd.DataFrame(rows)
    edgelist["pre_class"] = edgelist["pre"].map(node_to_label_map)

    if subsample:
        size = int(subsample * len(edgelist))
        subsample_inds = np.random.choice(len(edgelist), size=size, replace=False)
        edgelist = edgelist.iloc[subsample_inds]

    pre_edgelist = edgelist.copy()
    post_edgelist = edgelist.copy()

    pre_edgelist["x"] = pre_edgelist["pre"].map(data[x])
    pre_edgelist["y"] = pre_edgelist["pre"].map(data[y])

    post_edgelist["x"] = post_edgelist["post"].map(data[x])
    post_edgelist["y"] = post_edgelist["post"].map(data[y])

    # plot_edgelist = pd.concat((pre_edgelist, post_edgelist), axis=0, ignore_index=True)

    # edge_palette = dict(
    #     zip(edgelist["edge_idx"], edgelist["pre_class"].map(comm_palette))
    # )

    pre_coords = list(zip(pre_edgelist["x"], pre_edgelist["y"]))
    post_coords = list(zip(post_edgelist["x"], post_edgelist["y"]))
    coords = list(zip(pre_coords, post_coords))
    edge_colors = edgelist["pre_class"].map(palette)
    lc = LineCollection(
        coords, colors=edge_colors, linewidths=linewidth, alpha=alpha, zorder=0
    )
    ax.add_collection(lc)


from matplotlib.collections import LineCollection
import networkx as nx

plot_df = pd.DataFrame(
    data=ase_embedding.copy(),
    columns=[f"ase_{i}" for i in range(ase_embedding.shape[1])],
)
plot_df["labels"] = labels

# def reset_legend(ax):
plot_df["ase_0"] += np.random.normal(0, 0.1, size=len(plot_df))
plot_df["ase_1"] += np.random.normal(0, 0.1, size=len(plot_df))

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(
    data=plot_df,
    x="ase_0",
    y="ase_1",
    hue="labels",
    palette=palette,
    legend=False,
    s=60,
)
soft_axis_off(
    ax,
    spines=False,
)
draw_edges(adj, plot_df, ax, palette=palette, linewidth=0.5, alpha=0.5, subsample=0.1)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LinearDiscriminantAnalysis()

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)
pred_labels = gmm.fit_predict(plot_df[["ase_0", "ase_1"]].values)

if (pred_labels == labels).mean() < 0.5:
    gmm.means_ = gmm.means_[::-1]
    gmm.covariances_ = gmm.covariances_[::-1]
    gmm.precisions_ = gmm.precisions_[::-1]
# from src.cluster import make_ellipses

import matplotlib as mpl


def make_ellipses(gmm, ax, i, j, colors, alpha=0.5, equal=False, **kws):
    inds = [j, i]
    for n, color in enumerate(colors):
        print(color)
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][np.ix_(inds, inds)]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[np.ix_(inds, inds)]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][inds])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 3.5 * np.sqrt(2.0) * np.sqrt(v)  # changed from 2
        ell = mpl.patches.Ellipse(
            gmm.means_[n, inds], v[0], v[1], 180 + angle, color=color, **kws
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ax.add_artist(ell)
        if equal:
            ax.set_aspect("equal", "datalim")


colors = list(map(palette.get, [0, 1]))
# colors = [c + (1,) for c in colors]

make_ellipses(gmm, ax, 1, 0, colors, alpha=0.5, fill=False, linewidth=4)

# means = np.array([gmm.means_[0], gmm.means_[1]])
# ax.plot(means[0], means[1])
# midpoint = means.mean(axis=0)
# diff = means[0] - means[1]
# x2 = diff[0] * midpoint[0] / diff[1]
# # anti_means = 100 * means @ np.array([[0, -1], [1, 0]])
# ax.plot([0, midpoint[0]], [0, x2])

# ax.plot([-0.5, 2], [-0.3, 1.3], color="black", linestyle="--", linewidth=2)
# ax.set(xlim=(0.1, 0.9), ylim=(-0.6, 0.9))
# ax.relim()
ax.autoscale_view()
stashfig("sbm-2d")
