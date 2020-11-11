#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter
from sklearn.metrics import adjusted_rand_score

from graspologic.cluster import AutoGMMCluster, DivisiveCluster
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.plot import heatmap, pairplot
from graspologic.simulations import sbm
from src.io import savefig
from src.visualization import set_theme

set_theme()


FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


#%%

n_levels = 2
B = np.array([[0.6, 0.05], [0.05, 0.4]])
n_comm = len(B)
# scale = 0.5  # for this scale divclust and agmm both work.
scale = 0.5
lowest_B = np.repeat(
    np.repeat(B, n_comm ** n_levels, axis=1), n_comm ** n_levels, axis=0
)
Bs_by_level = [lowest_B.copy()]
cumulative_labels = []
for i in range(n_levels):
    if i < n_levels - 1:
        temp_B = np.repeat(
            np.repeat(B, n_comm * (n_levels - i - 1), axis=1),
            n_comm * (n_levels - i - 1),
            axis=0,
        )
    else:
        temp_B = B
    temp_B *= scale
    temp_B -= temp_B.mean()
    level_labels = np.repeat(
        np.arange(n_comm * (n_levels + i - 1)), n_comm * (n_levels - i)
    )
    for ul in np.unique(level_labels):
        mask = level_labels == ul
        lowest_B[np.ix_(mask, mask)] += temp_B
    Bs_by_level.append(lowest_B.copy())

vmax = np.max(Bs_by_level)


fig, axs = plt.subplots(1, n_levels + 1, figsize=(6 * n_levels, 6))
for i, level_B in enumerate(Bs_by_level):
    sns.heatmap(
        level_B,
        cmap="RdBu_r",
        square=True,
        xticklabels=False,
        yticklabels=False,
        center=0,
        ax=axs[i],
        cbar=False,
        vmin=0,
        vmax=vmax,
    )
stashfig("B-by-level")

#%%
n_per_comm = 200
comm_sizes = np.array(n_comm ** (n_levels + 1) * [n_per_comm])
graph, labels = sbm(
    comm_sizes, lowest_B, directed=False, loops=False, return_labels=True
)
heatmap(graph, cbar=False)
stashfig("graph-sample-adj")

#%%


ase = AdjacencySpectralEmbed(n_components=n_comm ** (n_levels + 1))
embedding = ase.fit_transform(graph)

pairplot(embedding, labels=labels)
stashfig("pairplot-true-labels")

#%%

div_clust = DivisiveCluster(max_level=3)
div_labels = div_clust.fit_predict(embedding)
_, div_flat_labels = np.unique(div_labels, axis=0, return_inverse=True)
div_ari = adjusted_rand_score(labels, div_flat_labels)
print(f"DivisiveCluster ARI: {div_ari}")

agmm_clust = AutoGMMCluster(min_components=8, max_components=8)
agmm_labels = agmm_clust.fit_predict(embedding)
agmm_ari = adjusted_rand_score(labels, agmm_labels)
print(f"AutoGMMCluster ARI: {agmm_ari}")

#%%
pg = pairplot(embedding, labels=div_flat_labels)


center = np.mean(embedding, axis=0)

last = center
lines = []
for level, group in enumerate(LevelOrderGroupIter(div_clust)):
    for node in group:
        if level > 0:
            last = node.parent.model_.means_[node.parent_model_index]
        if hasattr(node, "model_"):
            means = node.model_.means_
            for mean in means:
                line = np.array([last, mean])
                lines.append(line)

n_dims = embedding.shape[1]

for i in range(n_dims):
    for j in range(n_dims):
        if i != j:
            ax = pg.axes[i, j]
            for line in lines:
                xs = line[:, j]
                ys = line[:, i]
                ax.plot(xs, ys, color="black")

stashfig("pairplot-wih-dendrogram")
#%%


#%%

# from umap import UMAP


# dims = [2, 4, 8]
# umap = UMAP(min_dist=1, n_neighbors=100)
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# for level in range(n_levels + 1):
#     ax = axs[level]
#     _, flat_pred_labels = np.unique(
#         div_pred_labels[:, : level + 1], axis=0, return_inverse=True
#     )
#     umap_embedding = umap.fit_transform(embedding[:, : dims[level]])
#     plot_df = pd.DataFrame(
#         data=umap_embedding,
#         columns=[f"umap_{i}" for i in range(umap_embedding.shape[1])],
#     )
#     plot_df["labels"] = flat_pred_labels.astype("str")
#     sns.scatterplot(
#         data=plot_df, x="umap_0", y="umap_1", hue="labels", ax=ax, palette="deep", s=20
#     )

#%%

# import networkx as nx

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# umap = UMAP(min_dist=1, n_neighbors=100)
# umap_embedding = umap.fit_transform(embedding[:, : dims[level]])
# plot_df = pd.DataFrame(
#     data=umap_embedding,
#     columns=[f"umap_{i}" for i in range(umap_embedding.shape[1])],
# )
# plot_df["labels"] = labels.astype("str")
# sns.scatterplot(
#     data=plot_df,
#     x="umap_0",
#     y="umap_1",
#     hue="labels",
#     ax=ax,
#     palette="deep",
#     s=20,
# )
# ax.get_legend().remove()
# ax.axis("off")

# g = nx.from_numpy_array(graph)

# from matplotlib.collections import LineCollection

# node_to_label_map = dict(zip(np.arange(len(graph)), labels))
# comm_palette = dict(
#     zip(np.arange(len(np.unique(labels))), sns.color_palette("deep", 10))
# )
# x_key = "umap_0"
# y_key = "umap_1"

# rows = []
# for i, (pre, post) in enumerate(g.edges):
#     rows.append({"pre": pre, "post": post, "edge_idx": i})
# edgelist = pd.DataFrame(rows)
# edgelist["pre_class"] = edgelist["pre"].map(node_to_label_map)

# pre_edgelist = edgelist.copy()
# post_edgelist = edgelist.copy()

# pre_edgelist["x"] = pre_edgelist["pre"].map(plot_df[x_key])
# pre_edgelist["y"] = pre_edgelist["pre"].map(plot_df[y_key])

# post_edgelist["x"] = post_edgelist["post"].map(plot_df[x_key])
# post_edgelist["y"] = post_edgelist["post"].map(plot_df[y_key])

# plot_edgelist = pd.concat((pre_edgelist, post_edgelist), axis=0, ignore_index=True)

# edge_palette = dict(zip(edgelist["edge_idx"], edgelist["pre_class"].map(comm_palette)))

# pre_coords = list(zip(pre_edgelist["x"], pre_edgelist["y"]))
# post_coords = list(zip(post_edgelist["x"], post_edgelist["y"]))
# coords = list(zip(pre_coords, post_coords))
# edge_colors = edgelist["pre_class"].map(comm_palette)
# lc = LineCollection(coords, colors=edge_colors, linewidths=0.05, alpha=0.1, zorder=0)
# ax.add_collection(lc)
# stashfig("graph-layout-lowest-level")
