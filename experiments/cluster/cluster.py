#%%
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_cluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances

from giskard.plot import crosstabplot, dissimilarity_clustermap
from graspologic.plot.plot_matrix import scattermap
from graspologic.utils import symmetrize
from src.cluster import BinaryCluster
from src.data import load_maggot_graph
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import adjplot, set_theme

t0 = time.time()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/cluster/figs",
        **kws,
    )


out_dir = Path(__file__).parent / "outs"
embedding_loc = Path("maggot_models/experiments/embed/outs/stage2_embedding.csv")

#%%
embedding_df = pd.read_csv(embedding_loc, index_col=0)
embedding_df = embedding_df.groupby(embedding_df.index).mean()
mg = load_maggot_graph()
nodes = mg.nodes.copy()
nodes = nodes[nodes.index.isin(embedding_df.index)]
nodes = nodes[nodes["paper_clustered_neurons"]]
embedding_df = embedding_df[embedding_df.index.isin(nodes.index)]
nodes = nodes.reindex(embedding_df.index)
embedding = embedding_df.values

for idx, row in nodes.iterrows():
    if row["pair"] not in nodes.index and row["pair"] != -1:
        print(row["pair"])

#%%
# winners seem like
# ward, MAYBE average, single
# cosine or euclidean
# tried a bunch of others, didn't like most
linkages = []  # ["ward", "single", "complete", "average"]
metrics = []  # ["cosine", "euclidean"]
for metric in metrics:
    for linkage in linkages:
        distances = symmetrize(pairwise_distances(embedding, metric=metric))
        dissimilarity_clustermap(
            distances, colors=nodes["merge_class"], palette=palette, method=linkage
        )
        stashfig(f"dissimilarity-clustermap-metric={metric}-linkage={linkage}")

#%%
pair_mat = np.zeros((len(embedding), len(embedding)))
incomplete_mat = pair_mat.copy()
nodes["inds"] = range(len(nodes))
for idx, self_row in nodes.iterrows():
    pair = self_row["pair"]
    if pair != -1:
        pair_row = nodes.loc[pair]
        pair_ind = pair_row["inds"]
        self_ind = self_row["inds"]
        pair_mat[self_ind, pair_ind] = 1

        if isinstance(pair_ind, np.int64):
            is_incomplete = (
                pair_row["incomplete"]
                or self_row["incomplete"]
                or self_row["developmental_defect"]
                or pair_row["developmental_defect"]
            )

            if is_incomplete:
                incomplete_mat[self_ind, pair_ind] = 1

set_theme()
method = "ward"
metric = "cosine"
criterion = "distance"
n_components = 20 # had this at 16
t = 1
distances = symmetrize(pairwise_distances(embedding[:, :n_components], metric=metric))
clustergrid = dissimilarity_clustermap(
    distances,
    colors=nodes["merge_class"],
    palette=palette,
    method=method,
    cut=True,
    criterion=criterion,
    t=t,
    figsize=(16, 16),
)
inds = clustergrid.dendrogram_row.reordered_ind
pair_mat = pair_mat[np.ix_(inds, inds)]
incomplete_mat = incomplete_mat[np.ix_(inds, inds)]
scattermap(pair_mat, ax=clustergrid.ax_heatmap, sizes=(10, 10), zorder=5, color="green")
scattermap(
    incomplete_mat,
    ax=clustergrid.ax_heatmap,
    sizes=(20, 20),
    color="turquoise",
    zorder=10,
)

Z = linkage_cluster(squareform(distances), method=method)
flat_labels = fcluster(Z, t, criterion=criterion)

clustergrid.ax_col_colors.set_yticks([0.5, 1.5])
clustergrid.ax_col_colors.set_yticklabels(
    [f"Cluster indicator ({len(np.unique(flat_labels))})", "Known labels"]
)
clustergrid.ax_col_colors.yaxis.tick_right()
stashfig(f"cut-dissimilarity-clustermap-metric={metric}-linkage={method}-t={t}")

#%%
nodes["flat_labels"] = flat_labels
n_unique_by_pair = nodes.groupby("pair_id")["flat_labels"].nunique()
n_unique_by_pair = n_unique_by_pair[n_unique_by_pair.index != -1]
(n_unique_by_pair == 1).mean()

#%%
nodes["_inds"] = range(len(nodes))
left_nodes = nodes[nodes["hemisphere"] == "L"].copy()
left_paired_nodes = left_nodes[left_nodes["pair_id"] != -1]
right_nodes = nodes[nodes["hemisphere"] == "R"].copy()
right_paired_nodes = right_nodes[right_nodes["pair_id"] != -1]

# HACK this only works because all the duplicate nodes are on the right
lp_inds = left_paired_nodes.loc[right_paired_nodes["pair"]]["_inds"]
rp_inds = right_paired_nodes["_inds"]
print("Pairs all valid: ")
print((nodes.iloc[lp_inds].index == nodes.iloc[rp_inds]["pair"]).all())
left_inds = lp_inds
right_inds = rp_inds

from sklearn.neighbors import NearestNeighbors


def compute_nn_ranks(left_X, right_X, max_n_neighbors=None, metric="jaccard"):
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

    rank_data = np.concatenate((left_match_rank, right_match_rank))
    rank_data = pd.Series(rank_data, name="pair_nn_rank")
    rank_data = rank_data.to_frame()
    rank_data["metric"] = metric
    rank_data["side"] = len(left_X) * ["Left"] + len(right_X) * ["Right"]
    return rank_data

n_components = 32
left_embedding = embedding[left_inds, :n_components]
right_embedding = embedding[right_inds, :n_components]
rank_data = compute_nn_ranks(left_embedding, right_embedding, metric="euclidean")

#%%
import seaborn as sns

results = rank_data
if results["pair_nn_rank"].min() == 0:
    results["pair_nn_rank"] += 1
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.ecdfplot(
    data=results,
    x="pair_nn_rank",
    hue="metric",
    ax=ax,
)
stashfig("pair_nn_rank")

#%%

nodes["flat_labels"] = flat_labels
group_order = (
    nodes.groupby("flat_labels")["sum_signal_flow"]
    .mean()
    .sort_values(ascending=False)
    .index
)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
crosstabplot(
    nodes[nodes["hemisphere"] == "L"],
    group="flat_labels",
    group_order=group_order,
    hue="merge_class",
    hue_order="sum_signal_flow",
    palette=palette,
    outline=True,
    shift=-0.2,
    thickness=0.25,
    ax=ax,
)
crosstabplot(
    nodes[nodes["hemisphere"] == "R"],
    group="flat_labels",
    group_order=group_order,
    hue="merge_class",
    hue_order="sum_signal_flow",
    palette=palette,
    outline=True,
    shift=0.2,
    thickness=0.25,
    ax=ax,
)
ax.set(xticks=[], xlabel="Cluster")
stashfig(f"crosstabplot-metric={metric}-linkage={method}-t={t}")


# #%%
# index = nodes.index.unique()
# mg.nodes = mg.nodes.reindex(index)
# flat_labels_series = pd.Series(index=index, data=flat_labels)
# flat_labels_series[index.unique]
# adj = mg.sum.adj

# adjplot(adj, plot_type="scattermap", sort_class=flat_labels_series)


# %% [markdown]
# ## Clustering

from graspologic.cluster import DivisiveCluster

# parameters
n_levels = 10  # max # of splits in the recursive clustering
metric = "bic"  # metric on which to decide best split

params = [
    {"d": 8, "bic_ratio": 0, "min_split": 32},
    {"d": 8, "bic_ratio": 0.95, "min_split": 32},
]

for p in params:
    print(p)
    d = p["d"]
    bic_ratio = p["bic_ratio"]
    min_split = p["min_split"]
    X = embedding[:, :d]
    basename = f"-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"

    currtime = time.time()
    np.random.seed(8888)
    mc = BinaryCluster(
        "0",
        n_init=50,  # number of initializations for GMM at each stage
        meta=nodes,  # stored for plotting and adding labels
        X=X,  # input data that actually matters
        bic_ratio=bic_ratio,
        reembed=False,
        min_split=min_split,
    )

    mc.fit(n_levels=n_levels, metric=metric)
    print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for clustering")

    cluster_meta = mc.meta

    # save results
    cluster_meta.to_csv("meta" + basename)

    print()

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
