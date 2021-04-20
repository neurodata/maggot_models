#%%
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_cluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances

from giskard.plot import crosstabplot, dissimilarity_clustermap
from graspologic.plot.plot_matrix import scattermap
from graspologic.utils import symmetrize
from src.data import load_maggot_graph
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import adjplot, set_theme

t0 = time.time()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/agglomerative_cluster/figs",
        **kws,
    )


out_dir = Path(__file__).parent / "outs"
embedding_loc = Path("maggot_models/experiments/embed/outs/stage2_embedding.csv")

#%% load the embedding, get the correct subset of data
embedding_df = pd.read_csv(embedding_loc, index_col=0)
embedding_df = embedding_df.groupby(embedding_df.index).mean()
mg = load_maggot_graph()
nodes = mg.nodes.copy()
nodes = nodes[nodes.index.isin(embedding_df.index)]
nodes = nodes[nodes["paper_clustered_neurons"]]
embedding_df = embedding_df[embedding_df.index.isin(nodes.index)]
nodes = nodes.reindex(embedding_df.index)
embedding = embedding_df.values

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


def compute_distances(embedding, n_components=None, metric="cosine"):
    return symmetrize(pairwise_distances(embedding[:, :n_components], metric=metric))


pair_mat = np.zeros((len(embedding), len(embedding)))
incomplete_mat = pair_mat.copy()
nodes["inds"] = range(len(nodes))
for idx, self_row in nodes.iterrows():
    pair = self_row["pair"]
    if pair > -1:
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

method = "ward"
metric = "cosine"
criterion = "distance"
threshold = 2
n_components = 48


def plot_dissimilarity_clustering(
    embedding,
    pair_mat,
    incomplete_mat,
    metric="cosine",
    criterion="distance",
    threshold=2,
    n_components=32,
):
    distances = compute_distances(embedding, n_components=n_components, metric=metric)
    clustergrid = dissimilarity_clustermap(
        distances,
        colors=nodes["merge_class"],
        palette=palette,
        method=method,
        cut=True,
        criterion=criterion,
        t=threshold,
        figsize=(16, 16),
    )
    inds = clustergrid.dendrogram_row.reordered_ind
    pair_mat = pair_mat[np.ix_(inds, inds)]
    incomplete_mat = incomplete_mat[np.ix_(inds, inds)]
    scattermap(
        pair_mat, ax=clustergrid.ax_heatmap, sizes=(10, 10), zorder=5, color="green"
    )
    scattermap(
        incomplete_mat,
        ax=clustergrid.ax_heatmap,
        sizes=(20, 20),
        color="turquoise",
        zorder=10,
    )

    Z = linkage_cluster(squareform(distances), method=method)
    flat_labels = fcluster(Z, threshold, criterion=criterion)

    clustergrid.ax_col_colors.set_yticks([0.5, 1.5])
    clustergrid.ax_col_colors.set_yticklabels(
        [f"Cluster indicator ({len(np.unique(flat_labels))})", "Known labels"]
    )
    clustergrid.ax_col_colors.yaxis.tick_right()


set_theme()

n_components_range = [32, 48, 64]
thresholds = [1, 1.5, 2, 2.5]
for n_components in n_components_range:
    for threshold in thresholds:
        plot_dissimilarity_clustering(
            embedding,
            pair_mat,
            incomplete_mat,
            metric=metric,
            criterion=criterion,
            threshold=threshold,
            n_components=n_components,
        )
        name = "cut-dissimilarity-clustermap"
        name += f"-metric={metric}-linkage={method}-t={threshold}-n_components={n_components}"
        stashfig(name)
#%% choose a final set
n_components = 64
threshold = 2
basename = (
    f"-metric={metric}-linkage={method}-t={threshold}-n_components={n_components}"
)
distances = compute_distances(embedding, n_components=n_components, metric=metric)
Z = linkage_cluster(squareform(distances), method=method)
flat_labels = fcluster(Z, threshold, criterion=criterion)

linkage_df = pd.DataFrame(data=Z)
out_path = Path("maggot_models/experiments/agglomerative_cluster/outs")
linkage_df.to_csv(out_path / "linkage.csv")

linkage_index = pd.Series(nodes.index, name="skeleton_id")
linkage_index.to_csv(out_path / "linkage_index.csv")


#%%

rows = []
ts = list(np.linspace(Z[:, 2].max(), 0, 40))
if threshold not in ts:
    ts.append(threshold)
for t in ts:
    flat_labels = fcluster(Z, t, criterion=criterion)
    nodes["flat_labels"] = flat_labels
    n_unique_by_pair = nodes.groupby("pair_id")["flat_labels"].nunique()
    n_unique_by_pair = n_unique_by_pair[n_unique_by_pair.index != -1]
    p_same_cluster = (n_unique_by_pair == 1).mean()
    rows.append({"t": t, "p_same_cluster": p_same_cluster})
results = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="t", y="p_same_cluster", ax=ax)
ax.axvline(threshold, color="black", linestyle=":")
p_same_at_threshhold = results[results["t"] == threshold]["p_same_cluster"].iloc[0]
ax.axhline(
    p_same_at_threshhold,
    color="black",
    linestyle=":",
)
ax.text(
    threshold - 0.15,
    p_same_at_threshhold,
    f"{p_same_at_threshhold:0.3f}",
    color="black",
    ha="left",
    va="bottom",
)
ax.invert_xaxis()
ax.set(xlabel="Dendrogram threshold", ylabel="Proportion of pairs in same cluster")
stashfig("pair-concordance" + basename)
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
stashfig("crosstabplot" + basename)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
