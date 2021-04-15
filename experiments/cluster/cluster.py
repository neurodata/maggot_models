#%%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from giskard.plot import dissimilarity_clustermap
from graspologic.plot.plot_matrix import scattermap
from graspologic.utils import symmetrize
from src.data import load_maggot_graph
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT as palette


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/cluster/figs",
        **kws,
    )


embedding_loc = Path("maggot_models/experiments/embed/outs/stage2_embedding.csv")

#%%
embedding_df = pd.read_csv(embedding_loc, index_col=0)
mg = load_maggot_graph()
nodes = mg.nodes.copy()
nodes = nodes[nodes.index.isin(embedding_df.index)]
nodes = nodes.reindex(embedding_df.index)
# TODO just grab the ones michael wanted us to cluster
embedding_df.reindex()
embedding = embedding_df.values
# winners seem like
# ward, MAYBE average, single
# cosine or euclidean
# tried a bunch of others, didn't like most
linkages = ["ward", "single", "complete", "average"]
metrics = ["cosine", "euclidean"]
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

        is_incomplete = pair_row["incomplete"] | self_row["incomplete"]
        if isinstance(is_incomplete, pd.Series):
            is_incomplete = is_incomplete.max()
        if is_incomplete:
            incomplete_mat[self_ind, pair_ind] = 1

method = "ward"
metric = "cosine"
t = 2
distances = symmetrize(pairwise_distances(embedding, metric=metric))
clustergrid = dissimilarity_clustermap(
    distances,
    colors=nodes["merge_class"],
    palette=palette,
    method=method,
    cut=True,
    criterion="distance",
    t=t,
)
inds = clustergrid.dendrogram_row.reordered_ind
pair_mat = pair_mat[np.ix_(inds, inds)]
incomplete_map = incomplete_mat[np.ix_(inds, inds)]
scattermap(pair_mat, ax=clustergrid.ax_heatmap, sizes=(10, 10))
scattermap(
    incomplete_mat, ax=clustergrid.ax_heatmap, sizes=(20, 20), color="black", zorder=10
)

stashfig(f"cut-dissimilarity-clustermap-metric={metric}-linkage={linkage}-t={t}")

#%%


# def _get_separator_info(meta, group):
#     if meta is None or group is None:
#         return None

#     sep = meta.groupby(by=group, sort=False).first()
#     sep_inds = sep["sort_idx"].values
#     sep_inds = list(sep["sort_idx"].values)
#     last = meta.groupby(group, sort=False).last()
#     sep_inds.append(last["sort_idx"].values[-1] + 1)
#     return np.array(sep_inds)


# %% [markdown]
# ## Clustering

# parameters
n_levels = 10  # max # of splits in the recursive clustering
metric = "bic"  # metric on which to decide best split
# bic_ratio = 1  # ratio used for whether or not to split
# d = 8  # embedding dimension

params = [
    # {"d": 6, "bic_ratio": 0.8, "min_split": 32},
    # {"d": 6, "bic_ratio": 0.9, "min_split": 32},
    # {"d": 8, "bic_ratio": 0.8, "min_split": 32},
    # {"d": 8, "bic_ratio": 0.9, "min_split": 32},
    {"d": 8, "bic_ratio": 0, "min_split": 32},
    {"d": 8, "bic_ratio": 0.95, "min_split": 32},
    {"d": 8, "bic_ratio": 1, "min_split": 32},
    # {"d": 10, "bic_ratio": 0.9, "min_split": 32},
]

for p in params:
    print(p)
    d = p["d"]
    bic_ratio = p["bic_ratio"]
    min_split = p["min_split"]
    X = svd_embed[:, :d]
    basename = (
        f"-method={omni_method}-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"
    )
    title = f"Method={omni_method}, d={d}, BIC ratio={bic_ratio}-min_split={min_split}"

    currtime = time.time()

    np.random.seed(8888)
    mc = BinaryCluster(
        "0",
        adj=adj,  # stored for plotting, basically
        n_init=50,  # number of initializations for GMM at each stage
        meta=new_meta,  # stored for plotting and adding labels
        stashfig=stashfig,  # for saving figures along the way
        X=X,  # input data that actually matters
        bic_ratio=bic_ratio,
        reembed=False,
        min_split=min_split,
    )

    mc.fit(n_levels=n_levels, metric=metric)
    print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for clustering")

    inds = np.concatenate((lp_inds, rp_inds))
    cluster_adj = adj[np.ix_(inds, inds)]
    cluster_meta = mc.meta
    cluster_meta["sf"] = -signal_flow(cluster_adj)  # for some of the sorting

    # save results
    stashcsv(cluster_meta, "meta" + basename)
    adj_df = pd.DataFrame(
        cluster_adj, index=cluster_meta.index, columns=cluster_meta.index
    )
    stashcsv(adj_df, "adj" + basename)

    # # plot results
    # lowest_level = 7  # last level to show for dendrograms, adjacencies
    # plot_clustering_results(
    #     cluster_adj, cluster_meta, basename, lowest_level=lowest_level
    # )
    print()
