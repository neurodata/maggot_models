#%%
from src.data import load_maggot_graph
from pathlib import Path
import pandas as pd
import time
from graspologic.cluster import DivisiveCluster
from src.visualization import CLASS_COLOR_DICT
import numpy as np
import matplotlib.pyplot as plt
from giskard.plot import crosstabplot
import datetime
from src.visualization import set_theme
from src.data import join_node_meta
from src.io import savefig
import ast

set_theme()

t0 = time.time()
CLASS_KEY = "merge_class"
palette = CLASS_COLOR_DICT

mg = load_maggot_graph()
mg = mg[mg.nodes["has_embedding"]]
# mg = mg[mg.nodes["pair_id"] > 1]
nodes = mg.nodes

out_path = Path("./maggot_models/experiments/revamp_gaussian_cluster")

FORMAT = "png"


def stashfig(name, format=FORMAT, **kws):
    savefig(
        name, pathname=out_path / "figs", format=format, dpi=300, save_on=True, **kws
    )


def uncondense_series(condensed_nodes, nodes, key):
    for idx, row in condensed_nodes.iterrows():
        skids = row["skeleton_ids"]
        for skid in skids:
            nodes.loc[int(skid), key] = row[key]


#%%


def cluster_crosstabplot(
    nodes,
    group="cluster_labels",
    order="sum_signal_flow",
    hue="merge_class",
    palette=None,
):
    group_order = (
        nodes.groupby(group)[order].agg(np.median).sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    crosstabplot(
        nodes,
        group=group,
        group_order=group_order,
        hue=hue,
        hue_order=order,
        palette=palette,
        outline=True,
        thickness=0.5,
        ax=ax,
    )
    ax.set(xticks=[], xlabel="Cluster")
    return fig, ax


#%%

# for i, pred_labels in enumerate(hier_labels.T):
#     key = f"dc_labels_level={i}"
#     condensed_nodes[key] = pred_labels
#     fig, ax = cluster_crosstabplot(
#         condensed_nodes,
#         group=key,
#         palette=palette,
#         hue=CLASS_KEY,
#         order="sum_signal_flow",
#     )
#     ax.set_title(f"# clusters = {len(np.unique(pred_labels))}")
#     stashfig(f"crosstabplot-level={i}")
#     uncondense_series(condensed_nodes, nodes, key)
#     join_node_meta(nodes[key], overwrite=True)

#%%

import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import crosstabplot
from giskard.utils import get_paired_inds
from graspologic.cluster import DivisiveCluster
from graspologic.cluster.autogmm import _labels_to_onehot, _onehot_to_initial_params
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from src.data import join_node_meta, load_maggot_graph
from src.visualization import CLASS_COLOR_DICT, set_theme
from src.io import savefig
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


def initialize_gmm(labels, X, cov_type, **kwargs):
    onehot = _labels_to_onehot(labels)
    weights, means, precisions = _onehot_to_initial_params(X, onehot, cov_type)
    gm = GaussianMixture(
        n_components=len(weights),
        covariance_type=cov_type,
        weights_init=weights,
        means_init=means,
        precisions_init=precisions,
    )
    return gm


nodes["inds"] = range(len(nodes))
n_components = 12
latent_cols = [f"latent_{i}" for i in range(n_components)]
X = nodes[latent_cols].values.copy()
X = normalize(X, axis=1, norm="l2")
n_resamples = 25
resample_prop = 0.9
n_per_sample = int(np.ceil(resample_prop * len(nodes)))
models = {}
datas = {}
rows = []
for i in range(n_resamples):
    choices = np.random.choice(len(nodes), size=n_per_sample, replace=False)
    for n_clusters in np.arange(85, 86, 10):
        print((i, n_clusters))
        fit_inds = nodes.iloc[choices]["inds"].values

        agg = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="euclidean", linkage="ward"
        )
        sub_X = X[fit_inds]
        agg_pred_labels = agg.fit_predict(sub_X)
        gmm = initialize_gmm(agg_pred_labels, sub_X, cov_type="full")
        gmm.fit(sub_X)
        gmm_pred_labels = gmm.predict(X)
        row = {
            "subsample": i,
            "n_clusters": n_clusters,
            "gmm": gmm,
            "gmm_pred_labels": gmm_pred_labels,
        }
        rows.append(row)

aris = np.zeros((n_resamples, n_resamples))
for i, row1 in enumerate(rows):
    for j, row2 in enumerate(rows):
        if i < j:
            ari = adjusted_rand_score(row1["gmm_pred_labels"], row2["gmm_pred_labels"])
            aris[i, j] = ari

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    aris,
    square=True,
    mask=aris == 0,
    ax=ax,
    xticklabels=False,
    yticklabels=False,
    cmap="RdBu_r",
    center=0,
    vmax=1,
    cbar_kws=dict(shrink=0.6),
    annot=True,
    annot_kws=dict(fontsize=8),
)
ax.set(title="Pairwise ARIs, GMM o Agglom")
ax.text(
    0.33,
    0.33,
    f"Mean={aris[aris!=0].mean():0.2f}",
    transform=ax.transAxes,
    va="center",
    ha="center",
)
stashfig("pairwise-ari-gmm-o-agglom")

#%%
from graspologic.utils import remap_labels, symmetrize

co_cluster_mat = np.zeros((len(nodes), len(nodes)))
for i, row1 in enumerate(rows):
    for j, row2 in enumerate(rows):
        pred_labels1 = row1["gmm_pred_labels"]
        pred_labels2 = row2["gmm_pred_labels"]
        pred_labels2 = remap_labels(pred_labels1, pred_labels2)
        co_cluster_mat[pred_labels1[None, :] == pred_labels2[:, None]] += 1
        # same_cluster = pred_labels1 == pred_labels2

        #  += 1

co_cluster_mat /= n_resamples ** 2
co_cluster_mat = symmetrize(co_cluster_mat, method="triu")
co_cluster_mat[np.arange(len(co_cluster_mat)), np.arange(len(co_cluster_mat))] = 1
# #%%
# # sns.heatmap(co_cluster_mat)
# from src.visualization import adjplot

# adjplot(
#     co_cluster_mat,
#     meta=nodes,
#     sort_class="merge_class",
#     colors="merge_class",
#     palette=CLASS_COLOR_DICT,
#     cbar=False,
# )

#%%
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from giskard.plot import dissimilarity_clustermap

# sns.clustermap(co_cluster_mat, row)
n_clusters = 85
dissimilarity_clustermap(
    co_cluster_mat,
    colors=nodes["merge_class"].values,
    invert=True,
    method="average",
    palette=CLASS_COLOR_DICT,
    criterion="maxclust",
    t=n_clusters,
    cut=True,
    # t=2,
    # t=0.7,
)
stashfig("co-clustering-gmm-o-agglom")
# TODO look at the adjacency matrix
Z = linkage(squareform(1 - co_cluster_mat), method="average")
pred_labels = fcluster(Z, n_clusters, criterion="maxclust")
name = f"co_cluster_n_clusters={n_clusters}"
nodes[name] = pred_labels
join_node_meta(nodes[name], overwrite=True)

#%%

from anytree import NodeMixin


class RecursiveBiSplitter(NodeMixin):
    def __init__(self, matrix, ):
        self.matrix = matrix



#%%
from graspologic.partition import leiden
from src.visualization import adjplot

partition = leiden(co_cluster_mat, resolution=5)
keys = np.arange(len(co_cluster_mat))
flat_labels = np.vectorize(partition.get)(keys)
adjplot(
    co_cluster_mat,
    sort_class=flat_labels,
    colors=nodes["merge_class"].values,
    palette=palette,
)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
