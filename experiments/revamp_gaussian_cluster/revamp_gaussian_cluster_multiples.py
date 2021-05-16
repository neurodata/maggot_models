#%%
import ast
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
from sklearn.preprocessing import normalize
from src.data import join_node_meta, load_maggot_graph
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT, set_theme
from giskard.plot import dissimilarity_clustermap
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from graspologic.utils import remap_labels, symmetrize
from sklearn.neighbors import KNeighborsClassifier
from src.block import calculate_blockmodel_likelihood
from tqdm import tqdm
from giskard.plot import matched_stripplot

set_theme()

t0 = time.time()
CLASS_KEY = "merge_class"
palette = CLASS_COLOR_DICT

mg = load_maggot_graph()
mg = mg[mg.nodes["has_embedding"]]
nodes = mg.nodes

condensed_path = Path("./maggot_models/experiments/revamp_embed/outs")
condensed_nodes = pd.read_csv(
    condensed_path / "condensed_nodes.csv",
    index_col=0,
    converters=dict(skeleton_ids=ast.literal_eval),
)

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


def initialize_gmm(X, labels, cov_type, **kwargs):
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


def fit_best_gmm(X, labels, cov_types=["full", "spherical", "diag", "tied"]):
    min_bic = np.inf
    best_model = None
    for cov_type in cov_types:
        gmm = initialize_gmm(X, labels, cov_type)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < min_bic:
            best_model = gmm
            min_bic = bic
    return best_model


def construct_coclustering_matrix(rows):
    n = len(rows.iloc[0]["pred_labels"])
    co_cluster_mat = np.zeros((n, n))
    for i, (idx, row1) in enumerate(rows.iterrows()):
        for j, (idx, row2) in enumerate(rows.iterrows()):
            pred_labels1 = row1["pred_labels"]
            pred_labels2 = row2["pred_labels"]
            pred_labels2 = remap_labels(pred_labels1, pred_labels2)
            co_cluster_mat[pred_labels1[None, :] == pred_labels2[:, None]] += 1
    co_cluster_mat /= n_resamples ** 2
    co_cluster_mat = symmetrize(co_cluster_mat, method="triu")  # TODO I think this is
    # only necessary because of ties in the linear sum assignment problem being resolved
    # differently based on the order of inputs, but should look into it
    co_cluster_mat[np.arange(len(co_cluster_mat)), np.arange(len(co_cluster_mat))] = 1
    return co_cluster_mat


#%%
nodes["inds"] = range(len(nodes))
condensed_nodes["inds"] = range(len(condensed_nodes))
n_components = 14
latent_cols = [f"latent_{i}" for i in range(n_components)]
X = condensed_nodes[latent_cols].values.copy()

# paired_nodes = nodes[nodes["predicted_pair_id"] > 1]
# pair_ids = np.unique(paired_nodes["predicted_pair_id"])
# for pair_id in pair_ids:
#     pair_nodes = nodes[nodes["predicted_pair_id"] == pair_id]
#     if len(pair_nodes) == 2:
#         pair_inds = pair_nodes["inds"]
#         mean_vec = X[pair_inds].mean(axis=0)
#         X[pair_inds, :] = mean_vec[None, :]

refine_gmm = False
X = normalize(X, axis=1, norm="l2")
n_resamples = 100
resample_prop = 0.9
n_per_sample = int(np.ceil(resample_prop * len(condensed_nodes)))
models = {}
datas = {}
rows = []

for n_clusters in np.arange(85, 86, 10):
    for i in tqdm(range(n_resamples)):
        choices = np.random.choice(
            len(condensed_nodes), size=n_per_sample, replace=False
        )
        fit_inds = condensed_nodes.iloc[choices]["inds"].values
        agg = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="euclidean", linkage="ward"
        )
        sub_X = X[fit_inds]
        agg_pred_labels = agg.fit_predict(sub_X)
        for refine_gmm in [True, False]:
            if refine_gmm:
                gmm = fit_best_gmm(sub_X, agg_pred_labels)
                pred_labels = gmm.predict(X)
            else:
                non_fit_inds = np.setdiff1d(np.arange(len(condensed_nodes)), fit_inds)
                pred_labels = np.zeros(len(X))
                pred_labels[fit_inds] = agg_pred_labels
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(sub_X, agg_pred_labels)
                pred_labels[non_fit_inds] = knn.predict(X[non_fit_inds])
            row = {
                "subsample": i,
                "n_clusters": n_clusters,
                "pred_labels": pred_labels,
                "refine_gmm": refine_gmm,
            }
            rows.append(row)

results = pd.DataFrame(rows)

# # aris = np.zeros((n_resamples, n_resamples))a
# aris = []
# for i, row1 in enumerate(rows):
#     for j, row2 in enumerate(rows):
#         if i < j:
#             ari = adjusted_rand_score(row1["pred_labels"], row2["pred_labels"])
#             aris.append(ari)
# print(np.mean(aris))

#%%

adj = mg.sum.adj
lp_inds, rp_inds = get_paired_inds(nodes)
likelihood_rows = []
for i in range(10):
    sub_results = results[results["subsample"] == i]
    for refine_gmm in [True, False]:
        sub_sub_results = sub_results[sub_results["refine_gmm"] == refine_gmm]
        key = f"labels_subsample={i}_refine_gmm={refine_gmm}"
        condensed_nodes[key] = sub_sub_results["pred_labels"].iloc[0]
        uncondense_series(condensed_nodes, nodes, key)
        labels = nodes[key]
        likelihoods = calculate_blockmodel_likelihood(adj, labels, lp_inds, rp_inds)
        likelihoods = likelihoods.groupby("test").mean()
        likelihoods["subsample"] = i
        likelihoods["refine_gmm"] = refine_gmm
        likelihood_rows.append(likelihoods)
likelihood_results = pd.concat(likelihood_rows, axis=0, ignore_index=False)
likelihood_results = likelihood_results.reset_index()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]
matched_stripplot(
    data=likelihood_results[likelihood_results["test"] == "Same"],
    hue="refine_gmm",
    match="subsample",
    x="refine_gmm",
    y="norm_score",
    ax=ax,
)
ax = axs[1]
matched_stripplot(
    data=likelihood_results[likelihood_results["test"] == "Opposite"],
    hue="refine_gmm",
    match="subsample",
    x="refine_gmm",
    y="norm_score",
    ax=ax,
)
plt.tight_layout()

#%%
co_cluster_mat = construct_coclustering_matrix(results[results["refine_gmm"] == True])
n_clusters = 85
dissimilarity_clustermap(
    co_cluster_mat,
    colors=condensed_nodes["merge_class"].values,
    invert=True,
    method="average",
    palette=CLASS_COLOR_DICT,
    criterion="maxclust",
    t=n_clusters,
    cut=True,
)
stashfig("co-clustering-gmm-o-agglom")
Z = linkage(squareform(1 - co_cluster_mat), method="average")
pred_labels = fcluster(Z, n_clusters, criterion="maxclust")

uni_labels, counts = np.unique(pred_labels, return_counts=True)
counts

#%%
min_size = 4
# correct small clusters
uni_labels_full = uni_labels[counts >= min_size]
uni_labels_too_small = uni_labels[counts < min_size]

inds = np.arange(len(condensed_nodes))

for source_label in uni_labels_too_small:
    source_inds = inds[pred_labels == source_label]
    for ind in source_inds:
        dists = []
        for target_label in uni_labels_full:
            mask = pred_labels == target_label
            mean_dist = co_cluster_mat[ind][mask].mean()
            dists.append(mean_dist)
        selected_cluster_ind = np.argmax(dists)
        selected_cluster = uni_labels_full[selected_cluster_ind]
        pred_labels[ind] = selected_cluster

name = f"co_cluster_n_clusters={n_clusters}"
condensed_nodes[name] = pred_labels

#%%
cluster_crosstabplot(condensed_nodes, name, palette=CLASS_COLOR_DICT)

#%%
uncondense_series(condensed_nodes, nodes, name)
join_node_meta(nodes[name], overwrite=True)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")

# %%
