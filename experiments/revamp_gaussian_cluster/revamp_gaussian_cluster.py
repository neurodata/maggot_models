#%%
from src.data import load_maggot_graph
from pathlib import Path
import pandas as pd
import time
from graspologic.cluster import DivisiveCluster, AutoGMMCluster
from src.visualization import CLASS_COLOR_DICT
import numpy as np
import matplotlib.pyplot as plt
from giskard.plot import crosstabplot
import datetime
from src.visualization import set_theme
from src.data import join_node_meta
from src.io import savefig
import ast
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from graspologic.utils import symmetrize

set_theme()

t0 = time.time()
CLASS_KEY = "merge_class"
palette = CLASS_COLOR_DICT

mg = load_maggot_graph()
nodes = mg.nodes

out_path = Path("./maggot_models/experiments/revamp_embed")

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


condensed_nodes = pd.read_csv(
    out_path / "outs/condensed_nodes.csv",
    index_col=0,
    converters=dict(skeleton_ids=ast.literal_eval),
)

n_components = 12
latent = condensed_nodes[[f"latent_{i}" for i in range(n_components)]].values


#%%
currtime = time.time()
dc = DivisiveCluster(
    min_split=16,
    max_level=8,
    cluster_kws=dict(
        kmeans_n_init=25,
        affinity=["euclidean", "cosine", "none"],
        linkage=["ward", "average"],
    ),
)
# covariance_type=["full", "diag", "spherical"]
hier_labels = dc.fit_predict(latent, fcluster=True)
print(f"{time.time() - currtime:.3f} seconds elapsed for divisive clustering.")

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


for i, pred_labels in enumerate(hier_labels.T):
    key = f"dc_labels_level={i}"
    condensed_nodes[key] = pred_labels
    fig, ax = cluster_crosstabplot(
        condensed_nodes,
        group=key,
        palette=palette,
        hue=CLASS_KEY,
        order="sum_signal_flow",
    )
    ax.set_title(f"# clusters = {len(np.unique(pred_labels))}")
    stashfig(f"crosstabplot-level={i}")
    uncondense_series(condensed_nodes, nodes, key)
    join_node_meta(nodes[key], overwrite=True)

#%%

n_components = 14
n_clusters = 85
latent = condensed_nodes[[f"latent_{i}" for i in range(n_components)]].values.copy()
latent = normalize(latent, norm="l2", axis=1)
# agg = AgglomerativeClustering(
#     n_clusters=n_clusters, affinity="cosine", linkage="average"
# )
# pred_labels = agg.fit_predict(latent)

dists = symmetrize(pairwise_distances(latent, metric="euclidean"))
Z = linkage(squareform(dists), method="ward")
pred_labels = fcluster(Z, n_clusters, criterion="maxclust")

key = f"agg_labels_n_clusters={n_clusters}"
condensed_nodes[key] = pred_labels
fig, ax = cluster_crosstabplot(
    condensed_nodes,
    group=key,
    palette=palette,
    hue=CLASS_KEY,
    order="sum_signal_flow",
)
ax.set_title(f"# clusters = {len(np.unique(pred_labels))}")
stashfig(f"agg-n_clusters={n_clusters}")
uncondense_series(condensed_nodes, nodes, key)
join_node_meta(nodes[key], overwrite=True)

#%%
agmm = AutoGMMCluster(
    min_components=n_clusters,
    max_components=n_clusters,
    label_init=pred_labels,
    n_jobs=-1,
    covariance_type="full",
)
agmm_pred_labels = agmm.fit_predict(latent)

#%%
key = f"agmm_agg_n_clusters={n_clusters}"
condensed_nodes[key] = agmm_pred_labels
fig, ax = cluster_crosstabplot(
    condensed_nodes,
    group=key,
    palette=palette,
    hue=CLASS_KEY,
    order="sum_signal_flow",
)
ax.set_title(f"# clusters = {len(np.unique(pred_labels))}")
stashfig(key)
uncondense_series(condensed_nodes, nodes, key)
join_node_meta(nodes[key], overwrite=True)

#%%
from sklearn.metrics import adjusted_rand_score, rand_score

adjusted_rand_score(agmm_pred_labels, pred_labels)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
