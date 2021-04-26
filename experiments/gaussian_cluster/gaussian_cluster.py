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
from src.data import load_maggot_graph, join_node_meta
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import adjplot, set_theme

t0 = time.time()

set_theme()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/gaussian_cluster/figs",
        **kws,
    )


out_dir = Path(__file__).parent / "outs"
embedding_loc = Path("maggot_models/experiments/embed/outs/stage2_embedding.csv")

#%% load the embedding, get the correct subset of data
embedding_df = pd.read_csv(embedding_loc, index_col=0)
embedding_df = embedding_df.groupby(embedding_df.index).mean()
mg = load_maggot_graph()
mg = mg[mg.nodes["has_embedding"]]
nodes = mg.nodes.copy()
nodes = nodes[nodes.index.isin(embedding_df.index)]
embedding_df = embedding_df[embedding_df.index.isin(nodes.index)]
nodes = nodes.reindex(embedding_df.index)
embedding = embedding_df.values

# %% [markdown]
# ## Clustering

from graspologic.cluster import DivisiveCluster
from graspologic.cluster.autogmm import _labels_to_onehot, _onehot_to_initial_params
from sklearn.mixture import GaussianMixture

# parameters
n_levels = 10  # max # of splits in the recursive clustering
metric = "bic"  # metric on which to decide best split
n_components = 8
X = embedding[:, :n_components]

flat_labels = nodes["agglom_labels_t=0.65_n_components=64"].astype(int)
covariance_type = "full"
reg_covar = 1e-06
onehot = _labels_to_onehot(flat_labels)
weights_init, means_init, precisions_init = _onehot_to_initial_params(
    X, onehot, covariance_type, reg_covar=reg_covar
)
gm = GaussianMixture(
    n_components=len(weights_init),
    covariance_type=covariance_type,
    reg_covar=reg_covar,
    weights_init=weights_init,
    means_init=means_init,
    precisions_init=precisions_init,
)
pred_labels = gm.fit_predict(X)

stashfig("crosstabplot_gmm_o_agglom")


#%%
currtime = time.time()
n_components = 8
min_split = 32
n_components_range = [8, 10, 12]
min_split_range = [16, 32]
for n_components in n_components_range:
    for min_split in min_split_range:
        X = embedding[:, :n_components]
        dc = DivisiveCluster(
            cluster_kws=dict(n_init=1), min_split=min_split, max_level=10
        )
        hier_labels = dc.fit_predict(X, fcluster=True)
        name = f"dc_labels_n_components={n_components}_min_split={min_split}"
        label_series = pd.DataFrame(data=hier_labels, index=nodes.index, name=name)
        join_node_meta(label_series, overwrite=True)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%

dc_flat_labels = dc.predict(X, fcluster=True, level=7)

nodes["dc_flat_labels"] = dc_flat_labels
group_order = (
    nodes.groupby("dc_flat_labels")["sum_signal_flow"]
    .apply(np.median)
    .sort_values(ascending=False)
    .index
)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
crosstabplot(
    nodes[nodes["hemisphere"] == "L"],
    group="dc_flat_labels",
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
    group="dc_flat_labels",
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
stashfig("crosstabplot_divisive")


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")