#%%
import datetime
import time
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import crosstabplot
from graspologic.cluster import DivisiveCluster
from src.data import join_node_meta, load_maggot_graph, load_palette
from src.io import savefig
from src.visualization import set_theme

t0 = time.time()

set_theme()

palette = load_palette()

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/gaussian_cluster/figs",
        **kws,
    )


# out_dir = Path(__file__).parent / "outs"
embedding_method = "sym-ase"
if embedding_method == "mase":
    embedding_loc = Path("maggot_models/experiments/embed/outs/stage2_embedding.csv")
    embedding_df = pd.read_csv(embedding_loc, index_col=0)
    embedding_df = embedding_df.groupby(embedding_df.index).mean()
elif embedding_method == "sym-ase":
    embedding_loc = Path(
        "maggot_models/experiments/revamp_embed/outs/condensed_nodes.csv"
    )
    embedding_df = pd.read_csv(
        embedding_loc, index_col=0, converters=dict(skeleton_ids=literal_eval)
    )
    embedding_df = embedding_df.explode("skeleton_ids")
    embedding_df = embedding_df.set_index("skeleton_ids")
    embedding_df = embedding_df[[f"latent_{i}" for i in range(16)]]
elif embedding_method == "omni":
    pass
embedding_df
#%% load the embedding, get the correct subset of data

mg = load_maggot_graph()
mg = mg.node_subgraph(mg.nodes.query("has_embedding").index)
nodes = mg.nodes.copy()
print(len(nodes))

# #%%
# nodes.groupby("predicted_pair_id").size().sort_values()

#%%
nodes = nodes[nodes.index.isin(embedding_df.index)]
embedding_df = embedding_df[embedding_df.index.isin(nodes.index)]
nodes = nodes.reindex(embedding_df.index)
nodes["inds"] = range(len(nodes))
embedding = embedding_df.values

symmetrize_pairs = True
if symmetrize_pairs:
    pair_groups = nodes.groupby("pair_id")
    for pair_id, pair_group in pair_groups:
        if pair_id > 1 and len(pair_group) == 2:
            inds = pair_group["inds"].values
            pair_embeddings = embedding[inds]
            mean_embedding = pair_embeddings.mean(axis=0)
            embedding[inds[0]] = mean_embedding
            embedding[inds[1]] = mean_embedding


# %% [markdown]
# ## Clustering


# parameters
n_levels = 10  # max # of splits in the recursive clustering
metric = "bic"  # metric on which to decide best split
n_components = 8
X = embedding[:, :n_components]

# flat_labels = nodes["agglom_labels_t=0.65_n_components=64"].astype(int)
# covariance_type = "full"
# reg_covar = 1e-06
# onehot = _labels_to_onehot(flat_labels)
# weights_init, means_init, precisions_init = _onehot_to_initial_params(
#     X, onehot, covariance_type, reg_covar=reg_covar
# )
# gm = GaussianMixture(
#     n_components=len(weights_init),
#     covariance_type=covariance_type,
#     reg_covar=reg_covar,
#     weights_init=weights_init,
#     means_init=means_init,
#     precisions_init=precisions_init,
# )
# pred_labels = gm.fit_predict(X)

# stashfig("crosstabplot_gmm_o_agglom")


#%%
currtime = time.time()
n_components_range = [10]  # maybe try other n_components but seems fine
min_split_range = [32]
for n_components in n_components_range:
    for min_split in min_split_range:
        X = embedding[:, :n_components]
        dc = DivisiveCluster(
            cluster_kws=dict(kmeans_n_init=25), min_split=min_split, max_level=8
        )
        hier_labels = dc.fit_predict(X, fcluster=True) + 1
        cols = [
            f"dc_level_{i+1}_n_components={n_components}_min_split={min_split}"
            for i in range(hier_labels.shape[1])
        ]
        label_series = pd.DataFrame(data=hier_labels, index=nodes.index, columns=cols)
        label_series[
            f"dc_level_0_n_components={n_components}_min_split={min_split}"
        ] = np.zeros(len(label_series), dtype=int)
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
    hue="simple_group",
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
    hue="simple_group",
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
