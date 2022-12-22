# %%
import datetime
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymaid
import seaborn as sns
from navis import NeuronList
from src.data import load_maggot_graph, load_navis_neurons, load_palette
from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.nblast import preprocess_nblast
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    plot_neurons,
    plot_single_dendrogram,
    plot_volumes,
    set_axes_equal,
    set_theme,
    simple_plot_neurons,
)
from src.visualization import ORDER_KEY, HUE_KEY

t0 = time.time()


set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/plot_morphology/")


CLUSTER_KEY = "dc_level_7_n_components=10_min_split=32"

ORDER_ASCENDING = True
FORMAT = "png"


def stashfig(name, format=FORMAT, **kws):
    savefig(
        name, pathname=save_path / "figs", format=format, dpi=300, save_on=True, **kws
    )


start_instance()


# %% load data
mg = load_maggot_graph()
meta = mg.nodes
meta = meta[meta["paper_clustered_neurons"]].copy()

if HUE_KEY == "merge_class":
    palette = CLASS_COLOR_DICT
else:
    palette = load_palette()

#%% new
# load nblast scores/similarities
data_dir = Path("maggot_models/experiments/nblast/outs")

symmetrize_mode = "geom"
transform = "ptr"
nblast_type = "scores"

side = "left"
nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-{nblast_type}.csv", index_col=0)
nblast_sim.columns = nblast_sim.columns.values.astype(int)
print(f"{len(nblast_sim)} neurons in NBLAST data on {side}")
# get neurons that are in both
left_intersect_index = np.intersect1d(meta.index, nblast_sim.index)
print(f"{len(left_intersect_index)} neurons in intersection on {side}")
# reindex appropriately
nblast_sim = nblast_sim.reindex(
    index=left_intersect_index, columns=left_intersect_index
)
sim = preprocess_nblast(
    nblast_sim.values, symmetrize_mode=symmetrize_mode, transform=transform
)
left_sim = pd.DataFrame(data=sim, index=nblast_sim.index, columns=nblast_sim.index)

side = "right"
nblast_sim = pd.read_csv(data_dir / f"{side}-nblast-{nblast_type}.csv", index_col=0)
nblast_sim.columns = nblast_sim.columns.values.astype(int)
print(f"{len(nblast_sim)} neurons in NBLAST data on {side}")
# get neurons that are in both
right_intersect_index = np.intersect1d(meta.index, nblast_sim.index)
print(f"{len(right_intersect_index)} neurons in intersection on {side}")
# reindex appropriately
nblast_sim = nblast_sim.reindex(
    index=right_intersect_index, columns=right_intersect_index
)
sim = preprocess_nblast(
    nblast_sim.values, symmetrize_mode=symmetrize_mode, transform=transform
)
right_sim = pd.DataFrame(data=sim, index=nblast_sim.index, columns=nblast_sim.index)

#%%
# sorting for the clusters
median_cluster_order = meta.groupby(CLUSTER_KEY)[ORDER_KEY].apply(np.nanmedian)
meta["cluster_order"] = meta[CLUSTER_KEY].map(median_cluster_order)
meta = meta.sort_values(["cluster_order", CLUSTER_KEY], ascending=ORDER_ASCENDING)
uni_clusters = meta[CLUSTER_KEY].unique()  # preserves sorting from above
uni_clusters = uni_clusters[~np.isnan(uni_clusters)]

print(f"Number of clusters: {len(uni_clusters)}")

#%% new
left_meta = meta.loc[left_intersect_index]
left_clustering = left_meta[CLUSTER_KEY]
left_sim = left_sim.reindex(index=left_meta.index, columns=left_meta.index)
left_cluster_sim = {}
for label in uni_clusters:
    cluster_ids = left_meta[left_meta[CLUSTER_KEY] == label].index
    within_sim = left_sim.loc[cluster_ids, cluster_ids].values
    triu_inds = np.triu_indices_from(within_sim, k=1)
    upper = within_sim[triu_inds]
    mean_within_sim = np.mean(upper)
    left_cluster_sim[label] = mean_within_sim

right_meta = meta.loc[right_intersect_index]
right_clustering = right_meta[CLUSTER_KEY]
right_sim = right_sim.reindex(index=right_meta.index, columns=right_meta.index)
right_cluster_sim = {}
for label in uni_clusters:
    cluster_ids = right_meta[right_meta[CLUSTER_KEY] == label].index
    within_sim = right_sim.loc[cluster_ids, cluster_ids].values
    triu_inds = np.triu_indices_from(within_sim, k=1)
    upper = within_sim[triu_inds]
    mean_within_sim = np.mean(upper)
    right_cluster_sim[label] = mean_within_sim

mean_cluster_sim = {}
for cluster_label in uni_clusters:
    mean_cluster_sim[cluster_label] = (
        left_cluster_sim[cluster_label] + right_cluster_sim[cluster_label]
    ) / 2

#%%
from hyppo.discrim import DiscrimOneSample

rows = []
for level in range(7, 8):
    level_name = f"dc_level_{level}_n_components=10_min_split=32"
    labels = left_meta[level_name].values.astype(float)
    left_stat = DiscrimOneSample(is_dist=True)._statistic(1 - left_sim.values, labels)
    shuffle_labels = np.random.permutation(labels)
    fake_left_stat = DiscrimOneSample(is_dist=True)._statistic(1 - left_sim.values, shuffle_labels)
    print(left_stat)
    print(fake_left_stat)

    labels = right_meta[level_name].values.astype(float)
    right_stat = DiscrimOneSample(is_dist=True)._statistic(1 - right_sim.values, labels)

    mean_stat = (left_stat + right_stat) / 2

    rows.append({"level": level, "stat": mean_stat})

#%% 
results = pd.DataFrame(rows)

fig, ax = plt.subplots(1,1,figsize=(8,8))
sns.lineplot(data=results, x="level", y="stat", ax=ax)

#%%

from src.visualization import adjplot

adjplot(left_sim.values, meta=left_meta, sort_class=[CLUSTER_KEY])