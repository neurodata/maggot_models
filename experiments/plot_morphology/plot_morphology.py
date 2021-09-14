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

t0 = time.time()
# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/plot_morphology/")

CLASS_KEY = "simple_group"
ORDER_KEY = "sum_signal_flow"
# CLUSTER_KEY = "agglom_labels_t=0.625_n_components=64"
# CLUSTER_KEY = "gt_blockmodel_labels"
CLUSTER_KEY = "agg_labels_n_clusters=85"
CLUSTER_KEY = "co_cluster_n_clusters=85"
CLUSTER_KEY = "dc_level_7_n_components=10_min_split=32"
# CLUSTER_KEY = "agmm_agg_n_clusters=85"
ORDER_ASCENDING = False
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

if CLASS_KEY == "merge_class":
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
n_per_cluster = np.inf
show_metric = True

fig = plt.figure(figsize=(12 * 2, 8.5 * 2))
n_cols = 12
plot_mode = "3d"
volume_names = ["PS_Neuropil_manual"]

# plotting setup
n_rows = int(np.ceil(len(uni_clusters) / n_cols))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)
axs_flat = []
skeleton_color_dict = dict(zip(meta.index, np.vectorize(palette.get)(meta[CLASS_KEY])))
volumes = [pymaid.get_volume(v) for v in volume_names]
projection = None
if plot_mode == "3d":
    projection = "3d"


def get_neuron_ids(meta, level_key, cluster, n_per_cluster=np.inf):
    sub_meta = meta[meta[level_key] == cluster]
    neuron_ids = sub_meta.index
    # for possibly not plotting all neurons per cluster
    n_in_cluster = min(n_per_cluster, len(neuron_ids))
    plot_neuron_ids = np.random.choice(neuron_ids, size=n_in_cluster, replace=False)
    plot_neuron_ids = [int(neuron) for neuron in plot_neuron_ids]
    return plot_neuron_ids


neurons = load_navis_neurons()

#%%
x_lims_by_ax = []
z_lims_by_ax = []
for i, cluster in enumerate(uni_clusters[:]):
    print(f"{i / len(uni_clusters):0.2f}")
    plot_neuron_ids = get_neuron_ids(
        meta, CLUSTER_KEY, cluster, n_per_cluster=n_per_cluster
    )
    plot_neuron_list = neurons.idx[plot_neuron_ids]

    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds], projection=projection)
    axs[inds] = ax
    axs_flat.append(ax)

    simple_plot_neurons(
        plot_neuron_list,
        palette=skeleton_color_dict,
        ax=ax,
        azim=-90,
        elev=-90,
        dist=5,
        axes_equal=True,
        use_x=True,
        use_y=False,
        use_z=True,
    )

    x_lims_by_ax.append(ax.get_xlim3d())
    z_lims_by_ax.append(ax.get_zlim3d())

    if show_metric:
        val = mean_cluster_sim[cluster]
        if not np.isnan(val):
            ax.text2D(
                0.07,
                0.03,
                f"{val:.02f}",
                ha="left",
                va="bottom",
                color="black",
                fontsize="x-small",
                transform=ax.transAxes,
            )

stashfig(
    f"all-morpho-plot-clustering={CLUSTER_KEY}-discrim={show_metric}-wide",
    format="png",
)

# %%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
