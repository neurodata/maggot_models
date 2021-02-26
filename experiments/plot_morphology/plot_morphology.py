# %% [markdown]
# ##

import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymaid
import seaborn as sns

from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    CLASS_COLOR_DICT,
    plot_neurons,
    plot_single_dendrogram,
    plot_volumes,
    set_axes_equal,
    set_theme,
)

currtime = time.time()

# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/plot_morphology/")

CLASS_KEY = "merge_class"
CLASS_ORDER = "median_node_visits"
FORMAT = "png"


def stashfig(name, format=FORMAT, **kws):
    savefig(
        name, pathname=save_path / "figs", format=format, dpi=300, save_on=True, **kws
    )


start_instance()


# %% load data
omni_method = "color_iso"
d = 8
bic_ratio = 0.95
min_split = 32

basename = f"-method={omni_method}-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"
meta = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/meta{basename}.csv",
    index_col=0,
)
meta["lvl0_labels"] = meta["lvl0_labels"].astype(str)
adj_df = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/adj{basename}.csv",
    index_col=0,
)
adj = adj_df.values

name_map = {
    "Sens": "Sensory",
    "LN": "Local",
    "PN": "Projection",
    "KC": "Kenyon cell",
    "LHN": "Lateral horn",
    "MBIN": "MBIN",
    "Sens2o": "2nd order sensory",
    "unk": "Unknown",
    "MBON": "MBON",
    "FBN": "MB feedback",
    "CN": "Convergence",
    "PreO": "Pre-output",
    "Outs": "Output",
    "Motr": "Motor",
}
meta["simple_class"] = meta["simple_class"].map(name_map)
print(meta["simple_class"].unique())
meta["merge_class"] = meta["simple_class"]  # HACK


graph_type = "Gad"
n_init = 256
max_hops = 16
allow_loops = False
include_reverse = False
walk_spec = f"gt={graph_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}"
walk_meta = pd.read_csv(
    f"maggot_models/experiments/walk_sort/outs/meta_w_order-{walk_spec}-include_reverse={include_reverse}.csv",
    index_col=0,
)
meta["median_node_visits"] = walk_meta["median_node_visits"]  # make the sorting right

#%% new
# load nblast scores/similarities
from src.nblast import preprocess_nblast

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
level = 7
level_key = f"lvl{level}_labels"

# sorting for the clusters
median_group_visits = meta.groupby(level_key)["median_node_visits"].apply(np.nanmedian)
meta[f"lvl{level}_group_visits"] = meta[f"lvl{level}_labels"].map(median_group_visits)
meta = meta.sort_values([f"lvl{level}_group_visits", level_key], ascending=True)
uni_clusters = meta[level_key].unique()  # preserves sorting from above

#%% new

from giskard.stats import calc_discriminability_statistic

left_meta = meta.loc[left_intersect_index]
left_clustering = left_meta[level_key].values
left_sim = left_sim.reindex(index=left_meta.index, columns=left_meta.index)
left_total_discrim, left_cluster_discrim = calc_discriminability_statistic(
    1 - left_sim.values, left_clustering
)

right_meta = meta.loc[right_intersect_index]
right_clustering = right_meta[level_key].values
right_sim = right_sim.reindex(index=right_meta.index, columns=right_meta.index)
right_total_discrim, right_cluster_discrim = calc_discriminability_statistic(
    1 - right_sim.values, right_clustering
)

mean_cluster_discrim = {}
for cluster_label in uni_clusters:
    mean_cluster_discrim[cluster_label] = (
        left_cluster_discrim[cluster_label] + right_cluster_discrim[cluster_label]
    ) / 2

#%%
n_per_cluster = np.inf
show_discrim = True
# fig = plt.figure(figsize=(8.5 * 2, 11 * 2))
# n_cols = 8
fig = plt.figure(figsize=(11 * 2, 8.5 * 2))
n_cols = 11
plot_mode = "3d"
volume_names = ["PS_Neuropil_manual"]

from src.visualization import simple_plot_neurons

# plotting setup
n_rows = int(np.ceil(len(uni_clusters) / n_cols))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
axs = np.empty((n_rows, n_cols), dtype=object)
axs_flat = []
skeleton_color_dict = dict(
    zip(meta.index, np.vectorize(CLASS_COLOR_DICT.get)(meta["merge_class"]))
)
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


x_lims_by_ax = []
z_lims_by_ax = []
for i, cluster in enumerate(uni_clusters[:]):
    print(f"{i / len(uni_clusters):0.2f}")
    plot_neuron_ids = get_neuron_ids(
        meta, level_key, cluster, n_per_cluster=n_per_cluster
    )
    inds = np.unravel_index(i, shape=(n_rows, n_cols))

    ax = fig.add_subplot(gs[inds], projection=projection)
    axs[inds] = ax
    axs_flat.append(ax)

    # pymaid.plot2d(
    #     plot_neuron_ids,
    #     color=skeleton_color_dict,
    #     ax=ax,
    #     connectors=False,
    #     method=plot_mode,
    #     autoscale=False,
    # )
    # plot_volumes(volumes, ax)

    simple_plot_neurons(
        plot_neuron_ids,
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

    # if plot_mode == "3d":
    #     ax.azim = -90
    #     ax.elev = 0
    #     ax.dist = 5
    #     set_axes_equal(ax, use_y=False)

    x_lims_by_ax.append(ax.get_xlim3d())
    z_lims_by_ax.append(ax.get_zlim3d())

    if show_discrim:
        ax.text2D(
            0.07,
            0.03,
            f"{mean_cluster_discrim[cluster]:.02f}",
            ha="left",
            va="bottom",
            color="black",
            fontsize="x-small",
            transform=ax.transAxes,
        )

    ax.set_xlim3d((-4500, 110000))
    ax.set_ylim3d((-4500, 110000))

# HACK: have had the weirdest bug where the first axis (regardless of which cluster it
# is) always has different limits from the rest. So this hack just removes that entire
# axis and replaces it with the first cluster again
axs[0, 0].remove()
ax = fig.add_subplot(gs[0, 0], projection=projection)
axs_flat[0] = ax
plot_neuron_ids = get_neuron_ids(
    meta, level_key, uni_clusters[0], n_per_cluster=n_per_cluster
)
simple_plot_neurons(
    plot_neuron_ids,
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

if show_discrim:
    ax.text2D(
        0.07,
        0.03,
        f"{mean_cluster_discrim[cluster]:.02f}",
        ha="left",
        va="bottom",
        color="black",
        transform=ax.transAxes,
        fontsize="x-small",
    )

ax.set_xlim3d((-4500, 110000))
ax.set_ylim3d((-4500, 110000))

# # make limits for all plots the same
# # TODO this could actually make the axes slightly not equal, currently
# x_lims_by_ax = np.array(x_lims_by_ax)
# z_lims_by_ax = np.array(z_lims_by_ax)
# x_min = np.min(x_lims_by_ax[:, 0])
# x_max = np.max(x_lims_by_ax[:, 1])
# z_min = np.min(z_lims_by_ax[:, 0])
# z_max = np.max(z_lims_by_ax[:, 1])
# for ax in axs_flat[::-1]:
#     ax.set_xlim3d([x_min, x_max])
#     ax.set_zlim3d([z_min, z_max])

# plt.tight_layout()
stashfig(f"all-morpho-plot-level={level}-discrim={show_discrim}-wide", format="png")

print(f"{time.time() - currtime:.3f} elapsed for whole script.")