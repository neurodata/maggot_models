# %% [markdown]
# ##
import datetime
import os
import time
from pathlib import Path
from tqdm import tqdm
from src.data import join_node_meta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymaid
import seaborn as sns
from giskard.plot import dissimilarity_clustermap, simple_scatterplot
from graspologic.cluster import AutoGMMCluster
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.plot import pairplot_with_gmm
from graspologic.utils import remap_labels
from navis import NeuronList
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
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

set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/morphology_cluster/")

CLASS_KEY = "simple_group"
ORDER_KEY = "sum_signal_flow"
CLUSTER_KEY = "cluster_agglom_K=80"
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

#%%
# sorting for the clusters
median_cluster_order = meta.groupby(CLUSTER_KEY)[ORDER_KEY].apply(np.nanmedian)
meta["cluster_order"] = meta[CLUSTER_KEY].map(median_cluster_order)
meta = meta.sort_values(["cluster_order", CLUSTER_KEY], ascending=ORDER_ASCENDING)
uni_clusters = meta[CLUSTER_KEY].unique()  # preserves sorting from above
uni_clusters = uni_clusters[~np.isnan(uni_clusters)]

#%%

skeleton_color_dict = dict(zip(meta.index, np.vectorize(palette.get)(meta[CLASS_KEY])))


def plot_morphology_subclustering(neurons, labels, n_cols=4, scale=2.5):
    plot_neurons_kws = dict(
        palette=skeleton_color_dict,
        azim=-90,
        elev=-90,
        dist=5,
        axes_equal=True,
        use_x=True,
        use_y=False,
        use_z=True,
        axis_off=False,
    )
    text_kws = dict(ha="center", va="bottom", color="black", fontsize="x-small")

    uni_labels = np.unique(labels)
    n_clusters = len(uni_labels) + 1
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig = plt.figure(figsize=(scale * n_cols, scale * n_rows))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0)
    axs = np.empty((n_rows, n_cols), dtype=object)

    def add_axis(i):
        inds = np.unravel_index(i, shape=(n_rows, n_cols))
        ax = fig.add_subplot(gs[inds], projection="3d")
        axs[inds] = ax
        return ax

    i = 0
    plot_neuron_list = neurons
    ax = add_axis(i)
    simple_plot_neurons(plot_neuron_list, ax=ax, **plot_neurons_kws)
    ax.text2D(0.5, 0.95, "Whole cluster", transform=ax.transAxes, **text_kws)
    for i, label in enumerate(uni_labels):
        i += 1
        plot_neuron_list = neurons[labels == label]
        ax = add_axis(i)
        simple_plot_neurons(plot_neuron_list, ax=ax, **plot_neurons_kws)
        ax.text2D(
            0.5, 0.95, f"Subcluster {int(label)}", transform=ax.transAxes, **text_kws
        )


neurons = load_navis_neurons()

left_meta = meta.loc[left_intersect_index].copy()
left_clustering = left_meta[CLUSTER_KEY]
left_sim = left_sim.reindex(index=left_meta.index, columns=left_meta.index)

#%%
t = 0.1
criterion = "distance"
plot_neurons = True
plot_clustermap = False
for i, label in tqdm(enumerate(uni_clusters)):
    label = int(label)
    cluster_meta = left_meta[left_meta[CLUSTER_KEY] == label]
    cluster_ids = cluster_meta.index
    within_sim = left_sim.loc[cluster_ids, cluster_ids].values
    if len(within_sim) > 1:
        cluster_dissimilarity = 1 - within_sim
        method = "average"
        Z = linkage(squareform(cluster_dissimilarity), method=method)
        flat_labels = fcluster(Z, t, criterion=criterion)
        left_meta.loc[cluster_ids, "morphology_subcluster"] = flat_labels
        if plot_clustermap:
            dissimilarity_clustermap(
                within_sim,
                invert=True,
                colors=cluster_meta[CLASS_KEY].values,
                palette=palette,
                method="average",
                cut=True,
                t=t,
            )
            stashfig(f"morphology-dissimilarity-subcluster={label}")
        if plot_neurons:
            plot_neuron_list = neurons.idx[cluster_ids]
            plot_morphology_subclustering(plot_neuron_list, flat_labels)
            stashfig(
                f"agglom-morphologyt={t}-subcluster={label}-cluster_key={CLUSTER_KEY}"
            )

join_node_meta(left_meta[f"{CLUSTER_KEY}_morphology_subcluster"], overwrite=True)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
