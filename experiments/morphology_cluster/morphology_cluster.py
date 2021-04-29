# %% [markdown]
# ##
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

from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.pymaid import start_instance
from src.visualization import (
    plot_neurons,
    plot_single_dendrogram,
    plot_volumes,
    set_axes_equal,
    set_theme,
    CLASS_COLOR_DICT,
)
from src.data import load_palette

t0 = time.time()
# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


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

from src.data import load_maggot_graph

mg = load_maggot_graph()
meta = mg.nodes
meta = meta[meta["paper_clustered_neurons"]].copy()

if CLASS_KEY == "merge_class":
    palette = CLASS_COLOR_DICT
else:
    palette = load_palette()

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

#%%
# sorting for the clusters
median_cluster_order = meta.groupby(CLUSTER_KEY)[ORDER_KEY].apply(np.nanmedian)
meta["cluster_order"] = meta[CLUSTER_KEY].map(median_cluster_order)
meta = meta.sort_values(["cluster_order", CLUSTER_KEY], ascending=ORDER_ASCENDING)
uni_clusters = meta[CLUSTER_KEY].unique()  # preserves sorting from above
uni_clusters = uni_clusters[~np.isnan(uni_clusters)]

#%%
from graspologic.embed import AdjacencySpectralEmbed
from giskard.plot import simple_scatterplot
from src.data import load_navis_neurons
from navis import NeuronList
from graspologic.cluster import AutoGMMCluster

from src.visualization import simple_plot_neurons
from graspologic.plot import pairplot_with_gmm
from graspologic.utils import remap_labels

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
ase = AdjacencySpectralEmbed(check_lcc=False, diag_aug=True)

left_meta = meta.loc[left_intersect_index]
left_clustering = left_meta[CLUSTER_KEY]
left_sim = left_sim.reindex(index=left_meta.index, columns=left_meta.index)

#%%
for label in uni_clusters[50:80]:
    label = int(label)
    cluster_meta = left_meta[left_meta[CLUSTER_KEY] == label]
    # class_labels = cluster_meta[CLASS_KEY].values
    cluster_ids = cluster_meta.index
    within_sim = left_sim.loc[cluster_ids, cluster_ids].values
    if len(within_sim) > 6:
        morphology_embedding = ase.fit_transform(within_sim)
        # simple_scatterplot(morphology_embedding, labels=cluster_meta[CLASS_KEY].values)
        agmm = AutoGMMCluster(
            min_components=1, max_components=min(10, len(within_sim)), n_init=10
        )
        pred_labels = agmm.fit_predict(morphology_embedding)
        # remap_labels(class_labels, pred_labels)
        pairplot_with_gmm(morphology_embedding, agmm.model_, figsize=(10, 10))
        stashfig(f"gmm-pairplot-subcluster={label}")
        plot_neuron_list = neurons.idx[cluster_ids]
        plot_morphology_subclustering(plot_neuron_list, pred_labels)
        stashfig(f"morphology-subcluster={label}")

#%%
from giskard.plot import dissimilarity_clustermap
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

t = 0.1
criterion = "distance"
for label in uni_clusters[50:70]:
    label = int(label)
    cluster_meta = left_meta[left_meta[CLUSTER_KEY] == label]
    cluster_ids = cluster_meta.index
    within_sim = left_sim.loc[cluster_ids, cluster_ids].values
    if len(within_sim) > 3:
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
        cluster_dissimilarity = 1 - within_sim
        method = "average"
        Z = linkage(squareform(cluster_dissimilarity), method=method)
        flat_labels = fcluster(Z, t, criterion=criterion)
        plot_neuron_list = neurons.idx[cluster_ids]
        plot_morphology_subclustering(plot_neuron_list, flat_labels)
        stashfig(f"agglom-morphology-subcluster={label}")

#%%
t = 0.1
CLUSTER_KEY = "gt_blockmodel_labels"
median_cluster_order = meta.groupby(CLUSTER_KEY)[ORDER_KEY].apply(np.nanmedian)
meta["cluster_order"] = meta[CLUSTER_KEY].map(median_cluster_order)
meta = meta.sort_values(["cluster_order", CLUSTER_KEY], ascending=ORDER_ASCENDING)
uni_clusters = meta[CLUSTER_KEY].unique()  # preserves sorting from above
uni_clusters = uni_clusters[~np.isnan(uni_clusters)]

criterion = "distance"
for label in uni_clusters[10:20]:
    label = int(label)
    cluster_meta = left_meta[left_meta[CLUSTER_KEY] == label]
    cluster_ids = cluster_meta.index
    within_sim = left_sim.loc[cluster_ids, cluster_ids].values
    if len(within_sim) > 3:
        dissimilarity_clustermap(
            within_sim,
            invert=True,
            colors=cluster_meta[CLASS_KEY].values,
            palette=palette,
            method="average",
            cut=True,
            t=t,
        )
        stashfig(f"gt-morphology-dissimilarity-subcluster={label}")
        cluster_dissimilarity = 1 - within_sim
        method = "average"
        Z = linkage(squareform(cluster_dissimilarity), method=method)
        flat_labels = fcluster(Z, t, criterion=criterion)
        plot_neuron_list = neurons.idx[cluster_ids]
        plot_morphology_subclustering(plot_neuron_list, flat_labels)
        stashfig(f"gt-agglom-morphology-subcluster={label}")


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
