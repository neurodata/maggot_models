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
from src.data import load_palette
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

t0 = time.time()
# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/morphology_cluster/")

CLASS_KEY = "merge_class"
ORDER_KEY = "sum_signal_flow"
CLUSTER_KEY = "agglom_labels_t=0.625_n_components=64"
ORDER_ASCENDING = False
FORMAT = "png"


def stashfig(name, format=FORMAT, **kws):
    savefig(
        name, pathname=save_path / "figs", format=format, dpi=300, save_on=True, **kws
    )


start_instance()


# %% load data

from graspologic.utils import multigraph_lcc_intersection
from src.data import load_maggot_graph

mg = load_maggot_graph()
meta = mg.nodes
mg = mg[mg.nodes["paper_clustered_neurons"]]

if CLASS_KEY == "merge_class":
    palette = CLASS_COLOR_DICT
else:
    palette = load_palette()

ll_mg, rr_mg, lr_mg, rl_mg = mg.bisect(paired=True)
ll_adj = ll_mg.sum.adj.copy()
rr_adj = rr_mg.sum.adj.copy()

adjs, lcc_inds = multigraph_lcc_intersection([ll_adj, rr_adj], return_inds=True)
ll_adj = adjs[0]
rr_adj = adjs[1]
print(f"{len(lcc_inds)} in intersection of largest connected components.")
from giskard.graph import MaggotGraph

left_meta = ll_mg.nodes.iloc[lcc_inds]
right_meta = rr_mg.nodes.iloc[lcc_inds]
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
left_intersect_index = np.intersect1d(left_meta.index, nblast_sim.index)
print(f"{len(left_intersect_index)} neurons in intersection on {side}")
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
right_intersect_index = np.intersect1d(right_meta.index, nblast_sim.index)
print(f"{len(right_intersect_index)} neurons in intersection on {side}")
nblast_sim = nblast_sim.reindex(
    index=right_intersect_index, columns=right_intersect_index
)
sim = preprocess_nblast(
    nblast_sim.values, symmetrize_mode=symmetrize_mode, transform=transform
)
right_sim = pd.DataFrame(data=sim, index=nblast_sim.index, columns=nblast_sim.index)

left_meta["_inds"] = range(len(left_meta))
right_meta["_inds"] = range(len(right_meta))
left_meta = left_meta.loc[left_intersect_index]
right_meta = right_meta.loc[right_intersect_index]
ll_adj = ll_adj[np.ix_(left_meta["_inds"].values, left_meta["_inds"].values)]
rr_adj = rr_adj[np.ix_(right_meta["_inds"].values, right_meta["_inds"].values)]
#%%
from giskard.plot import dissimilarity_clustermap

dissimilarity_clustermap(
    right_sim.values,
    colors=right_meta[CLASS_KEY].values,
    invert=True,
    palette=palette,
    method="average",
)
#%%

from giskard.plot import simple_umap_scatterplot
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspologic.plot import pairplot

ase = LaplacianSpectralEmbed(n_components=32, form="R-DAD")
morpho_X = ase.fit_transform(right_sim.values)
simple_umap_scatterplot(morpho_X, labels=right_meta[CLASS_KEY].values, palette=palette)

#%%
pg = pairplot(morpho_X[:, :4], labels=right_meta[CLASS_KEY].values, palette=palette)
pg._legend.remove()

#%%
from graspologic.utils import pass_to_ranks

ase = AdjacencySpectralEmbed(n_components=32, check_lcc=False, concat=True)
connecto_X = ase.fit_transform(pass_to_ranks(rr_adj))

#%%
pg = pairplot(
    connecto_X[:, :4],
    labels=right_meta[CLASS_KEY].values,
    palette=palette,
    diag_kind="hist",
)
pg._legend.remove()

#%%

#%%
# sorting for the clusters
median_cluster_order = meta.groupby(CLUSTER_KEY)[ORDER_KEY].apply(np.nanmedian)
meta["cluster_order"] = meta[CLUSTER_KEY].map(median_cluster_order)
meta = meta.sort_values(["cluster_order", CLUSTER_KEY], ascending=ORDER_ASCENDING)
uni_clusters = meta[CLUSTER_KEY].unique()  # preserves sorting from above
uni_clusters = uni_clusters[~np.isnan(uni_clusters)]
