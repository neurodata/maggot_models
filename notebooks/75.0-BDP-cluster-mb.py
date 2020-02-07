# %% [markdown]
# # Imports
import os
import random
from operator import itemgetter
from pathlib import Path

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from graspy.utils import get_lcc
from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.data import load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.io import savefig, saveobj, saveskels
from src.visualization import (
    bartreeplot,
    get_color_dict,
    get_colors,
    remove_spines,
    sankey,
    screeplot,
)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True
SAVEFIGS = True
BRAIN_VERSION = "2020-01-29"

sns.set_context("talk")

base_path = Path("maggot_models/data/raw/Maggot-Brain-Connectome/")


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=SAVEFIGS, **kws)


threshold_raw = False
ptr = True
n_components = 3
if threshold_raw:
    graph_type = "G"
else:
    graph_type = "Gn"

remove_pdiff = True

mg = load_metagraph(graph_type, BRAIN_VERSION)
n_original_verts = len(mg)
if remove_pdiff:
    keep_inds = np.where(~mg["is_pdiff"])[0]
    mg = mg.reindex(keep_inds)
    print(f"Removed {n_original_verts - len(mg.meta)} partially differentiated")

mg.meta["Original index"] = range(len(mg.meta))
subset_classes = ["KC", "MBIN", "mPN", "tPN", "MBON", "uPN", "vPN", "mPN; FFN"]
subset_inds = mg.meta[mg.meta["Class 1"].isin(subset_classes)]["Original index"]
mg.reindex(subset_inds)
mg.meta["Original index"] = range(len(mg.meta))
subset_inds = mg.meta[mg.meta["Hemisphere"] == "L"]["Original index"]
mg.reindex(subset_inds)
mg.make_lcc()

# mg.verify(10000, graph_type=graph_type)

class_labels = mg["Class 1"].copy()
class_map = {
    "KC": "KC",
    "MBIN": "MBIN",
    "mPN": "PN",
    "tPN": "PN",
    "MBON": "MBON",
    "uPN": "PN",
    "vPN": "PN",
    "mPN; FFN": "PN",
}
class_labels = np.array(itemgetter(*class_labels)(class_map))


if threshold_raw:
    thresholds = np.linspace(0, 4, 5)
else:
    thresholds = np.linspace(0, 0.05, num=5)

rows = []
for threshold in thresholds:
    adj = mg.adj.copy()
    adj[adj <= threshold] = 0
    adj, inds = get_lcc(adj, return_inds=True)
    true_labels = class_labels[inds]
    latent = ase(adj, n_components, ptr=ptr)

    # cluster = GaussianCluster(
    #     min_components=2, max_components=10, covariance_type="all", n_init=100
    # )
    cluster = AutoGMMCluster(min_components=2, max_components=10)

    pred_labels = cluster.fit_predict(latent)

    ari = adjusted_rand_score(true_labels, pred_labels)

    row = {"ARI": ari, "Threshold": threshold}
    rows.append(row)

result_df = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.lineplot(data=result_df, x="Threshold", y="ARI", ax=ax)
remove_spines(ax, keep_corner=True)
ax.set_title(f"Mushroom Body, n_components={n_components}")
stashfig(f"mb-nc{n_components}-tr{threshold_raw}")
