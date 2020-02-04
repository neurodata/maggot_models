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


graph_type = "G"
remove_pdiff = True
input_thresh = 100

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


latent = ase(mg.adj, 3, ptr=True)

cluster = GaussianCluster(
    min_components=2, max_components=10, covariance_type="all", n_init=100
)

pred_labels = cluster.fit_predict(latent)
true_labels = mg["Class 1"]
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
true_labels = np.array(itemgetter(*true_labels)(class_map))

pairplot(latent, labels=pred_labels)
pairplot(latent, labels=true_labels)

print(adjusted_rand_score(true_labels, pred_labels))
