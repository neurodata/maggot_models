# %% [markdown]
# # Imports
import json
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from joblib.parallel import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import adjusted_rand_score, silhouette_score
from spherecluster import SphericalKMeans

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks
from src.cluster import DivisiveCluster
from src.data import load_everything
from src.embed import lse
from src.hierarchy import signal_flow
from src.io import stashfig
from src.utils import export_skeleton_json, savefig
from src.visualization import clustergram, palplot, sankey, stacked_barplot

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2019-12-09"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

SAVEFIGS = True
SAVESKELS = True
SAVEOBJS = True

MIN_CLUSTERS = 2
MAX_CLUSTERS = 3
N_INIT = 50
PTR = True
ONLY_RIGHT = False

embed = "LSE"
cluster = "AutoGMM"
n_components = 4
if cluster == "GMM":
    gmm_params = {"n_init": N_INIT, "covariance_type": "all"}
elif cluster == "AutoGMM":
    gmm_params = {"max_agglom_size": None}
elif cluster == "SKMeans":
    gmm_params = {"n_init": N_INIT}

np.random.seed(23409857)


def stashskel(name, ids, colors, palette=None, **kws):
    if SAVESKELS:
        return export_skeleton_json(
            name, ids, colors, palette=palette, foldername=FNAME, **kws
        )


def stashobj(obj, name, **kws):
    foldername = FNAME
    subfoldername = "objs"
    pathname = "./maggot_models/notebooks/outs"
    if SAVEOBJS:
        path = Path(pathname)
        if foldername is not None:
            path = path / foldername
            if not os.path.isdir(path):
                os.mkdir(path)
            if subfoldername is not None:
                path = path / subfoldername
                if not os.path.isdir(path):
                    os.mkdir(path)
        with open(path / str(name + ".pickle"), "wb") as f:
            pickle.dump(obj, f)


def sub_ari(known_inds, true_labels, pred_labels):
    true_known_labels = true_labels[known_inds]
    pred_known_labels = pred_labels[known_inds]
    ari = adjusted_rand_score(true_known_labels, pred_known_labels)
    return ari


# Set up plotting constants
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)


# %% [markdown]
# # Load the data


adj, class_labels, side_labels, skeleton_labels = load_everything(
    "Gad",
    version=BRAIN_VERSION,
    return_keys=["Merge Class", "Hemisphere"],
    return_ids=True,
)


# select the right hemisphere
if ONLY_RIGHT:
    side = "right hemisphere"
    right_inds = np.where(side_labels == "R")[0]
    adj = adj[np.ix_(right_inds, right_inds)]
    class_labels = class_labels[right_inds]
    skeleton_labels = skeleton_labels[right_inds]
else:
    side = "full brain"

# sort by number of synapses
degrees = adj.sum(axis=0) + adj.sum(axis=1)
sort_inds = np.argsort(degrees)[::-1]
adj = adj[np.ix_(sort_inds, sort_inds)]
class_labels = class_labels[sort_inds]
skeleton_labels = skeleton_labels[sort_inds]

# remove disconnected nodes
adj, lcc_inds = get_lcc(adj, return_inds=True)
class_labels = class_labels[lcc_inds]
skeleton_labels = skeleton_labels[lcc_inds]

# remove pendants
degrees = np.count_nonzero(adj, axis=0) + np.count_nonzero(adj, axis=1)
not_pendant_mask = degrees != 1
not_pendant_inds = np.array(range(len(degrees)))[not_pendant_mask]
adj = adj[np.ix_(not_pendant_inds, not_pendant_inds)]
class_labels = class_labels[not_pendant_inds]
skeleton_labels = skeleton_labels[not_pendant_inds]

# plot degree sequence
d_sort = np.argsort(degrees)[::-1]
degrees = degrees[d_sort]
plt.figure(figsize=(10, 5))
sns.scatterplot(x=range(len(degrees)), y=degrees, s=30, linewidth=0)

known_inds = np.where(class_labels != "Unk")[0]

# %% [markdown]
# #
n_verts = adj.shape[0]
latent = lse(adj, n_components, regularizer=None, ptr=PTR)
pairplot(latent, labels=class_labels, title=embed)

# %% [markdown]
# #

dc = DivisiveCluster(n_init=200)
dc.fit(latent)
dc.print_tree()
linkage, labels = dc.build_linkage()
pred_labels = dc.predict(latent)

# %% [markdown]
# #


fig = plt.figure(figsize=(12, 12))
gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[0.1, 0.9], wspace=0)
ax0 = fig.add_subplot(gs[0])

dendr_data = dendrogram(
    linkage,
    orientation="left",
    labels=labels,
    color_threshold=0,
    above_threshold_color="k",
    ax=ax0,
)
ax0.axis("off")

ticks = ax0.get_yticks()

leaf_names = np.array(dendr_data["ivl"])[::-1]
ax1 = fig.add_subplot(gs[1], sharey=ax0)
ax1 = stacked_barplot(
    pred_labels,
    class_labels,
    label_pos=ticks,
    category_order=leaf_names,
    ax=ax1,
    bar_height=5,
    horizontal_pad=0,
    palette="tab20",
    norm_bar_width=True,
)
ax1.set_frame_on(False)
ax1.yaxis.tick_right()
plt.title(
    r"Divisive hierarchical clustering, GraspyGMM, LSE, PTR, Full Brain, A $\to$ D"
)
stashfig("hierarchy-bars")
