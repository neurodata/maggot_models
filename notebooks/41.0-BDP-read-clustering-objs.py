# %% [markdown]
# #
import json
import os
import warnings
from operator import itemgetter
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from joblib.parallel import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
import networkx as nx

from graspy.cluster import GaussianCluster, AutoGMMCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks
from src.data import load_everything
from src.utils import export_skeleton_json, savefig
from src.visualization import clustergram, palplot, sankey
from src.hierarchy import signal_flow

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

BRAIN_VERSION = "2019-12-09"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

BRAIN_VERSION = "2019-12-09"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

SAVEFIGS = True
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

SAVESKELS = False
SAVEOBJS = True

MIN_CLUSTERS = 2
MAX_CLUSTERS = 40
N_INIT = 50
PTR = True
ONLY_RIGHT = True


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


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

file_loc = "maggot_models/notebooks/outs/39.2-BDP-unbiased-clustering/objs/gmm-k18-AutoGMM-LSE-right-ad-PTR-raw.pickle"
gmm = pickle.load(open(file_loc, "rb"))


# # %% [markdown]
# # #
node_signal_flow = signal_flow(adj)
mean_sf = np.zeros(k)
for i in np.unique(pred_labels):
    inds = np.where(pred_labels == i)[0]
    mean_sf[i] = np.mean(node_signal_flow[inds])

cluster_mean_latent = gmm.model_.means_[:, 0]
block_probs = SBMEstimator().fit(bin_adj, y=pred_labels).block_p_
block_prob_df = pd.DataFrame(data=block_probs, index=range(k), columns=range(k))
block_g = nx.from_pandas_adjacency(block_prob_df, create_using=nx.DiGraph)
plt.figure(figsize=(10, 10))
# don't ever let em tell you you're too pythonic
pos = dict(zip(range(k), zip(cluster_mean_latent, mean_sf)))
# nx.draw_networkx_nodes(block_g, pos=pos)
labels = nx.get_edge_attributes(block_g, "weight")
# nx.draw_networkx_edge_labels(block_g, pos, edge_labels=labels)
from matplotlib.cm import ScalarMappable
import matplotlib as mpl

norm = mpl.colors.LogNorm(vmin=0.01, vmax=0.1)

sm = ScalarMappable(cmap="Reds", norm=norm)
cmap = sm.to_rgba(np.array(list(labels.values())) + 0.01)
nx.draw_networkx(
    block_g,
    pos,
    edge_cmap="Reds",
    edge_color=cmap,
    connectionstyle="arc3,rad=0.2",
    width=1.5,
)

