# %% [markdown]
# # Imports
import os
import pickle
import warnings
from operator import itemgetter
from pathlib import Path
from timeit import default_timer as timer

import colorcet as cc
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable

from spherecluster import VonMisesFisherMixture
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything
from src.embed import lse, preprocess_graph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob
from src.visualization import bartreeplot, get_color_dict, get_colors, sankey, screeplot

warnings.simplefilter("ignore", category=FutureWarning)


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)


# %% [markdown]
# # Parameters
BRAIN_VERSION = "2019-12-18"


adj, class_labels, side_labels, pair_labels, skeleton_labels, = load_everything(
    "Gadn",
    version=BRAIN_VERSION,
    return_keys=["Merge Class", "Hemisphere", "Pair"],
    return_ids=True,
)

row_sums = np.sum(adj, axis=1)
adj = adj / row_sums[:, np.newaxis]
adj[np.isnan(adj)] = 0
row_sums = np.sum(adj, axis=1)
U, S, V = np.linalg.svd(adj)


sensory_classes = ["mPN-m", "mPN-o", "mPN;FFN-m", "tPN", "uPN", "vPN"]
sensory_map = {
    "mPN-m": "multimodal",
    "mPN-o": "olfaction",
    "mPN;FFN-m": "multimodal",
    "tPN": "thermo",
    "uPN": "olfaction",
    "vPN": "visual",
}
is_sensory = np.vectorize(lambda s: s in sensory_classes)(class_labels)
inds = np.arange(len(class_labels))
sensory_inds = inds[is_sensory]
nonsensory_inds = inds[~is_sensory]
sensory_labels = class_labels[is_sensory]
sensory_labels = np.array(itemgetter(*sensory_labels)(sensory_map))

n_timesteps = 5
n_verts = adj.shape[0]
sensory_mats = []
for i in sensory_inds:
    print(i)
    prob_mats = np.empty((n_verts, n_timesteps))
    for t in range(n_timesteps):
        state_vec = np.zeros((n_verts, 1))
        state_vec[i] = 1
        prob_vec = U @ (S ** t) @ V @ state_vec
        prob_mats[:, t] = prob_vec
