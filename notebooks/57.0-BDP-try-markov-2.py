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
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import symmetrize
from src.cluster import DivisiveCluster
from src.data import load_everything, load_metagraph
from src.embed import ase, lse, preprocess_graph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savefig, saveobj, saveskels
from src.utils import get_blockmodel_df, get_sbm_prob, invert_permutation
from src.visualization import bartreeplot, get_color_dict, get_colors, sankey, screeplot


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

print(nx.__version__)

# %% [markdown]
# #
BRAIN_VERSION = "2019-01-16"
norm_mg = load_metagraph("Gad")
norm_mg.make_lcc()
adj = norm_mg.adj
class_labels = norm_mg["Merge Class"]

trans_mat = adj.copy()
row_sums = np.sum(adj, axis=1)
trans_mat = trans_mat / row_sums[:, np.newaxis]
trans_mat[np.isnan(trans_mat)] = 0


# %% [markdown]
# #
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
        prob_vec = 1 @ state_vec
        prob_mats[:, t] = prob_vec

# %% [markdown]
# #
state_vec = np.zeros((n_verts, 1))
state_vec[i] = 1
plt.figure()
plt.plot(state_vec)

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec

new_state_vec = adj @ state_vec
plt.figure()
plt.plot(new_state_vec)
state_vec = new_state_vec


# %% [markdown]
# #

sensory_classes = ["mPN-m", "mPN-o", "mPN;FFN-m", "tPN", "uPN", "vPN"]
sensory_map = {
    "mPN-m": "multimodal",
    "mPN-o": "olfaction",
    "mPN;FFN-m": "multimodal",
    "tPN": "thermo",
    "uPN": "olfaction",
    "vPN": "visual",
}

sensory_mods = ["multimodal", "olfaction", "thermo", "visual"]


def filter(s):
    if s in sensory_map:
        return sensory_map[s]
    else:
        return ""


sensory_labels = np.vectorize(filter)(class_labels)
n_verts = len(sensory_labels)
n_timesteps = 5

response_mats = []
for s in sensory_mods:
    response_mat = np.zeros((n_verts, n_timesteps))
    inds = np.where(sensory_labels == s)[0]
    state_vec = np.zeros(len(sensory_labels))
    state_vec[inds] = 1 / len(inds)
    for t in range(n_timesteps):
        new_state_vec = trans_mat @ state_vec
        response_mat[:, t] = new_state_vec
        state_vec = new_state_vec
    response_mats.append(response_mat)

full_response_mat = np.concatenate(response_mats, axis=-1)

for i in range(full_response_mat.shape[1]):
    plt.figure()
    plt.plot(full_response_mat[:, i])

# %% [markdown]
# #
from sklearn.decomposition import PCA

in_deg = adj.sum(axis=0)
in_deg[in_deg == 0] = 1
full_response_mat = full_response_mat / in_deg[:, np.newaxis]
full_response_mat[np.isinf(full_response_mat)] = 0
pca = PCA(n_components=4)
latent = pca.fit_transform(full_response_mat)
pairplot(latent, labels=class_labels)

# %% [markdown]
# #
is_sensory = np.vectorize(lambda s: s in sensory_classes)(class_labels)
inds = np.arange(len(class_labels))
sensory_inds = inds[is_sensory]

response_mats = []
n_timesteps = 1
for s in sensory_inds:
    response_mat = np.zeros((n_verts, n_timesteps))
    state_vec = np.zeros(len(sensory_labels))
    state_vec[s] = 1
    for t in range(n_timesteps):
        new_state_vec = trans_mat @ state_vec
        response_mat[:, t] = new_state_vec
        state_vec = new_state_vec
    response_mats.append(response_mat)

full_response_mat = np.concatenate(response_mats, axis=-1)

for i in range(full_response_mat.shape[1]):
    plt.figure()
    plt.plot(full_response_mat[:, i])

