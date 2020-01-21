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
BRAIN_VERSION = "2020-01-16"
mg = load_metagraph("G", BRAIN_VERSION)
mg.make_lcc()
mg.sort_values("Merge Class")
adj = mg.adj
class_labels = mg["Class 1"]

trans_mat = adj.copy()
row_sums = np.sum(adj, axis=1)
trans_mat = trans_mat / row_sums[:, np.newaxis]
trans_mat[np.isnan(trans_mat)] = 0

trans_mat.sum(axis=1)

# %% [markdown]
# #
U, S, V = np.linalg.svd(adj)


is_sensory = class_labels == "sens"
inds = np.arange(len(class_labels))
sensory_inds = inds[is_sensory]

n_timesteps = 10
n_verts = adj.shape[0]
sensory_mats = []
for i in sensory_inds:
    prob_mat = np.zeros((n_verts, n_timesteps))
    state_vec = np.zeros((1, n_verts))
    state_vec[0, i] = 1
    for t in range(n_timesteps):
        new_state_vec = state_vec @ trans_mat
        prob_mat[:, t] = np.squeeze(new_state_vec)
        state_vec = new_state_vec
    sensory_mats.append(prob_mat)

# %% [markdown]
# #
from src.visualization import remove_spines

plot_df = pd.DataFrame(data=sensory_mats[0], columns=list(range(n_timesteps)))
# plot_df["Merge Class"] = mg.meta["Merge Class"]
# plot_df["ind"] = range(len(plot_df))
i = 21
fig, ax = plt.subplots(n_timesteps, 1, figsize=(5, 10))
for t in range(n_timesteps):
    ax[t].plot(sensory_mats[i][:, t])
    remove_spines(ax[t])
plt.suptitle(mg.meta.iloc[sensory_inds[i], 0] + " " + mg.meta.iloc[sensory_inds[i], 10])


# %% [markdown]
# #
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = np.array(sensory_mats)
X = X.reshape((len(sensory_inds), -1))
pcs = pca.fit_transform(X)
df = pd.DataFrame(data=pcs)
df["class"] = mg.meta.loc[is_sensory, "Merge Class"]
sns.scatterplot(data=df, x=0, y=1, hue="class")


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

