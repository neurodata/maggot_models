#%%
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.cluster import GaussianCluster, KMeansCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator
from graspy.plot import degreeplot, edgeplot, gridplot, heatmap, pairplot
from graspy.utils import augment_diagonal, binarize, cartprod, pass_to_ranks, to_laplace
from joblib.parallel import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from spherecluster import SphericalKMeans

from src.data import load_everything, load_networkx
from src.models import GridSearchUS
from src.utils import get_best, meta_to_array, relabel, savefig
from src.visualization import incidence_plot, screeplot

# Global general parameters
MB_VERSION = "mb_2019-09-23"
BRAIN_VERSION = "2019-09-18-v2"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

# Set up plotting constants
palette = "deep"
plt.style.use("seaborn-white")
sns.set_palette(palette)
sns.set_context("talk", font_scale=1)

# Load data
adj, class_labels, side_labels = load_everything(
    "G", version=BRAIN_VERSION, return_class=True, return_side=True
)


degrees = adj.sum(axis=0) + adj.sum(axis=1)
sort_inds = np.argsort(degrees)[::-1]
class_labels = class_labels[sort_inds]
adj = adj[np.ix_(sort_inds, sort_inds)]

name_map = {
    "CN": "Unknown",
    "DANs": "DAN",
    "KCs": "KC",
    "LHN": "Unknown",
    "LHN; CN": "Unknown",
    "MBINs": "MBIN",
    "MBON": "MBON",
    "MBON; CN": "MBON",
    "OANs": "OAN",
    "ORN mPNs": "mPN",
    "ORN uPNs": "uPN",
    "tPNs": "tPN",
    "vPNs": "vPN",
    "Unidentified": "Unknown",
    "Other": "Unknown",
}
class_labels = np.array(itemgetter(*class_labels)(name_map))


# Now load all 4 colors
color_adjs = []
for t in GRAPH_TYPES:
    adj = load_everything(t)
    adj = adj[np.ix_(sort_inds, sort_inds)]
    color_adjs.append(adj)

sum_adj = np.array(color_adjs).sum(axis=0)


# Print some stats
n_verts = adj.shape[0]
print("Full Brain")
print()
print(f"Number of vertices: {n_verts}")
print()

name = "G"
g = sum_adj
print(name)
print(f"Number of edges: {np.count_nonzero(g)}")
print(f"Sparsity: {np.count_nonzero(g) / (n_verts**2)}")
print(f"Number of synapses: {int(g.sum())}")
median_in_degree = np.median(np.count_nonzero(g, axis=0))
median_out_degree = np.median(np.count_nonzero(g, axis=1))
print(f"Median node in degree: {median_in_degree}")
print(f"Median node out degree: {median_out_degree}")
print()

for g, name in zip(color_adjs, GRAPH_TYPES):
    print(name)
    print(f"Number of edges: {np.count_nonzero(g)}")
    print(f"Sparsity: {np.count_nonzero(g) / (n_verts**2)}")
    print(f"Number of synapses: {int(g.sum())}")
    median_in_degree = np.median(np.count_nonzero(g, axis=0))
    median_out_degree = np.median(np.count_nonzero(g, axis=1))
    print(f"Median node in degree: {median_in_degree}")
    print(f"Median node out degree: {median_out_degree}")
    print()


# Plot the adjacency matrix for the summed graph
sns.set_context("talk", font_scale=1)

plt.figure(figsize=(5, 5))
ax = heatmap(
    binarize(sum_adj),
    inner_hier_labels=class_labels,
    hier_label_fontsize=10,
    sort_nodes=False,
    cbar=False,
    title="Full Brain (summed 4 channels)",
    title_pad=90,
    font_scale=1.7,
)

# Plot the adjacency matrix for the 4-color graphs
fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax = ax.ravel()
for i, g in enumerate(color_adjs):
    heatmap(
        binarize(g),
        inner_hier_labels=class_labels,
        hier_label_fontsize=10,
        sort_nodes=False,
        ax=ax[i],
        cbar=False,
        title=GRAPH_TYPE_LABELS[i],
        title_pad=70,
        font_scale=1.7,
    )
plt.suptitle("Full Brain (4 channels)", fontsize=45, x=0.525, y=1.02)
plt.tight_layout()

#%% Distributions of edge weights and degrees

in_edgesum = sum_adj.sum(axis=0)
out_edgesum = sum_adj.sum(axis=1)
in_degree = np.count_nonzero(sum_adj, axis=0)
out_degree = np.count_nonzero(sum_adj, axis=1)

figsize = (10, 5)
sns.plotting_context("talk", font_scale=1.25)

plt.figure(figsize=figsize)
sns.distplot(in_edgesum)
plt.title("Full brain, G, in edgesums")
plt.xlabel("# inbound synapses")

plt.figure(figsize=figsize)
sns.distplot(out_edgesum)
plt.title("Full brain, G, out edgesums")
plt.xlabel("# outbound synapses")


plt.figure(figsize=figsize)
sns.distplot(in_degree)
plt.title("Full brain, G, in degrees")
plt.xlabel("# inbound edges")

plt.figure(figsize=figsize)
sns.distplot(out_degree)
plt.title("Full brain, G, out degrees")
plt.xlabel("# outbound edges")

# %% Now do the same by class

#%% Look at the heatmap of edgesums by class


def calc_edgesums(adjs, *args):
    deg_mat = np.zeros((n_verts, 2 * N_GRAPH_TYPES))
    for i, g in enumerate(adjs):
        deg_mat[:, i] = g.sum(axis=0)
        deg_mat[:, i + N_GRAPH_TYPES] = g.sum(axis=1)
    return deg_mat


edgesum_flat_mat = np.stack((in_edgesum, out_edgesum), axis=1)
edgesum_flat_df = pd.DataFrame(
    data=edgesum_flat_mat, columns=("In edgesum", "Out edgesum")
)
edgesum_flat_df["Class"] = class_labels
sns.jointplot(
    data=edgesum_flat_df,
    x="In edgesum",
    y="Out edgesum",
    kind="hex",
    height=10,
    color="#4CB391",
)
plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=edgesum_flat_df,
    x="In edgesum",
    y="Out edgesum",
    hue="Class",
    s=20,
    alpha=0.3,
    linewidth=0,
    palette=palette,
)


edgesum_mat = calc_edgesums(color_adjs)
in_cols = ["In " + n for n in GRAPH_TYPES]
out_cols = ["Out " + n for n in GRAPH_TYPES]
cols = np.array(in_cols + out_cols)

edgesum_df = pd.DataFrame(data=edgesum_mat, columns=cols)

figsize = (20, 20)
sns.clustermap(edgesum_df, figsize=figsize)
plt.title("Edgesum matrix, single linkage euclidean dendrograms", loc="center")


screeplot(edgesum_mat, cumulative=False, title="Edgesum matrix screeplot")
plt.ylim((0, 0.5))

pca = PCA(n_components=3)
edgesum_pcs = pca.fit_transform(edgesum_mat)

var_exp = np.sum(pca.explained_variance_ratio_)

pairplot(
    edgesum_pcs, height=5, alpha=0.3, title=f"Edgesum PCs, {var_exp} variance explained"
)
pairplot(
    edgesum_pcs,
    labels=class_labels,
    height=5,
    alpha=0.3,
    title="Edgesum PCs colored by known types",
    palette=palette,
)

pairplot(
    edgesum_mat[:, [0, 1, 4, 5]],
    labels=class_labels,
    height=5,
    alpha=0.3,
    title="Full edgesum matrix colored by known types",
    palette=palette,
    col_names=list(cols[[0, 1, 4, 5]]),
)

