# %% [markdown]
#  # EDA on full brain Drosophila larva connectome
# %% [markdown]
#  ## Definitions
#
#  **G**: the raw-weight graph for the full graph.
#
#  **Gaa**: the raw-weight graph for axo-axonic synapses.
#
#  **Gad**: the raw-weight graph for axo-dendritic synapses.
#
#  **Gda**: the raw-weight graph for dendro-dendritic synapses.
#
#  **Gdd**: the raw-weight graph for dendro-axonic synapses.


#%%

import math
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.cluster import GaussianCluster, KMeansCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import degreeplot, edgeplot, gridplot, heatmap, pairplot
from graspy.utils import augment_diagonal, binarize, cartprod, pass_to_ranks, to_laplace
from joblib.parallel import Parallel, delayed
from matplotlib.colors import LogNorm
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
from src.utils import get_best, meta_to_array, relabel, savefig, unique_by_size
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
adj, class_labels, side_labels, id_labels = load_everything(
    "G", version=BRAIN_VERSION, return_class=True, return_side=True, return_ids=True
)

side_map = {" mw right": "R", " mw left": "L"}
side_labels = np.array(itemgetter(*side_labels)(side_map))


degrees = adj.sum(axis=0) + adj.sum(axis=1)
sort_inds = np.argsort(degrees)[::-1]
og_class_labels = class_labels[sort_inds]
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
class_labels = np.array(itemgetter(*og_class_labels)(name_map))


# Now load all 4 colors
color_adjs = []
for t in GRAPH_TYPES:
    adj = load_everything(t)
    adj = adj[np.ix_(sort_inds, sort_inds)]
    color_adjs.append(adj)

sum_adj = np.array(color_adjs).sum(axis=0)

# %% [markdown]
#  ## Print some summary statistics and plot the adjacency matrices

# %%
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

# %% [markdown]
#  ## Plot edge weight sums (edgesums) and degrees for the full (G) graph
#  Note that weights here are the actual number of synapses

# %%

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

# %% [markdown]
#  ## Look at edgesums in different ways, and make plots for the 4-color edgesums as well
#  First two plots are still for **G**, the rest are for the 4-color

# %%


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

# %% [markdown]
#  # Block-wise nonzero edge weight distributions
#  Shown for the full (summed) graph and split by the 4 colors

# %%


def unique_ind_map(labels):
    unique_labels, inverse_labels = np.unique(labels, return_inverse=True)
    label_ind_map = {}
    for i, class_name in enumerate(unique_labels):
        inds = np.where(inverse_labels == i)[0]
        label_ind_map[class_name] = inds
    return label_ind_map


def get_block_edgeweights(adj, labels):
    unique_labels = unique_by_size(labels)[0]
    ind_map = unique_ind_map(labels)
    dfs = []
    for i, from_label in enumerate(unique_labels):
        for j, to_label in enumerate(unique_labels):
            from_inds = ind_map[from_label]
            to_inds = ind_map[to_label]
            subgraph = adj[np.ix_(from_inds, to_inds)]
            edges = subgraph[subgraph != 0]
            from_indicator = len(edges) * [from_label]
            to_indicator = len(edges) * [to_label]
            temp_df = pd.DataFrame()
            temp_df["From"] = from_indicator
            temp_df["To"] = to_indicator
            temp_df["Weight"] = edges
            temp_df["Log weight"] = np.log(edges)
            dfs.append(temp_df)
    return pd.concat(dfs)


name_map = {
    "CN": "Unknown",
    "DANs": "MBIN",
    "KCs": "KC",
    "LHN": "Unknown",
    "LHN; CN": "Unknown",
    "MBINs": "MBIN",
    "MBON": "MBON",
    "MBON; CN": "MBON",
    "OANs": "MBIN",
    "ORN mPNs": "PN",
    "ORN uPNs": "PN",
    "tPNs": "PN",
    "vPNs": "PN",
    "Unidentified": "Unknown",
    "Other": "Unknown",
}
simple_class_labels = np.array(itemgetter(*og_class_labels)(name_map))


edge_df = get_block_edgeweights(sum_adj, simple_class_labels)

# %% [markdown]
#  ## Nonzero edge weight for full graph

# %%
fg = sns.FacetGrid(
    edge_df, row="From", col="To", sharex=True, sharey=False, margin_titles=True
)
fg.map(sns.distplot, "Weight", kde=False)
fg.set(yticks=[])
# %% [markdown]
#  ## Nonzero log edge weight for full graph

# %%
fg = sns.FacetGrid(
    edge_df, row="From", col="To", sharex=True, sharey=False, margin_titles=True
)
fg.map(sns.distplot, "Log weight", kde=False)
fg.set(yticks=[])


# %%
dfs = []
for g, name in zip(color_adjs, GRAPH_TYPE_LABELS):
    edge_df = get_block_edgeweights(g, class_labels)
    edge_df["Edge type"] = name
    dfs.append(edge_df)
color_edge_df = pd.concat(dfs)

# %% [markdown]
#  # Nonzero edge weight split by synapse type

# %%
fg = sns.FacetGrid(
    color_edge_df,
    row="From",
    col="To",
    hue="Edge type",
    sharex=True,
    sharey=False,
    margin_titles=True,
)
fg = fg.map(sns.distplot, "Weight")
fg = fg.add_legend()

# %% [markdown]
#  # Nonzero log edge weight split by synapse type

# %%
fg = sns.FacetGrid(
    color_edge_df,
    row="From",
    col="To",
    hue="Edge type",
    sharex=True,
    sharey=False,
    margin_titles=True,
)
fg = fg.map(sns.distplot, "Log weight")
fg = fg.add_legend()

# %% [markdown]
# # Let's fit some binary SBM's to just look at edge density


def probplot(
    adj,
    labels,
    log_scale=False,
    figsize=(20, 20),
    cmap="Purples",
    title="Edge probability",
    vmin=None,
    vmax=None,
    ax=None,
):
    sbm = SBMEstimator(directed=True, loops=True)
    sbm.fit(binarize(adj), y=labels)
    data = sbm.block_p_
    uni_labels = np.unique(labels)

    cbar_kws = {"fraction": 0.08, "shrink": 0.8, "pad": 0.03}

    if log_scale:
        data = data + 0.001

        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        cbar_ticks = [
            math.pow(10, i)
            for i in range(
                math.floor(math.log10(data.min().min())),
                1 + math.ceil(math.log10(data.max().max())),
            )
        ]
        cbar_kws["ticks"] = cbar_ticks

    prob_df = pd.DataFrame(columns=uni_labels, index=uni_labels, data=data)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.set_title(title, pad=30, fontsize=30)

    sns.set_context("talk", font_scale=1)

    heatmap_kws = dict(
        cbar_kws=cbar_kws, annot=True, square=True, cmap=cmap, vmin=vmin, vmax=vmax
    )
    if log_scale:
        heatmap_kws["norm"] = log_norm
    if ax is not None:
        heatmap_kws["ax"] = ax

    ax = sns.heatmap(prob_df, **heatmap_kws)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return ax, prob_df


def merge_labels(labels1, labels2):
    new_labels = []
    for l1, l2 in zip(labels1, labels2):
        new_str = l1 + "-" + l2
        new_labels.append(new_str)
    return np.array(new_labels)


name_map = {
    "CN": "Other",
    "DANs": "MB",
    "KCs": "MB",
    "LHN": "Other",
    "LHN; CN": "Other",
    "MBINs": "MB",
    "MBON": "MB",
    "MBON; CN": "MB",
    "OANs": "MB",
    "ORN mPNs": "PN",
    "ORN uPNs": "PN",
    "tPNs": "PN",
    "vPNs": "PN",
    "Unidentified": "Other",
    "Other": "Other",
}
mb_labels = np.array(itemgetter(*og_class_labels)(name_map))


probplot(sum_adj, class_labels, title="Edge probability - class")
probplot(sum_adj, class_labels, title="Edge probability - class", log_scale=True)

probplot(sum_adj, side_labels, title="Edge probability - side")
probplot(sum_adj, side_labels, title="Edge probability - side", log_scale=True)

side_class_labels = merge_labels(side_labels, simple_class_labels)

probplot(sum_adj, side_class_labels, title="Edge probability - side/class")
probplot(
    sum_adj, side_class_labels, title="Edge probability - side/class", log_scale=True
)

probplot(sum_adj, mb_labels, title="Edge probability - MB")
probplot(sum_adj, mb_labels, title="Edge probability - MB", log_scale=True)

side_mb_labels = merge_labels(side_labels, mb_labels)
probplot(sum_adj, side_mb_labels, title="Edge probability - side/MB")
ax, prob_df = probplot(
    sum_adj, side_mb_labels, title="Edge probability - side/MB", log_scale=True
)

# %% [markdown]
# #
long_prob_df = prob_df.unstack().reset_index()
long_prob_df.columns = ["To", "From", "Probability"]
to_hemisphere = long_prob_df["To"].values.astype("<U1")
from_hemisphere = long_prob_df["From"].values.astype("<U1")
pairs = long_prob_df["From"] + long_prob_df["To"]
new_pairs = []
for f, t in zip(long_prob_df["From"].values, long_prob_df["To"].values):
    f = f[2:]
    t = t[2:]
    print(t)
    print()
    new_pairs.append(f + "-" + t)

hemidirection = [f"{f} to {t}" for f, t in zip(from_hemisphere, to_hemisphere)]
long_prob_df["Hemidirection"] = hemidirection
long_prob_df["Pair"] = new_pairs
long_prob_df["Log prob"] = np.log(long_prob_df["Probability"].values)
plt.figure(figsize=(20, 10))

ax = sns.pointplot(data=long_prob_df, x="Hemidirection", y="Log prob", hue="Pair")
ax.get_legend().remove()
# %% [markdown]
# #
sns.catplot(
    data=long_prob_df,
    x="Hemidirection",
    y="Probability",
    hue="Pair",
    kind="point",
    palette="Set1",
)

# %% [markdown]
# #

pairs = np.unique(long_prob_df["Pair"].values)
for p in pairs:
    df = long_prob_df[long_prob_df["Pair"] == p]
    print(df)
    print()
    sns.pointplot(data=df, x="Hemidirection", y="Probability")


# ax.get_legend().remove()
#%% Do some of the same but for the 4-colors


def multi_probplot(
    adjs,
    labels,
    titles=None,
    cmaps=None,
    nrows=2,
    ncols=2,
    log_scale=False,
    figsize=(20, 20),
    title="Edge probability",
    vmin=None,
    vmax=None,
    ax=None,
):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax = ax.ravel()

    if cmaps is None:
        cmaps = len(adjs) * ["Purples"]
    if titles is None:
        titles = len(adjs) * [None]

    dfs = []
    for i, a in enumerate(adjs):
        _, prob_df = probplot(
            a,
            labels,
            log_scale=log_scale,
            cmap=cmaps[i],
            title=titles[i],
            vmin=vmax,
            vmax=vmax,
            ax=ax[i],
        )
        dfs.append(prob_df)

    return ax, dfs


cmaps = ["Purples", "Greens", "Oranges", "Reds"]
titles = GRAPH_TYPE_LABELS
figsize = (30, 30)
multi_probplot(
    color_adjs, side_labels, cmaps=cmaps, titles=titles, vmin=0, figsize=figsize
)
multi_probplot(
    color_adjs, simple_class_labels, cmaps=cmaps, titles=titles, vmin=0, figsize=figsize
)
multi_probplot(
    color_adjs, side_class_labels, cmaps=cmaps, titles=titles, vmin=0, figsize=figsize
)
#%%
_, dfs = multi_probplot(
    color_adjs, side_mb_labels, cmaps=cmaps, titles=titles, vmin=0, figsize=figsize
)
#%%
new_dfs = []
for df, name in zip(dfs, GRAPH_TYPE_LABELS):
    new_df = df.unstack().reset_index()
    new_df.columns = ["To", "From", "Probability"]
    new_df["Pair"] = new_df["From"] + new_df["To"]
    new_df["Edge type"] = name
    new_dfs.append(new_df)

overall_prob_df = pd.concat(new_dfs)
plt.figure(figsize=(20, 10))
ax = sns.pointplot(
    data=overall_prob_df, x="Edge type", y="Probability", hue="Pair", linestyles="--"
)
ax.get_legend().remove()

#%% how does degree or edge weight correlate across color?

color_in_degrees = []
color_out_degrees = []
color_degrees = []

for a in color_adjs:
    in_degree = np.count_nonzero(a, axis=0)
    out_degree = np.count_nonzero(a, axis=1)
    degree = in_degree + out_degree
    color_in_degrees.append(in_degree)
    color_out_degrees.append(out_degree)
    color_degrees.append(degree)

color_in_degrees = np.array(color_in_degrees).T
color_out_degrees = np.array(color_out_degrees).T
color_degrees = np.array(color_degrees).T

pairplot(color_in_degrees, col_names=GRAPH_TYPE_LABELS, title="In degree")
pairplot(color_out_degrees, col_names=GRAPH_TYPE_LABELS, title="Out degree")
pairplot(color_degrees, col_names=GRAPH_TYPE_LABELS, title="Degree")
pairplot(
    color_degrees, col_names=GRAPH_TYPE_LABELS, title="Degree", labels=class_labels
)
#%%
color_degrees_df = pd.DataFrame(columns=GRAPH_TYPE_LABELS, data=color_degrees)
sns.pairplot(
    color_degrees_df, kind="reg", plot_kws=dict(scatter_kws=dict(s=10, alpha=0.3))
)
#%%


# Borrowed from http://stackoverflow.com/a/31385996/4099925
def hexbin(x, y, color, max_series=None, min_series=None, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    ax = plt.gca()
    xmin, xmax = min_series[x.name], max_series[x.name]
    ymin, ymax = min_series[y.name], max_series[y.name]
    plt.hexbin(x, y, gridsize=15, cmap=cmap, extent=[xmin, xmax, ymin, ymax], **kwargs)


def jankplot(df, title=None):
    pg = sns.PairGrid(df, diag_sharey=False, height=5)
    pg.map_lower(
        sns.regplot, scatter_kws=dict(s=10, alpha=0.3), line_kws=dict(color="red")
    )
    pg.map_upper(hexbin, min_series=df.min(), max_series=df.max())
    pg.map_diag(sns.distplot, kde=False)
    plt.suptitle(title, y=1, verticalalignment="bottom", fontsize=30)


jankplot(color_degrees_df, title="Total degree (in + out)")
color_in_degrees_df = pd.DataFrame(columns=GRAPH_TYPE_LABELS, data=color_in_degrees)
jankplot(color_in_degrees_df, title="In degree")
color_out_degrees_df = pd.DataFrame(columns=GRAPH_TYPE_LABELS, data=color_out_degrees)
jankplot(color_out_degrees_df, title="Out degree")

# %% [markdown]
# # Generate line plots showing the probability of block-block edges


# %% look at the distribution of fraction of edges/synapses that cross


def calc_class_output_proportions(adj, labels, weights=True):
    ind_map = unique_ind_map(labels)
    df = pd.DataFrame(index=list(range(len(labels))))
    for key, inds in ind_map.items():
        submatrix = adj[:, inds]
        if weights:
            outputs = submatrix.sum(axis=1)
        else:
            outputs = np.count_nonzero(submatrix, axis=1)
        df[key] = outputs
    total_output = np.sum(df.values, axis=1)
    for key in ind_map.keys():
        df[key] = df[key] / total_output
    return df


# %% [markdown]
# # On average, cells project similarly to the two different hemispheres

side_output_df = calc_class_output_proportions(sum_adj, side_labels, weights=True)
side_output_df["On L"] = side_labels == "L"
side_output_df["ID"] = id_labels
side_output_df.dropna(inplace=True)
left_to_left = side_output_df[side_output_df["On L"]]["L"].values
right_to_left = side_output_df[~side_output_df["On L"]]["L"].values

plt.figure(figsize=(15, 10))
sns.distplot(left_to_left, label="Left")
sns.distplot(right_to_left, label="Right")
plt.xlabel("Proprtion output synapses to left")
plt.legend(title="Cell side")

side_output_df = calc_class_output_proportions(sum_adj, side_labels, weights=False)
side_output_df["On L"] = side_labels == "L"
side_output_df["ID"] = id_labels
side_output_df.dropna(inplace=True)
left_to_left = side_output_df[side_output_df["On L"]]["L"].values
right_to_left = side_output_df[~side_output_df["On L"]]["L"].values

plt.figure(figsize=(15, 10))
sns.distplot(left_to_left, label="Left")
sns.distplot(right_to_left, label="Right")
plt.xlabel("Proprtion output edges to left")
plt.legend(title="Cell side")

# %% [markdown]
# # Look at the same thing but split by graph type
for color_adj, name in zip(color_adjs, GRAPH_TYPE_LABELS):
    side_output_df = calc_class_output_proportions(color_adj, side_labels, weights=True)
    side_output_df["On L"] = side_labels == "L"
    side_output_df["ID"] = id_labels
    side_output_df.dropna(inplace=True)
    left_to_left = side_output_df[side_output_df["On L"]]["L"].values
    right_to_left = side_output_df[~side_output_df["On L"]]["L"].values

    plt.figure(figsize=(15, 10))
    sns.distplot(left_to_left, label="Left", kde=False)
    sns.distplot(right_to_left, label="Right", kde=False)
    plt.xlabel("Proprtion output synapses to left")
    plt.legend(title="Cell side")
    plt.title(name)


# %% [markdown]
#  ## TODO
#  - Test different block-block SBM hypotheses
#  - Look at distribution of weights for left right blocks
#  - check whether the left right class matrices are the same
#       - for the edge probability side/class plot
#       = draw a vector for each block unraveled, draw lines in between slope graph
#   - check if the muchsroom body is the most connected
#       - look at newman modularity clusters?
#   - A to A and D to D look different if looking at separate MB class, not for whole

