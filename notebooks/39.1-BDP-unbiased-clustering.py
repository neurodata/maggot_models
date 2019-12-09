# %% [markdown]
# # Imports
import json
import os
import warnings
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from joblib.parallel import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from graspy.cluster import GaussianCluster, AutoGMMCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks
from src.data import load_everything
from src.utils import export_skeleton_json, savefig
from src.visualization import clustergram, palplot, sankey

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
DEFAULT_FMT = "png"
DEFUALT_DPI = 150

SAVESKELS = True

MIN_CLUSTERS = 8
MAX_CLUSTERS = 8
N_INIT = 50
PTR = True
ONLY_RIGHT = True

embed = "LSE"
cluster = "GMM"
n_components = 4
if cluster == "GMM":
    gmm_params = {"n_init": N_INIT, "covariance_type": "all"}
elif cluster == "AutoGMM":
    gmm_params = {"max_agglom_size": None}

np.random.seed(23409857)


def stashfig(name, **kws):
    if SAVEFIGS:
        savefig(name, foldername=FNAME, fmt=DEFAULT_FMT, dpi=DEFUALT_DPI, **kws)


def stashskel(name, ids, colors, palette=None, **kws):
    if SAVESKELS:
        return export_skeleton_json(
            name, ids, colors, palette=palette, foldername=FNAME, **kws
        )


def ase(adj, n_components):
    if PTR:
        adj = pass_to_ranks(adj)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    latent = ase.fit_transform(adj)
    latent = np.concatenate(latent, axis=-1)
    return latent


def to_laplace(graph, form="DAD", regularizer=None):
    r"""
    A function to convert graph adjacency matrix to graph laplacian. 
    Currently supports I-DAD, DAD, and R-DAD laplacians, where D is the diagonal
    matrix of degrees of each node raised to the -1/2 power, I is the 
    identity matrix, and A is the adjacency matrix.
    
    R-DAD is regularized laplacian: where :math:`D_t = D + regularizer*I`.
    Parameters
    ----------
    graph: object
        Either array-like, (n_vertices, n_vertices) numpy array,
        or an object of type networkx.Graph.
    form: {'I-DAD' (default), 'DAD', 'R-DAD'}, string, optional
        
        - 'I-DAD'
            Computes :math:`L = I - D*A*D`
        - 'DAD'
            Computes :math:`L = D*A*D`
        - 'R-DAD'
            Computes :math:`L = D_t*A*D_t` where :math:`D_t = D + regularizer*I`
    regularizer: int, float or None, optional (default=None)
        Constant to be added to the diagonal of degree matrix. If None, average 
        node degree is added. If int or float, must be >= 0. Only used when 
        ``form`` == 'R-DAD'.
    Returns
    -------
    L: numpy.ndarray
        2D (n_vertices, n_vertices) array representing graph 
        laplacian of specified form
    References
    ----------
    .. [1] Qin, Tai, and Karl Rohe. "Regularized spectral clustering
           under the degree-corrected stochastic blockmodel." In Advances
           in Neural Information Processing Systems, pp. 3120-3128. 2013
    """
    valid_inputs = ["I-DAD", "DAD", "R-DAD"]
    if form not in valid_inputs:
        raise TypeError("Unsuported Laplacian normalization")

    A = graph

    in_degree = np.sum(A, axis=0)
    out_degree = np.sum(A, axis=1)

    # regularize laplacian with parameter
    # set to average degree
    if form == "R-DAD":
        if regularizer is None:
            regularizer = 1
        elif not isinstance(regularizer, (int, float)):
            raise TypeError(
                "Regularizer must be a int or float, not {}".format(type(regularizer))
            )
        elif regularizer < 0:
            raise ValueError("Regularizer must be greater than or equal to 0")
        regularizer = regularizer * np.mean(out_degree)

        in_degree += regularizer
        out_degree += regularizer

    with np.errstate(divide="ignore"):
        in_root = 1 / np.sqrt(in_degree)  # this is 10x faster than ** -0.5
        out_root = 1 / np.sqrt(out_degree)

    in_root[np.isinf(in_root)] = 0
    out_root[np.isinf(out_root)] = 0

    in_root = np.diag(in_root)  # just change to sparse diag for sparse support
    out_root = np.diag(out_root)

    if form == "I-DAD":
        L = np.diag(in_degree) - A
        L = in_root @ L @ in_root
    elif form == "DAD" or form == "R-DAD":
        L = out_root @ A @ in_root
    # return symmetrize(L, method="avg")  # sometimes machine prec. makes this necessary
    return L


def lse(adj, n_components, regularizer=None):
    if PTR:
        adj = pass_to_ranks(adj)
    lap = to_laplace(adj, form="R-DAD")
    ase = AdjacencySpectralEmbed(n_components=n_components)
    latent = ase.fit_transform(lap)
    latent = np.concatenate(latent, axis=-1)
    return latent


def omni(adjs, n_components):
    if PTR:
        adjs = [pass_to_ranks(a) for a in adjs]
    omni = OmnibusEmbed(n_components=n_components // len(adjs))
    latent = omni.fit_transform(adjs)
    latent = np.concatenate(latent, axis=-1)  # first is for in/out
    latent = np.concatenate(latent, axis=-1)  # second is for concat. each graph
    return latent


def ase_concatenate(adjs, n_components):
    if PTR:
        adjs = [pass_to_ranks(a) for a in adjs]
    ase = AdjacencySpectralEmbed(n_components=n_components // len(adjs))
    graph_latents = []
    for a in adjs:
        latent = ase.fit_transform(a)
        latent = np.concatenate(latent, axis=-1)
        graph_latents.append(latent)
    latent = np.concatenate(graph_latents, axis=-1)
    return latent


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


# Remap the names
# name_map = {
#     "CN": "Unk",
#     "DANs": "MBIN",
#     "KCs": "KC",
#     "LHN": "Unk",
#     "LHN; CN": "Unk",
#     "MBINs": "MBIN",
#     "MBON": "MBON",
#     "MBON; CN": "MBON",
#     "OANs": "MBIN",
#     "ORN mPNs": "mPN",
#     "ORN uPNs": "uPN",
#     "tPNs": "tPN",
#     "vPNs": "vPN",
#     "Unidentified": "Unk",
#     "Other": "Unk",
# }
# simple_class_labels = np.array(itemgetter(*class_labels)(name_map))

# name_map = {
#     "CN": "Other",
#     "DANs": "MB",
#     "KCs": "MB",
#     "LHN": "Other",
#     "LHN; CN": "Other",
#     "MBINs": "MB",
#     "MBON": "MB",
#     "MBON; CN": "MB",
#     "OANs": "MB",
#     "ORN mPNs": "PN",
#     "ORN uPNs": "PN",
#     "tPNs": "PN",
#     "vPNs": "PN",
#     "Unidentified": "Other",
#     "Other": "Other",
# }
# mb_labels = np.array(itemgetter(*class_labels)(name_map))

known_inds = np.where(class_labels != "Unk")[0]


# %% [markdown]
# # Run clustering using LSE on the sum graph

n_verts = adj.shape[0]


latent = lse(adj, n_components, regularizer=None)
pairplot(latent, labels=class_labels, title=embed)

k_list = list(range(MIN_CLUSTERS, MAX_CLUSTERS + 1))
n_runs = len(k_list)
out_dicts = []

bin_adj = binarize(adj)

last_pred_labels = np.zeros(n_verts)

if cluster == "GMM":
    ClusterModel = GaussianCluster
elif cluster == "AutoGMM":
    ClusterModel = AutoGMMCluster

for k in k_list:
    run_name = f"k = {k}, {cluster}, {embed}, {side} (A to D), PTR, raw"
    print(run_name)
    print()

    # Do clustering
    # TODO: make this autogmm instead
    gmm = ClusterModel(min_components=k, max_components=k, **gmm_params)
    gmm.fit(latent)
    pred_labels = gmm.predict(latent)

    # Score unsupervised metrics
    base_dict = {
        "K": k,
        "Cluster": cluster,
        "Embed": embed,
        "Method": f"{cluster} o {embed}",
    }

    # GMM likelihood
    score = gmm.model_.score(latent)
    temp_dict = base_dict.copy()
    temp_dict["Metric"] = "GMM likelihood"
    temp_dict["Score"] = score
    out_dicts.append(temp_dict)

    # GMM BIC
    score = gmm.model_.bic(latent)
    temp_dict = base_dict.copy()
    temp_dict["Metric"] = "GMM BIC"
    temp_dict["Score"] = score
    out_dicts.append(temp_dict)

    # SBM likelihood
    sbm = SBMEstimator(directed=True, loops=False)
    sbm.fit(bin_adj, y=pred_labels)
    score = sbm.score(bin_adj)
    temp_dict = base_dict.copy()
    temp_dict["Metric"] = "SBM likelihood"
    temp_dict["Score"] = score
    out_dicts.append(temp_dict)

    # DCSBM likelihood
    dcsbm = DCSBMEstimator(directed=True, loops=False)
    dcsbm.fit(bin_adj, y=pred_labels)
    score = dcsbm.score(bin_adj)
    temp_dict = base_dict.copy()
    temp_dict["Metric"] = "DCSBM likelihood"
    temp_dict["Score"] = score
    out_dicts.append(temp_dict)

    # ARI of the subset with labels
    score = sub_ari(known_inds, class_labels, pred_labels)
    temp_dict = base_dict.copy()
    temp_dict["Metric"] = "Simple ARI"
    temp_dict["Score"] = score
    out_dicts.append(temp_dict)

    # ARI vs K - 1
    score = adjusted_rand_score(last_pred_labels, pred_labels)
    temp_dict = base_dict.copy()
    temp_dict["Metric"] = "K-1 ARI"
    temp_dict["Score"] = score
    out_dicts.append(temp_dict)
    last_pred_labels = pred_labels

    save_name = f"k{k}-{cluster}-{embed}-right-ad-PTR-raw"

    # Plot embedding
    # pairplot(latent, labels=pred_labels, title=run_name)
    # stashfig("latent-" + save_name)

    # Plot everything else
    clustergram(adj, class_labels, pred_labels)
    stashfig("clustergram-" + save_name)

    # output skeletons
    if SAVESKELS:
        _, colormap, pal = stashskel(
            save_name, skeleton_labels, pred_labels, palette="viridis", multiout=True
        )

        palplot(k, cmap="viridis")
        stashfig("palplot-" + save_name)

        # save dict colormapping
        filename = (
            Path("./maggot_models/notebooks/outs")
            / Path(FNAME)
            / str("colormap-" + save_name + ".json")
        )
        with open(filename, "w") as fout:
            json.dump(colormap, fout)

        stashskel(
            save_name, skeleton_labels, pred_labels, palette="viridis", multiout=False
        )

# %% [markdown]
# #  Plot results of unsupervised metrics

result_df = pd.DataFrame(out_dicts)
fg = sns.FacetGrid(result_df, col="Metric", col_wrap=3, sharey=False, height=4)
fg.map(sns.lineplot, "K", "Score")
stashfig(f"metrics-{cluster}-{embed}-right-ad-PTR-raw")


# Modifications i need to make to the above
# - Increase the height of the sankey diagram overall
# - Look into color maps that could be better
# - Color the cluster labels by what gets written to the JSON
# - Plot the clusters as nodes in a small network

