#%% Imports
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.cluster import GaussianCluster, KMeansCluster
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator
from graspy.plot import gridplot, heatmap, pairplot
from graspy.utils import augment_diagonal, binarize, cartprod, pass_to_ranks, to_laplace
from joblib.parallel import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy
from sklearn.cluster import KMeans
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


# Functions


def annotate_arrow(ax, coords=(0.061, 0.93)):
    arrow_args = dict(
        arrowstyle="-|>",
        color="k",
        connectionstyle="arc3,rad=-0.4",  # "angle3,angleA=90,angleB=90"
    )
    t = ax.annotate("Target", xy=coords, xycoords="figure fraction")

    ax.annotate(
        "Source", xy=(0, 0.5), xycoords=t, xytext=(-1.4, -2.1), arrowprops=arrow_args
    )


def ase(adj, n_components):
    if PTR:
        adj = pass_to_ranks(adj)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    latent = ase.fit_transform(adj)
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


def calc_weighted_entropy(true_labels, pred_labels):
    total_entropy = 0
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    for true_label in unique_true_labels:
        if (
            true_label == -1 or true_label == "Unknown"
        ):  # this is for "unlabeled" points
            continue
        probs = np.zeros(unique_pred_labels.shape)
        true_inds = np.where(true_labels == true_label)[0]
        class_pred_labels = pred_labels[
            true_inds
        ]  # get the predicted class assignments for this true class
        uni_inds, counts = np.unique(class_pred_labels, return_counts=True)
        probs[uni_inds] = counts
        probs /= len(class_pred_labels)
        e = entropy(probs)
        e *= len(class_pred_labels) / len(true_labels)
        e /= np.log(len(unique_pred_labels))
        total_entropy += e
    return total_entropy


def generate_experiment_arglist(latents, true_labels):
    arglist = []
    for i, (latent, latent_name) in enumerate(zip(latents, EMBED_FUNC_NAMES)):
        for j, (estimator, estimator_name) in enumerate(
            zip(ESTIMATORS, ESTIMATOR_NAMES)
        ):
            for k in range(MIN_CLUSTERS, MAX_CLUSTERS):
                arglist.append(
                    (
                        true_labels,
                        latent,
                        latent_name,
                        estimator,
                        estimator_name,
                        k,
                        params[j],
                    )
                )
    return arglist


def ari_scorer(estimator, latent, y=None):
    pred_labels = estimator.fit_predict(latent)
    return adjusted_rand_score(y, pred_labels)


def entropy_scorer(estimator, latent, y=None):
    pred_labels = estimator.fit_predict(latent)
    return calc_weighted_entropy(y, pred_labels)


def bic_scorer(estimator, latent, y=None):
    if type(estimator) == GaussianCluster:
        bic = estimator.model_.bic(latent)
        return bic
    else:
        return np.nan


def inertia_scorer(estimator, latent, y=None):
    if type(estimator) == KMeans or type(estimator) == SphericalKMeans:
        inert = estimator.inertia_
        return inert
    else:
        return np.nan


def run_clustering(
    seed,
    true_labels,
    latent,
    latent_name,
    estimator,
    estimator_name,
    n_clusters,
    params,
):
    np.random.seed(seed)
    if estimator == GaussianCluster:
        e = estimator(min_components=n_clusters, max_components=n_clusters, **params)
    else:
        e = estimator(n_clusters=n_clusters, **params)
    e.fit(latent)
    ari = ari_scorer(e, latent, y=true_labels)
    ent = entropy_scorer(e, latent, y=true_labels)
    bic = bic_scorer(e, latent, y=true_labels)
    inert = inertia_scorer(e, latent, y=true_labels)
    out_dict = {
        "ARI": ari,
        "Entropy": ent,
        "Embed": latent_name,
        "Cluster": estimator_name,
        "# Clusters": n_clusters,
        "BIC": bic,
        "Inertia": inert,
    }
    return out_dict


def run_clustering_experiment(
    latents, true_labels, min_clusters, max_clusters, n_sims, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    arglist = generate_experiment_arglist(latents, true_labels)
    arglist = arglist * n_sims

    seeds = np.random.randint(1e8, size=n_sims * len(arglist))

    outs = Parallel(n_jobs=-2, verbose=10)(
        delayed(run_clustering)(s, *i) for s, i in zip(seeds, arglist)
    )

    cluster_df = pd.DataFrame.from_dict(outs)

    return cluster_df


# Global alg parameters
PTR = True
EMBED_FUNC_NAMES = ["ASE", "OMNI", "ASE-Cat"]
EMBED_FUNCS = [ase, omni, ase_concatenate]

ESTIMATORS = [GaussianCluster, SphericalKMeans, KMeans]
ESTIMATOR_NAMES = ["GMM", "SKmeans", "Kmeans"]

MAX_CLUSTERS = 12
MIN_CLUSTERS = 2
N_SIMS = 1
N_INIT = 100

# Set up plotting constants
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)


#%%
adj, class_labels, side_labels = load_everything(
    "G", version=BRAIN_VERSION, return_class=True, return_side=True
)

right_inds = np.where(side_labels == " mw right")[0]
adj = adj[np.ix_(right_inds, right_inds)]
degrees = adj.sum(axis=0) + adj.sum(axis=1)
sort_inds = np.argsort(degrees)[::-1]
class_labels = class_labels[right_inds]  # need to do right inds, then sort_inds
class_labels = class_labels[sort_inds]
adj = adj[np.ix_(sort_inds, sort_inds)]

# Remap the names
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


skmeans_kws = dict(n_init=200, n_jobs=-2)
n_clusters = 12
n_components = None
regularizer = 1

# embed
adj = pass_to_ranks(adj)
lse = LaplacianSpectralEmbed(n_components=n_components, regularizer=regularizer)
latent = lse.fit_transform(adj)
latent = np.concatenate(latent, axis=-1)


# uni_class, class_counts = np.unique(class_labels, return_counts=True)
# inds = np.argsort(class_counts)[::-1]
# uni_class = uni_class[inds]
# class_counts = class_counts[inds]


priority_map = {
    "DAN": 0,
    "KC": 0,
    "Unknown": 2,
    "MBIN": 0,
    "MBON": 0,
    "OAN": 0,
    "mPN": 0,
    "uPN": 0,
    "tPN": 0,
    "vPN": 0,
}
label_priorities = np.array(itemgetter(*class_labels)(priority_map))
sort_df = pd.DataFrame()
sort_df["ind"] = list(range(len(class_labels)))
sort_df["class"] = class_labels
sort_df["priority"] = label_priorities
sort_df = sort_df.sort_values(["priority", "class"])
sort_df.head()


from src.visualization import sankey

inds = sort_df["ind"].values
latent = latent[inds, :]
class_labels = sort_df["class"].values
adj = adj[np.ix_(inds, inds)]

n_clusters = 7
for k in range(2, n_clusters):
    skmeans = SphericalKMeans(n_clusters=k, **skmeans_kws)
    pred_labels = skmeans.fit_predict(latent)
    pred_labels = relabel(pred_labels)

    fig, ax = plt.subplots(2, 2, figsize=(30, 30))
    ax = ax.ravel()
    heatmap(
        binarize(adj),
        inner_hier_labels=pred_labels,
        # outer_hier_labels=side_labels,
        hier_label_fontsize=18,
        ax=ax[0],
        cbar=False,
        sort_nodes=True,
    )
    # uni_labels = np.unique(pred_labels)
    sankey(ax[1], class_labels, pred_labels)
    ax[1].axis("off")

    rand_perm = np.random.permutation(len(pred_labels))

    heatmap(
        binarize(adj),
        inner_hier_labels=pred_labels[rand_perm],
        # outer_hier_labels=side_labels,
        hier_label_fontsize=18,
        ax=ax[2],
        cbar=False,
        sort_nodes=True,
    )
    sankey(ax[3], class_labels, pred_labels[rand_perm])
    ax[3].axis("off")
    print(k)
    if k == 6:
        print(k)
        savefig("sankey6", fmt="png", dpi=200)


# %%

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.set_context("talk", font_scale=2)
sankey(ax, class_labels, pred_labels, aspect=20, fontsize=24)
ax.axis("off")

# %%
