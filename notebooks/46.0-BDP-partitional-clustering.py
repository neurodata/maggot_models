# %% [markdown]
# # Imports
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
from sklearn.metrics import adjusted_rand_score, silhouette_score
import networkx as nx
from spherecluster import SphericalKMeans

from graspy.cluster import GaussianCluster, AutoGMMCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks
from src.data import load_everything
from src.utils import export_skeleton_json, savefig
from src.visualization import clustergram, palplot, sankey
from src.hierarchy import signal_flow
from src.embed import lse
from src.io import stashfig

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
ONLY_RIGHT = True

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


# %% [markdown]
# # Run clustering using LSE on the sum graph

n_verts = adj.shape[0]


latent = lse(adj, n_components, regularizer=None, ptr=PTR)
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
elif cluster == "SKMeans":
    ClusterModel = SphericalKMeans

for k in k_list:
    run_name = f"k = {k}, {cluster}, {embed}, {side} (A to D), PTR, raw"
    print(run_name)
    print()

    # Do clustering
    # TODO: make this autogmm instead
    if cluster in ["GMM", "AutoGMM"]:
        gmm = ClusterModel(min_components=k, max_components=k, **gmm_params)
    elif cluster in ["SKMeans", "KMeans"]:
        gmm = ClusterModel(n_clusters=k, **gmm_params)

    gmm.fit(latent)
    pred_labels = gmm.predict(latent)

    # Score unsupervised metrics
    base_dict = {
        "K": k,
        "Cluster": cluster,
        "Embed": embed,
        "Method": f"{cluster} o {embed}",
    }

    if cluster in ["GMM", "AutoGMM"]:
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

    elif cluster in ["SKMeans", "KMeans"]:
        score = gmm.score(latent)
        temp_dict = base_dict.copy()
        temp_dict["Metric"] = "Inertia score"
        temp_dict["Score"] = score
        out_dicts.append(temp_dict)

        score = silhouette_score(latent, pred_labels, metric="cosine")
        temp_dict = base_dict.copy()
        temp_dict["Metric"] = "Silhouette score"
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

    stashobj(gmm, str("cluster-" + save_name))

    # Plot everything else
    clustergram(adj, class_labels, pred_labels)
    stashfig("clustergram-" + save_name, save_on=SAVEFIGS)

    # New plot
    # - Compute signal flow
    # - Get the centroid of each cluster and project to 1d
    # - Alternatively, just take the first dimension
    # - For each cluster plot as a node

    # output skeletons
    if SAVESKELS:
        _, colormap, pal = stashskel(
            save_name, skeleton_labels, pred_labels, palette="viridis", multiout=True
        )
        stashskel(
            save_name, skeleton_labels, pred_labels, palette="viridis", multiout=False
        )

        palplot(k, cmap="viridis")
        stashfig("palplot-" + save_name, save_on=SAVEFIGS)

        # save dict colormapping
        filename = (
            Path("./maggot_models/notebooks/outs")
            / Path(FNAME)
            / Path("jsons")
            / str("colormap-" + save_name + ".json")
        )
        with open(filename, "w") as fout:
            json.dump(colormap, fout)


# %% [markdown]
# #  Plot results of unsupervised metrics

result_df = pd.DataFrame(out_dicts)
stashobj(result_df, f"metrics-{cluster}-{embed}-right-ad-PTR-raw")
fg = sns.FacetGrid(result_df, col="Metric", col_wrap=3, sharey=False, height=4)
fg.map(sns.lineplot, "K", "Score")
stashfig(f"metrics-{cluster}-{embed}-right-ad-PTR-raw", save_on=SAVEFIGS)


# Modifications i need to make to the above
# - Increase the height of the sankey diagram overall
# - Look into color maps that could be better
# - Color the cluster labels in the sankey diagram by what gets written to the JSON
# - Plot the clusters as nodes in a small network

