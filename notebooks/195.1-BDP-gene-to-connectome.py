#%% [markdown]
## Linking transcriptome and connectome via graph matching
# > A simple attempt to see if one 'ome can be aligned to the other 'ome
#
# - toc: true
# - badges: false
# - categories: [pedigo, graspologic]
# - hide: true
# - search_exclude: true

#%%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from umap import UMAP

from graspologic.match import GraphMatch
from graspologic.plot import heatmap
from graspologic.simulations import er_corr
from src.io import savefig
from src.visualization import set_theme

set_theme()


FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


data_dir = Path("maggot_models/data/raw/BP_Barabasi_Share/ScRNAData/")

# gene expression data
sequencing_loc = data_dir / "Celegans_ScRNA_OnlyLabeledNeurons.csv"
sequencing_df = pd.read_csv(sequencing_loc, skiprows=[1])
currtime = time.time()
sequencing_df = sequencing_df.pivot(index="genes", columns="neurons", values="Count")
print(f"{time.time() - currtime} elapsed")

# metadata for each neuron in the gene expression data
class_map_loc = data_dir / "Labels2_CElegansScRNA_onlyLabeledNeurons.csv"
scrna_meta = pd.read_csv(class_map_loc)
scrna_meta = scrna_meta.set_index("OldIndices")

# single neuron connectome data
connectome_loc = data_dir / "NeuralWeightedConn.csv"
adj_df = pd.read_csv(connectome_loc, index_col=None, header=None)
adj = adj_df.values

# metadata for neurons in the connectome
label_loc = data_dir / "NeuralWeightedConn_Labels.csv"
connectome_meta = pd.read_csv(label_loc)
connectome_meta["cell_name"] = connectome_meta["Var1"].map(lambda x: x.strip("'"))
connectome_meta["broad_type"] = connectome_meta["Var2"].map(lambda x: x.strip("'"))
connectome_meta["cell_type"] = connectome_meta["Var3"].map(lambda x: x.strip("'"))
connectome_meta["neurotransmitter"] = connectome_meta["Var4"].map(
    lambda x: x.strip("'")
)
connectome_meta["cell_type_index"] = connectome_meta["Var5"]
broad_labels = connectome_meta["broad_type"].values

#%% [markdown]
### The data

#%% [markdown]
#### scRNAseq
# The single cell RNA sequencing data can be thought of as a matrix
# $X \in \mathbb{R}^{G \times M}$ where $G$ is the number of genes that were measured in
# the experiment and $M$ is the number of neurons. Each element in $X$ is a positive integer
# representing the number of times that gene was measured in each neuron.
#
# Note that here we are primarily concerned with *neuron classes* a.k.a. cell types. The
# goal is to see if we can match neurons in the scRNAseq data to their corresponding
# cell type in the connectome. Multiple neurons from each of the $C$ classes are measured
# in the scRNAseq data - that is, anywhere between ~10 - 6,000 rows of $X$ could
# correspond to neurons from a single cell type.

#%%
print(f"Number of genes measured: {sequencing_df.shape[0]}")
print(f"Number of neurons measured: {sequencing_df.shape[1]}")

#%%
# TODO just look at the expression data
# umapper = UMAP(min_dist=0.7)
# umapper.fit_transform()

#%% [markdown]
#### Connectome data

#
# Below I just plot $A$, sorting by cell type
#%%
heatmap(
    adj,
    transform="simple-all",
    inner_hier_labels=broad_labels,
    sort_nodes=True,
    title=r"Connectome ($A$)",
    hier_label_fontsize=12,
    cbar=False,
)

#%% [markdown]
### Matching transcriptome correlations to the connectome
# The experiment I ran goes as follows:
#
# 1. Uniformly at random, sample 1 neuron from each class to get
# $X_{sub} \in \mathbb{R}^{G \times 89}$.
# 2. Compute the correlation between columns of $X_{sub}$, yielding
# a $89 \times 89$ correlation matrix $S$.
# 3. Run graph matching between $A$ and $S$.
#    - Parameters of the graph matching: initialized at the barycenter,
#      best of 50 initializations for each trial, maximum of 30 iterations.
# 4. Repeat 1 - 3 100 times.
# 5. Compute a "confusion matrix" of sorts between neuron classes. The rows and columns
# represent the neuron classes, and each element says how often (out of the 100 trials)
# a neuron in class $i$ was matched to a neuron in class $j$.
#
# If the matching works well, then the diagonal of that matrix should be heavy.


#%%
# %% [markdown]
# ##
compute_full_similarity = False
if compute_full_similarity:
    from scipy.sparse import csr_matrix

    # takes about a minute
    currtime = time.time()
    scrna_X = sequencing_df.T.values
    inds = np.nonzero(~np.isnan(scrna_X))
    sparse_scrna_X = csr_matrix((scrna_X[inds], inds), shape=scrna_X.shape)
    print(f"{time.time() - currtime} elapsed")

    # takes a long ass time (didn't finish after ~40 mins)
    # TODO run on the server
    currtime = time.time()
    scrna_similarity = cosine_similarity(sparse_scrna_X)
    print(f"{time.time() - currtime} elapsed")


#%%

# sampling_mode \in {"random", "proportional", "n_per_class"}

n_trials = 10
n_subsample = 2 ** 10
n_per_class = 5
sampling_mode = "n_per_class"

currtime = time.time()

neuron_matches = {i: [] for i in range(len(adj_df))}
results = []

for i in tqdm(range(n_trials)):
    if sampling_mode == "n_per_class":
        neuron_sample = scrna_meta.groupby("CellTypeIndex").sample(n=n_per_class).index
        subset_sequencing_df = sequencing_df[neuron_sample]
    elif sampling_mode == "random":
        neuron_sample = np.random.choice(
            sequencing_df.columns, size=n_subsample, replace=False
        )
        subset_sequencing_df = sequencing_df[neuron_sample]

    inner_currtime = time.time()
    scrna_similarity = cosine_similarity(subset_sequencing_df.T.fillna(0).values)
    elapsed = time.time() - inner_currtime
    results.append({"replicate": i, "wall_time": elapsed, "stage": "Similarity"})

    inner_currtime = time.time()
    gm = GraphMatch(n_init=10, init="barycenter")
    # second graph is the one being permuted
    perm_inds = gm.fit_predict(adj_df.values, scrna_similarity)
    elapsed = time.time() - inner_currtime
    results.append({"replicate": i, "wall_time": elapsed, "stage": "Graph match"})

    perm_neuron_indices = subset_sequencing_df.columns[perm_inds][: len(adj_df)]
    perm_meta = scrna_meta.loc[perm_neuron_indices]

    for i in range(len(adj_df)):
        neuron_matches[i].append(perm_meta.index[i])

results = pd.DataFrame(results)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.stripplot(data=results, x="stage", y="wall_time", ax=ax)

print(f"{time.time() - currtime} elapsed")

#%%
perm_corr = scrna_similarity[perm_inds][:, perm_inds]
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
heatmap(
    adj_df.values, transform="simple-all", ax=axs[0], cbar=False, title="Connectome"
)
heatmap(
    perm_corr[: len(adj_df), : len(adj_df)],
    ax=axs[1],
    cbar=False,
    title=r"Cosine o scRNA (first $N$ neurons)",
)
stashfig("example-adj-cos-match")

#%% class-wise confusion matrix
uni_classes = np.unique(connectome_meta["cell_type"])
class_map = dict(zip(uni_classes, range(len(uni_classes))))
n_classes = len(uni_classes)
class_conf_mat = np.zeros((n_classes, n_classes))

for i in range(len(adj_df)):
    matches = neuron_matches[i]
    for match in matches:
        scrna_neuron_type = scrna_meta.loc[match]["Neuron_type"]
        # scrna_neuron_type_index = scrna_meta.loc[match]["CellTypeIndex"]
        connectome_neuron_type = connectome_meta.iloc[i]["cell_type"]
        # connectome_neuron_type_index = connectome_meta.iloc[i]["Var5"]
        class_conf_mat[
            class_map[connectome_neuron_type], class_map[scrna_neuron_type]
        ] += 1

class_sizes = connectome_meta.groupby("cell_type").size()
class_sizes = class_sizes.reindex(uni_classes).values
class_conf_mat /= class_sizes[:, None]
class_conf_mat /= n_trials


def outline_diagonal(n, ax, color="black", linestyle="-", linewidth=0.2, alpha=0.7):
    for i in range(n):
        low = i
        high = i + 1
        xs = [low, high, high, low, low]
        ys = [low, low, high, high, low]
        ax.plot(
            xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha
        )


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    class_conf_mat,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    square=True,
    cbar_kws=dict(shrink=0.7),
)
outline_diagonal(n_classes, ax)
ax.set(
    ylabel="Connectome class index", xlabel="scRNA class index", xticks=[], yticks=[]
)
stashfig("class-confusion-mat")

#%%
rank_conf_mat = class_conf_mat.copy()
top_n = 10
print("Neuron in connectome - Neuron in scRNA")
print("______________________________________")
print()
for i in range(top_n):
    ind = np.unravel_index(np.argmax(rank_conf_mat, axis=None), rank_conf_mat.shape)
    print(f"{uni_classes[ind[0]]} - {uni_classes[ind[1]]}")
    print()
    rank_conf_mat[ind] = 0

#%% verify that the above make sense


#%% neuron by class confusion matrix
cell_conf_mat = np.zeros((len(adj_df), n_classes))

for i in range(len(adj_df)):
    matches = neuron_matches[i]
    for match in matches:
        scrna_neuron_type = scrna_meta.loc[match]["Neuron_type"]
        cell_conf_mat[i, class_map[scrna_neuron_type]] += 1
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    cell_conf_mat,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    square=True,
    cbar_kws=dict(shrink=0.7),
)
ax.set(ylabel="True class index", xlabel="Matched class index", xticks=[], yticks=[])
stashfig("cell-confusion-mat")

#%% look at whether this is just the result of the prior on class types
scrna_class_counts = scrna_meta.groupby("Neuron_type").size()
scrna_class_counts = scrna_class_counts.sort_values(ascending=False)
scrna_match_counts = class_conf_mat.sum(axis=0)
scrna_match_counts = pd.Series(index=uni_classes, data=scrna_match_counts)
scrna_match_counts = scrna_match_counts.reindex_like(scrna_class_counts)

scrna_class_props = scrna_class_counts / scrna_class_counts.sum()
scrna_match_props = scrna_match_counts / scrna_match_counts.sum()

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.scatterplot(x=range(n_classes), y=scrna_class_props, ax=ax, label="scRNA", s=40)
sns.scatterplot(x=range(n_classes), y=scrna_match_props, ax=ax, label="Matching", s=40)
ax.set(xlabel="Cell type index", ylabel="Frequency")
stashfig("prior-frequency-vs-matching")

#%%

from graspologic.embed import ClassicalMDS
from graspologic.plot import pairplot
import colorcet as cc
from umap import UMAP

scrna_neurons = subset_sequencing_df.columns
temp_scrna_meta = scrna_meta.loc[scrna_neurons]
scrna_embed = ClassicalMDS(n_components=4).fit_transform(scrna_similarity)
labels = temp_scrna_meta["Neuron_type"].values
pairplot(scrna_embed, labels=labels, palette=cc.glasbey_light)

#%%
umapper = UMAP(n_components=3, metric="cosine", min_dist=1, n_neighbors=10)
umap_scrna_embed = umapper.fit_transform(subset_sequencing_df.T.fillna(0).values)
labels = temp_scrna_meta["Neuron_type"].values
pg = pairplot(umap_scrna_embed, labels=labels, palette=cc.glasbey_light)
pg._legend.remove()