#%% [markdown]
## Matching ALPNs
#> Exploring the use of quadradic assignment problem solvers to match neuron morphologies.
#- toc: true
#- badges: true
#- categories: [pedigo, graspologic, graph-matching, drosophila]
#- hide: false
#- search_exclude: false
# %% [markdown]
# ## Load the data
#%%
# collapse
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from graspologic.match import GraphMatch
from graspologic.match.qap import _doubly_stochastic
from src.visualization import adjplot

t0 = time.time()

sns.set_context("talk")

meta_path = "ALPN_crossmatching/data/meta.csv"
nblast_path = "ALPN_crossmatching/data/nblast_scores.csv"

meta = pd.read_csv(meta_path, index_col=0)
meta = meta.set_index("id")
meta["label"].fillna("unk", inplace=True)
nblast_scores = pd.read_csv(nblast_path, index_col=0, header=0)
nblast_scores.columns = nblast_scores.columns.astype(int)

#%% [markdown]
# ## Look at the data
#%%
# collapse
adjplot(
    nblast_scores.values,
    meta=meta,
    sort_class=["source"],
    item_order="lineage",
    colors="lineage",
    cbar_kws=dict(shrink=0.7),
)

#%%
# collapse
adjplot(
    nblast_scores.values,
    meta=meta,
    sort_class=["lineage"],
    item_order="source",
    colors="source",
    cbar_kws=dict(shrink=0.7),
)

# %% [markdown]
# ### Plot the distribution of pairwise scores
#%%
# collapse
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.histplot(nblast_scores.values.ravel(), element="step", stat="density")

# %% [markdown]
# ## Split the NBLAST scores by dataset
#%%
# collapse
datasets = ["FAFB(L)", "FAFB(R)"]
dataset1_meta = meta[meta["source"] == datasets[0]]
dataset2_meta = meta[meta["source"] == datasets[1]]

dataset1_ids = dataset1_meta.index
dataset1_intra = nblast_scores.loc[dataset1_ids, dataset1_ids].values

dataset2_ids = dataset2_meta.index
dataset2_intra = nblast_scores.loc[dataset2_ids, dataset2_ids].values

# TODO use these also via the linear term in GMP
dataset1_to_dataset2 = nblast_scores.loc[dataset1_ids, dataset2_ids].values
dataset2_to_dataset1 = nblast_scores.loc[dataset2_ids, dataset1_ids].values

#%% [markdown]
# ## Plot the NBLAST scores before alignment
# %%
# collapse
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
adjplot(dataset1_intra, cbar=False, ax=axs[0])
adjplot(dataset2_intra, cbar=False, ax=axs[1])

#%% [markdown]
# ## Run the NBLAST score matching without using any prior information
#%%
# collapse
gm = GraphMatch(
    n_init=100,
    init="barycenter",
    max_iter=200,
    shuffle_input=True,
    eps=1e-5,
    gmp=True,
    padding="naive",
)

gm.fit(dataset1_intra, dataset2_intra)
perm_inds = gm.perm_inds_
print(f"Matching objective function: {gm.score_}")

#%%
# collapse
dataset2_intra_matched = dataset2_intra[perm_inds][:, perm_inds][: len(dataset1_ids)]
dataset2_meta_matched = dataset2_meta.iloc[perm_inds][: len(dataset1_ids)]

#%% [markdown]
# ### Plot the NBLAST scores after alignment
#%%
# collapse
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
adjplot(dataset1_intra, cbar=False, ax=axs[0])
adjplot(dataset2_intra_matched, cbar=False, ax=axs[1])

#%% [markdown]
# ### Peek at the metadata after alignment
#%%
# collapse
dataset1_meta
#%%
# collapse
dataset2_meta_matched
#%% [markdown]
# ### Plot confusion matrices for the predicted matching
#%%
# collapse


def confusionplot(
    labels1,
    labels2,
    ax=None,
    figsize=(10, 10),
    xlabel="",
    ylabel="",
    title="Confusion matrix",
    annot=True,
    add_diag_proportion=True,
    **kwargs,
):
    unique_labels = np.unique(list(labels1) + list(labels2))
    conf_mat = confusion_matrix(labels1, labels2, labels=unique_labels, normalize=None)
    conf_mat = pd.DataFrame(data=conf_mat, index=unique_labels, columns=unique_labels)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(
        conf_mat,
        ax=ax,
        square=True,
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(shrink=0.6),
        annot=annot,
        fmt="d",
        mask=conf_mat == 0,
        **kwargs,
    )
    ax.set(ylabel=ylabel, xlabel=xlabel)
    if add_diag_proportion:
        on_diag = np.trace(conf_mat.values) / np.sum(conf_mat.values)
        title += f" ({on_diag:0.2f} correct)"
    ax.set_title(title, fontsize="large", pad=10)
    return ax


#%% [markdown]
# #### Confusion matrix for neuron type
#%%
# collapse
confusionplot(
    dataset1_meta["ntype"],
    dataset2_meta_matched["ntype"],
    ylabel=datasets[0],
    xlabel=datasets[1],
    title="Type confusion matrix",
)
#%% [markdown]
# #### Confusion matrix for lineage
#%%
# collapse
confusionplot(
    dataset1_meta["lineage"],
    dataset2_meta_matched["lineage"],
    ylabel=datasets[0],
    xlabel=datasets[1],
    title="Lineage confusion matrix",
)
#%% [markdown]
# #### Confusion matrix for label
# NB: There are many "unknown" in the label category, which was messinig up the color
# palette here, so I clipped the color range at the maximum for the non-unknown
# categories. It could be skewing the accuracy thought (e.g. unk matched to unk counts
# as correct).
#%%
# collapse
labels1 = dataset1_meta["label"]
dataset1_vmax = labels1.value_counts()[1:].max()
labels2 = dataset2_meta_matched["label"]
dataset2_vmax = labels2.value_counts()[1:].max()
vmax = max(dataset1_vmax, dataset2_vmax)


confusionplot(
    labels1,
    labels2,
    ylabel=datasets[0],
    xlabel=datasets[1],
    title="Label confusion matrix",
    annot=False,
    vmax=vmax,
    xticklabels=False,
    yticklabels=False,
)

#%% [markdown]
# #### Accuracy for the above, ignoring unclear/unknown
#%%
# collapse
unique_labels = np.unique(list(labels1) + list(labels2))
conf_mat = confusion_matrix(labels1, labels2, labels=unique_labels, normalize=None)
conf_mat = pd.DataFrame(data=conf_mat, index=unique_labels, columns=unique_labels)
conf_mat = conf_mat.iloc[:-5, :-5]  # hack to ignore anything "unclear"
on_diag = np.trace(conf_mat.values) / np.sum(conf_mat.values)
print(f"{on_diag:.2f}")

#%% [markdown]
# ## Matching with a prior
# Here we try to use the group label as a soft prior (not a hard constraint) on the matching
# proceedure.
#
# We do this by initializing from the "groupycenter" as opposed to the barycenter of the
# doubly stochastic matrices.

#%% [markdown]
# ### Construct an initialization from the lineages
#%%
# collapse

groups1 = dataset1_meta["lineage"]
groups2 = dataset2_meta["lineage"]

unique_groups = np.unique(list(groups1) + list(groups2))

n = len(groups2)  # must be the size of the larger
D = np.zeros((n, n))

group = unique_groups[-1]
layers = []
for group in unique_groups:
    inds1 = np.where(groups1 == group)[0]
    inds2 = np.where(groups2 == group)[0]
    not_inds1 = np.where(groups1 != group)[0]
    not_inds2 = np.where(groups2 != group)[0]
    n_groups = [len(inds1), len(inds2)]
    argmax_n_group = np.argmax(n_groups)
    max_n_group = n_groups[argmax_n_group]
    if min(n_groups) != 0:
        val = 1 / max_n_group
        layer = np.zeros((n, n))
        layer[np.ix_(inds1, inds2)] = val
        D += layer
    # if n_groups[0] != n_groups[1]:
    #     if argmax_n_group == 1:
    #         # then the column sums will be less than 0
    #         col_sum = layer[np.ix_(inds1, inds2)].sum(axis=0).mean()
    #         layer[np.ix_(not_inds1, inds2)] = 1 / len(not_inds1) * (1 - col_sum)

    #     elif argmax_n_group == 0:
    #         # then the row sums  will be less than 0
    #         row_sum = layer[np.ix_(inds1, inds2)].sum(axis=1).mean()
    #         layer[np.ix_(inds1, not_inds2)] = 1 / len(not_inds2) * (1 - row_sum)_d

    #
    #
    #     D[np.ix_(inds1, inds2)] = val

    #     # row_sums = np.sum(layer[inds1], axis=1).mean()
    #     # col_sums = np.sum(layer[:, inds2], axis=0).mean()
    #     layers.append(layer)


# D[:, D.sum(axis=0) == 0] = 1 / n
# D[D.sum(axis=1) == 0] = 1 / n
D += 1 / (n ** 2)  # need to add somthing small for sinkhorn to converge
D0 = _doubly_stochastic(D)

#%% [markdown]
# ### Run matching from the informed initialization
#%%
# collapse
gm = GraphMatch(
    n_init=100,
    init=D0,
    max_iter=200,
    shuffle_input=True,
    eps=1e-5,
    gmp=True,
    padding="naive",
)

gm.fit(dataset1_intra, dataset2_intra)
perm_inds = gm.perm_inds_
print(f"Matching objective function: {gm.score_}")

#%%
# collapse
dataset2_intra_matched = dataset2_intra[perm_inds][:, perm_inds][: len(dataset1_ids)]
dataset2_meta_matched = dataset2_meta.iloc[perm_inds][: len(dataset1_ids)]

#%% [markdown]
# ### Plot confusion matrices for the predicted matching started from the prior
#%% [markdown]
# #### Confusion matrix for neuron type
#%%
# collapse
confusionplot(
    dataset1_meta["ntype"],
    dataset2_meta_matched["ntype"],
    ylabel=datasets[0],
    xlabel=datasets[1],
    title="Type confusion matrix",
)
#%% [markdown]
# #### Confusion matrix for lineage
#%%
# collapse
confusionplot(
    dataset1_meta["lineage"],
    dataset2_meta_matched["lineage"],
    ylabel=datasets[0],
    xlabel=datasets[1],
    title="Lineage confusion matrix",
)
#%% [markdown]
# #### Confusion matrix for label
#%%
# collapse
labels1 = dataset1_meta["label"]
dataset1_vmax = labels1.value_counts()[1:].max()
labels2 = dataset2_meta_matched["label"]
dataset2_vmax = labels2.value_counts()[1:].max()
vmax = max(dataset1_vmax, dataset2_vmax)


confusionplot(
    labels1,
    labels2,
    ylabel=datasets[0],
    xlabel=datasets[1],
    title="Label confusion matrix",
    annot=False,
    vmax=vmax,
    xticklabels=False,
    yticklabels=False,
)

#%% [markdown]
# ## Observations/notes
# - Matching accuracy looked worse when I tried random initializations instead of barycenter
# - Open question of what to do with the weights themselves, I was expecting to have to
#   use pass to ranks or some other transform but the raw scores seemed to work fairly
#   well
# - 'VUMa2' is a lineage in one FAFB and not the other hemisphere
# - solution using my groupycenter thing doesn't seem that different. possible that the
#   barycenter initialization finds a similar score/matching?

# %% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
