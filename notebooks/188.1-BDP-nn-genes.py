#%% [markdown]
## Modeling the distribution of neural networks
# > An attempt to characterize the distribution of neural network parameters that are learned on the same task

# - toc: true
# - badges: true
# - categories: [pedigo, graspologic]
# - hide: false
# - search_exclude: false
#%%
# collapse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.plot import heatmap, pairplot
from matplotlib.transforms import blended_transform_factory
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from graspologic.embed import AdjacencySpectralEmbed, select_dimension
from graspologic.match import GraphMatch
from src.io import savefig
from src.visualization import set_theme

FNAME = os.path.basename(__file__)[:-3]

set_theme()
np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


#%% [markdown]
### Trying to learn distributions on NNs trained on the same task

#%% [markdown]
#### Example training and predicted labels
# Adapted from Sklearn docs

#%%
# collapse
# The digits dataset
digits = load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(digits.images, digits.target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label, fontsize="x-small")

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001)
classifier = MLPClassifier()

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

images_and_predictions = list(zip(digits.images[n_samples // 2 :], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Prediction: %i" % prediction, fontsize="x-small")

plt.show()
#%% [markdown]
#### Training multiple NNs on the same task

#%%
# collapse

hidden_layer_sizes = (15,)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

n_replicates = 8
adjs = []
all_biases = []
all_weights = []
accuracies = []
mlps = []
for i in range(n_replicates):
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=True
    )
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=600)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    weights_by_layer = mlp.coefs_
    all_biases.append(mlp.intercepts_)
    all_weights.append(mlp.coefs_)
    accuracies.append(acc)
    mlps.append(mlp)
    print(f"Test accuracy score for NN {i+1}: {acc}")

    n_nodes = 0
    for weights in weights_by_layer:
        n_source, n_target = weights.shape
        n_nodes += n_source
    n_nodes += n_target
    n_nodes += len(hidden_layer_sizes) + 1
    adj = np.zeros((n_nodes, n_nodes))

    n_nodes_visited = 0
    for i, weights in enumerate(weights_by_layer):
        n_source, n_target = weights.shape
        adj[
            n_nodes_visited : n_nodes_visited + n_source,
            n_nodes_visited + n_source : n_nodes_visited + n_source + n_target,
        ] = weights
        adj[
            -i - 1, n_nodes_visited + n_source : n_nodes_visited + n_source + n_target
        ] = mlp.intercepts_[i]
        n_nodes_visited += n_source

    adjs.append(adj)


all_biases = [np.concatenate(b) for b in all_biases]
all_biases = np.stack(all_biases).T

all_weights = [[w.ravel() for w in weights] for weights in all_weights]
all_weights = [np.concatenate(w) for w in all_weights]
all_weights = np.stack(all_weights).T
#%% [markdown]
#### Plotting the adjacency matrices for each NN
# %%
# collapse
vmax = max(map(np.max, adjs))
vmin = min(map(np.min, adjs))
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for i, ax in enumerate(axs.ravel()):
    heatmap(adjs[i], cbar=False, vmin=vmin, vmax=vmax, ax=ax, title=f"NN {i + 1}")
fig.suptitle("Adjacency matrices", fontsize="large", fontweight="bold")
plt.tight_layout()

stashfig("multi-nn-adjs")
#%% [markdown]
#### Trying to model the weights with something simple
# I make a new NN where the weights are set to the simple average of weights across all of
# the NN that I fit and see how it performs.
# %%
# collapse
adj_bar = np.mean(np.stack(adjs), axis=0)


def mlp_from_adjacency(adj, X_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mlp.fit(
            X_train, y_train
        )  # dummy fit, just to set parameters like shape of input/output
    n_nodes_visited = 0
    for i, weights in enumerate(mlp.coefs_):
        n_source, n_target = weights.shape
        mlp.coefs_[i] = adj[
            n_nodes_visited : n_nodes_visited + n_source,
            n_nodes_visited + n_source : n_nodes_visited + n_source + n_target,
        ]
        mlp.intercepts_[i] = adj[
            -i - 1, n_nodes_visited + n_source : n_nodes_visited + n_source + n_target
        ]
        n_nodes_visited += n_source
    return mlp


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.7, shuffle=True
)
mlp = mlp_from_adjacency(adj_bar, X_train, y_train)
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy score for NN with mean weights: {acc}")

#%% [markdown]
### Why I think this doesn't work, and what we might be able to do about it
# There are two big issues as I see it:
# - Permuation nonidentifiability in the model
# - Lack of edge-edge dependence structure in our models
#
# Below I investigate the first issue, haven't thought about what to do for the second
#%% [markdown]
#### Plotting the learned weights against each other
# If each network has $d$ free weight parameters, and there are $T$ of them,
# I form the $T$ by $d$ matrix of weights per neural network, and then plot each network's
# weights against each other.
#%%
# collapse


def corrplot(x, y, *args, ax=None, fontsize="xx-small", **kwargs):
    if ax is None:
        ax = plt.gca()
    pearsons, _ = pearsonr(x, y)
    spearmans, _ = spearmanr(x, y)
    text = r"$\rho_p: $" + f"{pearsons:.3f}\n"
    text += r"$\rho_s: $" + f"{spearmans:.3f}"
    ax.text(1, 1, text, ha="right", va="top", transform=ax.transAxes, fontsize=fontsize)


pg = pairplot(
    all_weights,
    alpha=0.1,
    title="Weights",
    col_names=[f"NN {i+1}" for i in range(all_weights.shape[1])],
)
pg.map_offdiag(corrplot)
stashfig(
    "weight-pairplot",
)

#%% [markdown]
#### Can graph matching fix the permutation nonidentifiability?
# Given one neural network architecture, one could permute the labels/orders of the hidden
# units, and the network would be functionally equivalent. This means that when comparing the
# architectures of two learned neural networks against each other, there is a nonidentifiability
# problem caused by this arbitrary permutation. Even if we imagine that two neural networks
# learned the exact same weights, they are unlikely to look similar at a glance because it
# is unlikely they learned the same weights and the same permutation. Let's see if graph matching
# can help resolve this nonidentifiability.
# %%
# collapse
heatmap_kws = dict(vmin=vmin, vmax=vmax, cbar=False)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
heatmap(adjs[0], ax=axs[0], title="NN 1 pre-GM", **heatmap_kws)
heatmap(adjs[1], ax=axs[1], title="NN 2 pre-GM", **heatmap_kws)
heatmap(adjs[0] - adjs[1], ax=axs[2], title="Difference", **heatmap_kws)
stashfig("pre-gm-adjs")

seeds = np.concatenate(
    (
        np.arange(data.shape[1]),
        np.arange(len(adj) - 10 - len(hidden_layer_sizes) - 1, len(adj)),
    )
)

gm = GraphMatch(n_init=20, init_method="barycenter")
gm.fit(adjs[0], adjs[1], seeds_A=seeds, seeds_B=seeds)
perm_inds = gm.perm_inds_
adj_1_matched = adjs[1][np.ix_(perm_inds, perm_inds)].copy()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
heatmap(adjs[0], ax=axs[0], title="NN 1 post-GM", **heatmap_kws)
heatmap(adj_1_matched, ax=axs[1], title="NN 2 post-GM", **heatmap_kws)
heatmap(adjs[0] - adj_1_matched, ax=axs[2], title="Difference", **heatmap_kws)
stashfig("post-gm-adjs")

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
ax = axs[0]
sns.scatterplot(adjs[0].ravel(), adjs[1].ravel(), ax=ax, alpha=0.3, linewidth=0, s=15)
corrplot(adjs[0].ravel(), adjs[1].ravel(), ax=ax, fontsize="medium")
ax.set(
    title="Weights pre-GM",
    xticks=[],
    yticks=[],
    xlabel="NN 1 weights",
    ylabel="NN 2 weights",
)
ax.axis("equal")
ax = axs[1]
sns.scatterplot(
    adjs[0].ravel(), adj_1_matched.ravel(), ax=ax, alpha=0.3, linewidth=0, s=15
)
corrplot(adjs[0].ravel(), adj_1_matched.ravel(), ax=ax, fontsize="medium")
ax.set(
    title="Weights post-GM",
    xticks=[],
    yticks=[],
    xlabel="NN 1 weights",
    ylabel="NN 2 weights",
)
ax.axis("equal")
stashfig("pre-post-weights-gm")

#%% [markdown]
### Unraveling the nonidentifiability with GM
# I match the weights in each network to the best performing one using graph matching.
#
# NB: the way I'm doing this is more convenient but probably dumb, really should just
# be matching on a per-hidden-layer basis. But in this case I have one hidden layer and
# the others are seeds so it doesn't matter.
#%%
# collapse
best_model_ind = np.argmax(accuracies)
best_adj = adjs[best_model_ind]
matched_adjs = []

for i, adj in enumerate(adjs):
    gm = GraphMatch(n_init=20, init_method="barycenter")
    gm.fit(best_adj, adj, seeds_A=seeds, seeds_B=seeds)
    perm_inds = gm.perm_inds_
    matched_adj = adj[np.ix_(perm_inds, perm_inds)].copy()
    matched_adjs.append(matched_adj)

#%% [markdown]
#### All pairwise weight comparisons after matching
# We see that the correlation in weights improves somewhat, though they are still not
# highly correlated.
#%%
# collapse
all_matched_weights = [a.ravel() for a in matched_adjs]
all_matched_weights = np.stack(all_matched_weights, axis=1)
all_matched_weights = all_matched_weights[
    np.linalg.norm(all_matched_weights, axis=1) != 0, :
]
all_matched_weights.shape

pg = pairplot(
    all_matched_weights,
    alpha=0.1,
    title="Matched weights",
    col_names=[f"NN {i+1}" for i in range(all_matched_weights.shape[1])],
)
pg.map_offdiag(corrplot)
stashfig(
    "matched-weight-pairplot",
)

#%% [markdown]
#### Using $\bar{A}_{matched}$ as the weight matrix
# I take the mean of the weights after matching, and ask how well this mean weight matrix
# performs when converted back to a neural net.
#%%
# collapse
matched_adj_bar = np.mean(np.stack(matched_adjs), axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.7, shuffle=True
)
mlp = mlp_from_adjacency(matched_adj_bar, X_train, y_train)
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy score for NN with average adjacency (matched): {acc}")

#%% [markdown]
#### Decomposing the matched and unmatched adjacency matrices
#%%
# collapse


def embed(A):
    ase = AdjacencySpectralEmbed(n_components=len(A), algorithm="full")
    X, Y = ase.fit_transform(A)
    elbow_inds, _ = select_dimension(A, n_elbows=4)
    elbow_inds = np.array(elbow_inds)
    return X, Y, ase.singular_values_, elbow_inds


def screeplot(sing_vals, elbow_inds, color=None, ax=None, label=None, linestyle="-"):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(
        range(1, len(sing_vals) + 1),
        sing_vals,
        color=color,
        label=label,
        linestyle=linestyle,
    )
    plt.scatter(
        elbow_inds,
        sing_vals[elbow_inds - 1],
        marker="x",
        s=50,
        zorder=10,
        color=color,
    )
    ax.set(ylabel="Singular value", xlabel="Index")
    return ax


X_matched, Y_matched, sing_vals_matched, elbow_inds_matched = embed(matched_adj_bar)
X_unmatched, Y_unmatched, sing_vals_unmatched, elbow_inds_unmatched = embed(adj_bar)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
screeplot(sing_vals_matched, elbow_inds_matched, ax=ax, label="matched")
screeplot(
    sing_vals_unmatched, elbow_inds_unmatched, ax=ax, label="unmatched", linestyle="--"
)
ax.legend()
stashfig("screeplot-adj-bars")

#%% [markdown]
#### Plotting accuracy as a function of adjacency rank
#%%
# collapse
match_latent_map = {
    "matched": (X_matched, Y_matched),
    "unmatched": (X_unmatched, Y_unmatched),
}

n_components_range = np.unique(
    np.geomspace(1, len(matched_adj_bar) + 1, num=10, dtype=int)
)

rows = []
n_resamples = 8
for resample in range(n_resamples):
    for n_components in n_components_range:
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.7, shuffle=True
        )
        for method in ["matched", "unmatched"]:
            X, Y = match_latent_map[method]
            low_rank_adj = X[:, :n_components] @ Y[:, :n_components].T
            mlp = mlp_from_adjacency(low_rank_adj, X_train, y_train)
            y_pred = mlp.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            rows.append({"accuracy": acc, "rank": n_components, "type": method})
results = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    x="rank",
    y="accuracy",
    data=results,
    style="type",
    hue="type",
    ax=ax,
    # markers=["o", "o"],
)
results["jitter_rank"] = results["rank"] + np.random.uniform(
    -1, 1, size=len(results["rank"])
)
sns.scatterplot(
    x="jitter_rank", y="accuracy", data=results, hue="type", ax=ax, s=10, legend=False
)
ax.set(yticks=[0.2, 0.4, 0.6, 0.8], ylim=(0, 0.82), xlabel="Rank", ylabel="Accuracy")
ax.axhline(1 / 10, linestyle=":", linewidth=1.5, color="black")

ax.text(
    1,
    1 / 10,
    "Chance",
    ha="right",
    va="bottom",
    transform=blended_transform_factory(ax.transAxes, ax.transData),
    fontsize="small",
)
stashfig("acc-by-rank")

# %%
# TODO: overparameterized model, does this do better, maybe the independent edge assumption makes more sense
# TODO: ecdf on the edges. maybe is a more complicated rather than less complicated model.
# TODO: otherwise, just sample edge weights uniformly from the marginal ECDF.
