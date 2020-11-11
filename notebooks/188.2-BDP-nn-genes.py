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

rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    # "axes.edgecolor": "lightgrey",
    # "ytick.color": "grey",
    # "xtick.color": "grey",
    # "axes.labelcolor": "dimgrey",
    # "text.color": "dimgrey",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}

set_theme(rc_dict=rc_dict, font_scale=1.25)

np.random.seed(88888)

colors = sns.color_palette("deep", 10)


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

baseline_acc = np.mean(accuracies)
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

gm = GraphMatch(n_init=20, init="barycenter")
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
    gm = GraphMatch(n_init=20, init="barycenter")
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
            left_latent, right_latent = match_latent_map[method]
            low_rank_adj = (
                left_latent[:, :n_components] @ right_latent[:, :n_components].T
            )
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
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
stashfig("acc-by-rank")

# %% [markdown]
# ##


def mlp_from_adjacency(adj, X_train, y_train):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=1,
        warm_start=True,
        solver="lbfgs",
    )
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
        ].copy()
        mlp.intercepts_[i] = adj[
            -i - 1, n_nodes_visited + n_source : n_nodes_visited + n_source + n_target
        ].copy()
        n_nodes_visited += n_source
    return mlp


def perturb_mlp(mlp, scale=0.01):
    for i, weights in enumerate(mlp.coefs_):
        n_source, n_target = weights.shape
        mlp.coefs_[i] += np.random.normal(0, scale, size=mlp.coefs_[i].shape)
        mlp.intercepts_[i] += np.random.normal(0, scale, size=mlp.intercepts_[i].shape)
    return mlp


# mlp = perturb_mlp(mlp, scale=0.2)
# mlp.warm_start = True
# mlp.max_iter = 1

n_components = 5

left_latent, right_latent = match_latent_map["matched"]
low_rank_adj = left_latent[:, :n_components] @ right_latent[:, :n_components].T

from sklearn.exceptions import ConvergenceWarning

rows = []
for replicate in range(10):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        print(replicate)
        low_rank_mlp = mlp_from_adjacency(low_rank_adj, X_train, y_train)
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.3, shuffle=True
        )
        clean_mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=1,
            warm_start=True,
            solver="lbfgs",
        )
        for method in ["Low rank", "Random"]:
            if method == "Random":
                mlp = clean_mlp
            elif method == "Low rank":
                mlp = low_rank_mlp
            for i in range(500):
                inds = np.random.choice(len(X_train), size=128, replace=False)
                mlp = mlp.fit(X_train[inds], y_train[inds])
                mlp.n_iter_ = 0
                acc = accuracy_score(y_test, mlp.predict(X_test))
                rows.append(
                    {
                        "accuracy": acc,
                        "method": method,
                        "iteration": i,
                        "replicate": replicate,
                    }
                )

results = pd.DataFrame(rows)
#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

# sns.lineplot(
#     x="iteration",
#     y="accuracy",
#     hue="method",
#     estimator=None,
#     alpha=0.3,
#     data=results,
#     ax=ax,
# )
start_palette = dict(zip(["Low rank", "Random"], colors[1:3]))
sns.lineplot(
    x="iteration",
    y="accuracy",
    hue="method",
    data=results,
    ax=ax,
    palette=start_palette,
)
# ax.set(title=f"Rank = {n_components}", )
ax.set(xlabel="Training iteration", ylabel="Accuracy")
# ax.annotate()
handles, labels = ax.get_legend_handles_labels()
labels[0] = "Initialization"
labels[1] = f"Rank = {n_components}"
ax.legend(handles=handles, labels=labels)
ax.axhline(baseline_acc, linestyle=":", linewidth=1.5, color="black")
stashfig(f"low-rank-training-continued-n_components={n_components}")

#%%
# matched_adj_bar = np.mean(np.stack(matched_adjs), axis=0)
n_components_range = np.unique(np.geomspace(1, 25, num=10, dtype=int))

rows = []
for resample in range(n_resamples):
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=True
    )
    for n_components in n_components_range:
        for i, matched_adj in enumerate(matched_adjs):
            X_matched, Y_matched, sing_vals_matched, elbow_inds_matched = embed(
                matched_adj
            )
            low_rank_adj = X_matched[:, :n_components] @ Y_matched[:, :n_components].T
            mlp = mlp_from_adjacency(low_rank_adj, X_train, y_train)
            y_pred = mlp.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            rows.append(
                {
                    "accuracy": acc,
                    "network": i,
                    "replicate": resample,
                    "n_components": n_components,
                }
            )

results = pd.DataFrame(rows)
palette = dict(
    zip(results["network"].unique(), results["network"].nunique() * [colors[0]])
)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    data=results,
    x="n_components",
    y="accuracy",
    hue="network",
    palette=palette,
    legend=False,
    alpha=0.1,
    ax=ax,
)
sns.lineplot(
    data=results,
    x="n_components",
    y="accuracy",
    palette=palette,
    legend=False,
    alpha=1,
    ci=None,
    c=colors[0],
    ax=ax,
)
ax.set(xlabel="Rank", ylabel="Accuracy")
stashfig("individual-ranks")

#%%

ase = AdjacencySpectralEmbed(
    n_components=matched_adj.shape[0], algorithm="full", diag_aug=False
)
ase.fit(matched_adj)
ase.singular_values_
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(ase.singular_values_)
ax.axvline(25, color="red")

#%%
# scree plots
rows = []
for i, adj in enumerate(matched_adjs):
    _, _, sing_vals, _ = embed(adj)
    sing_vals = sing_vals[:25]
    for j in range(len(sing_vals)):
        rows.append(
            {
                "sing_val": sing_vals[: j + 1].sum() / sing_vals.sum(),
                "network": i,
                "index": j,
            }
        )
results = pd.DataFrame(rows)

color_ind = 4

palette = dict(
    zip(results["network"].unique(), results["network"].nunique() * [colors[color_ind]])
)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(
    data=results,
    x="index",
    y="sing_val",
    hue="network",
    palette=palette,
    legend=False,
    alpha=0.2,
    ax=ax,
)
sns.lineplot(
    data=results,
    x="index",
    y="sing_val",
    palette=palette,
    legend=False,
    alpha=1,
    ci=None,
    color=colors[color_ind],
    ax=ax,
)
ax.set(xlabel="Index", ylabel="Cumulative var.")
ax.set(xlabel="Index", ylabel="Cumulative variance", yticks=[0, 0.5, 1])
stashfig("nn-scree")

#%%


from src.data import load_metagraph

mg = load_metagraph("G")
mg = mg.make_lcc()
adj = mg.adj
rows = []
_, _, sing_vals, _ = embed(adj)
for j in range(len(sing_vals)):
    rows.append(
        {
            "sing_val": sing_vals[: j + 1].sum() / sing_vals.sum(),
            "network": i,
            "index": j,
        }
    )
results = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(data=results, x="index", y="sing_val", ax=ax, color=colors[color_ind])
ax.set(xlabel="Index", ylabel="Cumulative variance", yticks=[0, 0.5, 1])
stashfig("maggot-scree")

# %%
# TODO: overparameterized model, does this do better, maybe the independent edge assumption makes more sense
# TODO: ecdf on the edges. maybe is a more complicated rather than less complicated model.
# TODO: otherwise, just sample edge weights uniformly from the marginal ECDF.
# overarching thing here is dont have a way to sample in a way that makes sense?


# # %% [markdown]
# # ##
# from sklearn.utils import check_random_state

# mlp = MLPClassifier(max_iter=1)
# mlp._validate_hyperparameters()
# mlp._validate_input(X_train, y_train, False)
# n_features = X_train.shape[1]
# n_outputs = len(np.unique(y_train))
# layer_units = [n_features] + list(hidden_layer_sizes) + [n_outputs]
# mlp._random_state = check_random_state(mlp.random_state)
# mlp._initialize(y_train.reshape((-1, 1)), layer_units)
# # mlp.fit(X_train, y_train)


# def split_data(test_size=0.3):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, shuffle=True
#     )
#     return X_train, X_test, y_train, y_test


# X = data
# y = digits.target


# def initialize_mlp(hidden_layer_sizes=(15,)):
#     mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
#     mlp._validate_hyperparameters()
#     mlp._validate_input(X, y, False)
#     n_features = X.shape[1]
#     n_outputs = len(np.unique(y))
#     layer_units = [n_features] + list(hidden_layer_sizes) + [n_outputs]
#     mlp._random_state = check_random_state(mlp.random_state)
#     mlp._initialize(y.reshape((-1, 1)), layer_units)
#     return mlp


# def train_mlp(hidden_layer_sizes=(15,), test_size=0.3, return_test_data=False):
#     X_train, X_test, y_train, y_test = split_data(test_size=test_size)
#     mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=600)
#     mlp.fit(X_train, y_train)
#     if not return_test_data:
#         return mlp
#     else:
#         return mlp, X_test, y_test


# def get_mlp_params(mlp):
#     params = []
#     for coefs in mlp.coefs_:
#         params.append(coefs.ravel())
#     for intercepts in mlp.intercepts_:
#         params.append(intercepts)
#     return np.concatenate(params)


# n_nn_samples = 100
# y_nn = np.zeros(2 * n_nn_samples)

# mlps = []
# mlp_params = []
# for i in range(n_nn_samples):
#     mlp = train_mlp()
#     mlps.append(mlp)
#     mlp_params.append(get_mlp_params(mlp))
#     y_nn[i] = 1


# for i in range(n_nn_samples):
#     mlp = initialize_mlp()
#     mlps.append(mlp)
#     mlp_params.append(get_mlp_params(mlp))

# X_nn = np.stack(mlp_params)

# # %% [markdown]
# # ##
# from sklearn.ensemble import RandomForestClassifier

# X_train, X_test, y_train, y_test = train_test_split(
#     X_nn, y_nn, test_size=0.3, shuffle=True
# )

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print(score)

# #%%

# images = digits.images.copy()

# for i, image in enumerate(images):
#     images[i] = image.T

# data_rot = images.reshape((n_samples, -1))


# # def train_mlp(hidden_layer_sizes=(15,), test_size=0.3, return_test_data=False):
# #     mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=600)
# #     mlp.fit(X_train, y_train)
# #     if not return_test_data:
# #         return mlp
# #     else:
# #         return mlp, X_test, y_test

# n_nn_samples = 100
# y_nn = np.zeros(2 * n_nn_samples)
# mlp_kwargs = dict(hidden_layer_sizes=hidden_layer_sizes, max_iter=600)

# mlps = []
# mlp_params = []
# for i in range(n_nn_samples):
#     X_train, X_test, y_train, y_test = train_test_split(
#         data, y, test_size=0.3, shuffle=True
#     )
#     mlp = MLPClassifier(**mlp_kwargs)
#     mlp.fit(X_train, y_train)
#     mlps.append(mlp)
#     mlp_params.append(get_mlp_params(mlp))
#     y_nn[i] = 1

# for i in range(n_nn_samples):
#     X_train, X_test, y_train, y_test = train_test_split(
#         data_rot, y, test_size=0.3, shuffle=True
#     )
#     mlp = MLPClassifier(**mlp_kwargs)
#     mlp.fit(X_train, y_train)
#     mlps.append(mlp)
#     mlp_params.append(get_mlp_params(mlp))

# X_nn = np.stack(mlp_params)

# # %%
# X_train, X_test, y_train, y_test = train_test_split(
#     X_nn, y_nn, test_size=0.3, shuffle=True
# )

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print(score)

# # %% [markdown]
# # ##
# from sklearn.linear_model import SGDClassifier

# log_reg_cls = SGDClassifier(loss="log")
# log_reg_cls.fit(X_train, y_train)
# y_pred = log_reg_cls.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print(score)
# # probs =

# # %% [markdown]
# # ##
# from sklearn.decomposition import PCA
# from graspologic.plot import pairplot

# X_nn_pca = PCA(n_components=4).fit_transform(X_nn)

# # X_nn_latent += np.random.uniform()
# plt.figure()
# sns.scatterplot(X_nn_pca[:, 0], X_nn_pca[:, 1], hue=y_nn, legend=False, s=10)

# pairplot(X_nn_pca, labels=y_nn)

# # %% [markdown]
# # ##
# from sklearn.manifold import Isomap

# X_nn_iso = Isomap(n_components=4, metric="cosine").fit_transform(X_nn)

# pairplot(X_nn_iso, labels=y_nn)

# # from umap import UMAP

# # X_nn_umap = UMAP(n_components=2, min_dist=1, n_neighbors=5).fit_transform(X_nn)
# # plt.figure()
# # sns.scatterplot(X_nn_umap[:, 0], X_nn_umap[:, 1], hue=y_nn, legend=False)

# #%%
# # TODO going back to the idea of trying to find the distribution of weights within a class...
