#%% [markdown]
## Embedding graphs with covariates
# > Some experiments on a real dataset of interest using CASE and MASE

# - toc: true
# - badges: true
# - categories: [pedigo, graspologic]
# - hide: true
# - search_exclude: true
#%% [markdown]
### Preliminaries
#%%
# collapse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from umap import UMAP

from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspologic.plot import pairplot
from graspologic.utils import get_lcc, pass_to_ranks, to_laplace
from src.io import savefig
from src.visualization import adjplot, matrixplot, set_theme

set_theme()

FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


#%%
# collapse
data_dir = Path("maggot_models/data/raw/OneDrive_1_10-21-2020")
covariate_loc = data_dir / "product_node_embedding.csv"
edges_loc = data_dir / "product_edges.csv"
category_loc = data_dir / "partition_mapping.csv"

covariate_df = pd.read_csv(covariate_loc, index_col=0, header=None).sort_index()
meta_df = pd.read_csv(category_loc, index_col=0).sort_index()
assert (covariate_df.index == meta_df.index).all()

edges_df = pd.read_csv(edges_loc).sort_index()
g = nx.from_pandas_edgelist(edges_df, edge_attr="weight", create_using=nx.DiGraph)
adj = nx.to_numpy_array(g, nodelist=meta_df.index)

#%%
# collapse
print(f"Number of vertices (original): {len(adj)}")

make_lcc = False
if make_lcc:
    adj, keep_inds = get_lcc(adj, return_inds=True)
    print(f"Number of vertices (lcc): {len(adj)}")
else:
    # HACK need to throw out some entire classes here that have very few members
    y = meta_df["cat_id"]
    unique, inv, count = np.unique(y, return_inverse=True, return_counts=True)
    low_count = count < 5
    print(f"Removing categories with fewer than 5 examples: {unique[low_count]}")
    keep_inds = ~np.isin(inv, unique[low_count])
    adj = adj[np.ix_(keep_inds, keep_inds)]
    print(f"Number of vertices (small classes removed): {len(adj)}")
meta_df = meta_df.iloc[keep_inds]
covariate_df = covariate_df.iloc[keep_inds]
X = covariate_df.values

#%%
# collapse
colors = sns.color_palette("deep")
palette = dict(zip(np.unique(meta_df["cat_id"]), colors))

#%% [markdown]
### Adjacency matrix (sorted by category)
#%%
# collapse
adjplot(
    pass_to_ranks(adj),
    plot_type="scattermap",
    meta=meta_df,
    colors="cat_id",
    sort_class="cat_id",
    palette=palette,
    title=r"Adjacency matrix ($A$)",
)
#%%
# collapse
matrixplot(
    X,
    row_meta=meta_df,
    row_colors="cat_id",
    row_sort_class="cat_id",
    row_palette=palette,
    title=r"Metadata ($X$)",
)
#%% [markdown]
### R-LSE
#%%
# collapse
lse = LaplacianSpectralEmbed(form="R-DAD")
embedding = lse.fit_transform(pass_to_ranks(adj))
pairplot(embedding[0], labels=meta_df["cat_id"].values, palette=palette)

#%%
# collapse
concat_embedding = np.concatenate(embedding, axis=1)

umapper = UMAP(min_dist=0.7, metric="cosine")
umap_embedding = umapper.fit_transform(concat_embedding)

plot_df = pd.DataFrame(
    data=umap_embedding,
    columns=[f"umap_{i}" for i in range(umap_embedding.shape[1])],
    index=meta_df.index,
)
plot_df["cat_id"] = meta_df["cat_id"]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=plot_df,
    x="umap_0",
    y="umap_1",
    s=20,
    alpha=0.7,
    hue="cat_id",
    palette=palette,
    ax=ax,
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.set_title("UMAP o R-LSE")
ax.axis("off")

#%% [markdown]
### CASE
# %%
# collapse
L = to_laplace(pass_to_ranks(adj), form="R-DAD")  # D_{tau}^{-1/2} A D_{tau}^{-1/2}
X = covariate_df.values


def build_case_matrix(L, X, alpha, method="assort"):
    if method == "assort":
        L_case = L + alpha * X @ X.T
    elif method == "nonassort":
        L_case = L @ L.T + alpha * X @ X.T
    elif method == "cca":  # doesn't make sense here, I don't thinks
        L_case = L @ X
    return L_case


def fit_case(L, X, alpha, method="assort", n_components=None):
    L_case = build_case_matrix(L, X, alpha, method=method)
    case_embedder = AdjacencySpectralEmbed(
        n_components=n_components, check_lcc=False, diag_aug=False, concat=True
    )
    case_embedding = case_embedder.fit_transform(L_case)
    return case_embedding


n_components = 8
alphas = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
methods = ["assort", "nonassort"]
case_by_params = {}
umap_by_params = {}
for method in methods:
    for alpha in alphas:
        case_embedding = fit_case(L, X, alpha, method=method, n_components=n_components)
        umapper = UMAP(min_dist=0.7, metric="cosine")
        umap_embedding = umapper.fit_transform(case_embedding)
        case_by_params[(method, alpha)] = case_embedding
        umap_by_params[(method, alpha)] = umap_embedding

#%% [markdown]
# Plot each of the embeddings
#%%
# collapse
fig, axs = plt.subplots(
    len(methods), len(alphas), figsize=5 * np.array([len(alphas), len(methods)])
)
for i, method in enumerate(methods):
    for j, alpha in enumerate(alphas):
        ax = axs[i, j]
        umap_embedding = umap_by_params[(method, alpha)]
        plot_df = pd.DataFrame(
            data=umap_embedding,
            columns=[f"umap_{c}" for c in range(umap_embedding.shape[1])],
            index=meta_df.index,
        )
        plot_df["cat_id"] = meta_df["cat_id"]
        sns.scatterplot(
            data=plot_df,
            x="umap_0",
            y="umap_1",
            s=20,
            alpha=0.7,
            hue="cat_id",
            palette="deep",
            ax=ax,
        )
        ax.get_legend().remove()

        # ax.axis("off")
        ax.set(xticks=[], yticks=[], ylabel="", xlabel="")
        ax.set_title(r"$\alpha$ = " + f"{alpha}")

axs[0, -1].legend(bbox_to_anchor=(1, 1), loc="upper left", title="Category")
axs[0, 0].set_ylabel("CASE (assortative)")
axs[1, 0].set_ylabel("CASE (non-assortative)")
fig.suptitle("UMAP o CASE embeddings")
stashfig("casc-umaps")


#%% [markdown]
### A simple classifier on the embeddings
#%%
# collapse
classifier = KNeighborsClassifier(n_neighbors=5)
y = meta_df["cat_id"].values

rows = []
for method in methods:
    for alpha in alphas:
        X = case_by_params[(method, alpha)]
        cval_scores = cross_val_score(classifier, X, y=y)
        for score in cval_scores:
            rows.append({"score": score, "alpha": alpha, "method": method})
results = pd.DataFrame(rows)

#%%
# collapse
x_range = np.array(alphas)
x_half_bin = 0.5 * (x_range[1] - x_range[0])

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
results["jitter_alpha"] = results["alpha"] + np.random.uniform(
    -0.0025, 0.0025, size=len(results)
)
method_palette = dict(zip(np.unique(results["method"]), colors))
sns.lineplot(
    x="alpha",
    hue="method",
    y="score",
    data=results,
    ax=ax,
    ci=None,
    palette=method_palette,
)
sns.scatterplot(
    x="jitter_alpha",
    hue="method",
    y="score",
    data=results,
    ax=ax,
    palette=method_palette,
)
ax.set(
    ylabel="Accuracy",
    xlabel=r"$\alpha$",
    title="5-fold cross validation, KNN o CASE (with isolates)",
)
handles, labels = ax.get_legend_handles_labels()
handles = handles[3:]
labels = labels[3:]
labels[0] = "Method"
ax.get_legend().remove()
ax.legend(
    bbox_to_anchor=(
        1,
        1,
    ),
    loc="upper left",
    handles=handles,
    labels=labels,
)
for i, x in enumerate(x_range):
    if i % 2 == 0:
        ax.axvspan(
            x - x_half_bin,
            x + x_half_bin,
            color="lightgrey",
            alpha=0.3,
            linewidth=0,
            zorder=-1,
        )
stashfig(f"knn-case-lcc={make_lcc}")

#%% [markdown]
### MASE


#%%

# def mase(ase, Y):

# #Raw mase: svd( concatenate( ASE(A) , SVD(Y Y') ).

# #ase: nx1; Y: nx1

# u_y, s_y, vh_y = np.linalg.svd(Y @ Y.transpose())

# mase_input = np.concatenate( (ase, u_y[:,0:1]), axis=-1)

# mase_raw, _, _ = np.linalg.svd(mase_input)

# return mase_raw[:,0:2] #dim=2

#%%
_, lcc_inds = get_lcc(adj, return_inds=True)
not_lcc_inds = np.setdiff1d(np.arange(len(adj)), lcc_inds)

# just pick one CASE embedding
method = "assort"
alpha = 0.02
case_embedding = case_by_params[(method, alpha)]

classifier = KNeighborsClassifier(n_neighbors=5)

y = meta_df["cat_id"].values

y_train = y[lcc_inds]
y_test = y[not_lcc_inds]
X_train = case_embedding[lcc_inds]
X_test = case_embedding[not_lcc_inds]

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)
incorrect = y_test != y_pred

# umap_embedding = umap_by_params[(method, alpha)]

# pad_prop = 0.05
# mins = X.min(axis=0)
# maxs = X.max(axis=1)
# extent = maxs - mins
# mins -= extent * pad_prop
# maxs += extent * pad_prop
# step = 0.01
# xx, yy = np.meshgrid(
#     np.arange(mins[0], maxs[0], step), np.arange(mins[1], maxs[1], step)
# )

umapper = UMAP(n_neighbors=10, min_dist=0.8, metric="cosine", negative_sample_rate=30)
umap_embedding = umapper.fit_transform(case_embedding)

# pad_prop = 0.05
# mins = umap_embedding.min(axis=0)
# maxs = umap_embedding.max(axis=0)
# extent = maxs - mins
# mins -= extent * pad_prop
# maxs += extent * pad_prop
# step = 0.5
# xx, yy = np.meshgrid(
#     np.arange(mins[0], maxs[0], step), np.arange(mins[1], maxs[1], step)
# )
# plot_mesh = np.c_[xx.ravel(), yy.ravel()]
# print(len(plot_mesh))
# native_plot_mesh = umapper.inverse_transform(plot_mesh)
# print("transformed")
# y_plot_mesh = classifier.predict(native_plot_mesh)
# plot_mesh_labels = y_plot_mesh.reshape(xx.shape)
# from matplotlib.colors import ListedColormap

# cmap = ListedColormap(list(map(palette.get, np.unique(y))))

plot_df = pd.DataFrame(
    data=umap_embedding,
    columns=[f"umap_{c}" for c in range(umap_embedding.shape[1])],
    index=meta_df.index,
)
plot_df["cat_id"] = meta_df["cat_id"]
plot_df["in_lcc"] = False
plot_df.loc[plot_df.index[lcc_inds], "in_lcc"] = True
plot_df["correct"] = True
plot_df.loc[plot_df.index[not_lcc_inds[incorrect]], "correct"] = False

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(
    data=plot_df[plot_df["in_lcc"]],
    x="umap_0",
    y="umap_1",
    s=20,
    alpha=0.1,
    hue="cat_id",
    palette=palette,
    ax=ax,
)
markers = dict(zip([True, False], ["o", "X"]))
sns.scatterplot(
    data=plot_df[~plot_df["in_lcc"]],
    x="umap_0",
    y="umap_1",
    s=30,
    alpha=0.9,
    hue="cat_id",
    palette=palette,
    style="correct",
    markers=markers,
    ax=ax,
)
ax.get_legend().remove()

# plt.pcolormesh(xx, yy, plot_mesh_labels, cmap=cmap, alpha=0.1)

ax.axis("off")
# ax.set(xticks=[], yticks=[], ylabel="", xlabel="")
ax.set_title(f"Predictions on isolates: accuracy {score:.2f}")
stashfig("isolate-predictions-umap")
