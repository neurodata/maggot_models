#%% [markdown]
## Embedding graphs with covariates
# > Some experiments on a real dataset of interest using CASE (and soon to use MASE/OMNI)

# - toc: true
# - badges: true
# - categories: [pedigo, graspologic]
# - hide: true
# - search_exclude: true
#%% [markdown]
### Preliminaries
#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
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
print(f"Number of vertices (original): {len(adj)}")
adj, lcc_inds = get_lcc(adj, return_inds=True)
print(f"Number of vertices (lcc): {len(adj)}")
meta_df = meta_df.iloc[lcc_inds]
covariate_df = covariate_df.iloc[lcc_inds]
X = covariate_df.values

#%%
colors = sns.color_palette("deep")
palette = dict(zip(np.unique(meta_df["cat_id"]), colors))

#%% [markdown]
### Adjacency matrix (sorted by category)
#%%
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
lse = LaplacianSpectralEmbed(form="R-DAD")
embedding = lse.fit_transform(pass_to_ranks(adj))
pairplot(embedding[0], labels=meta_df["cat_id"].values, palette=palette)

#%%
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
ax.axis("off")

#%% [markdown]
### CASE
# %%
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

#%%
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
stashfig("casc-umaps")

#%% [markdown]
### A simple classifier on the embeddings
#%%
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
    ylabel="Accuracy", xlabel=r"$\alpha$", title="5-fold cross validation, KNN o CASE"
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
stashfig("knn-case")
