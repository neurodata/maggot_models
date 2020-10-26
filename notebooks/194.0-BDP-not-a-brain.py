#%%
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from graspologic.embed import LaplacianSpectralEmbed
from graspologic.plot import heatmap, pairplot
from graspologic.utils import get_lcc, pass_to_ranks

data_dir = Path("maggot_models/data/raw/OneDrive_1_10-21-2020")
covariate_loc = data_dir / "product_node_embedding.csv"
edges_loc = data_dir / "product_edges.csv"
category_loc = data_dir / "partition_mapping.csv"

covariate_df = pd.read_csv(covariate_loc, index_col=0, header=None).sort_index()
meta_df = pd.read_csv(category_loc, index_col=0).sort_index()
print((covariate_df.index == meta_df.index).all())

edges_df = pd.read_csv(edges_loc).sort_index()
g = nx.from_pandas_edgelist(edges_df, edge_attr="weight", create_using=nx.DiGraph)
g.edges
adj = nx.to_numpy_array(g, nodelist=meta_df.index)

#%%

print(f"Number of vertices (original): {len(adj)}")
adj, lcc_inds = get_lcc(adj, return_inds=True)
print(f"Number of vertices (lcc): {len(adj)}")
meta_df = meta_df.iloc[lcc_inds]
covariate_df = covariate_df.iloc[lcc_inds]

heatmap(pass_to_ranks(adj))

#%%


lse = LaplacianSpectralEmbed(form="R-DAD")
embedding = lse.fit_transform(pass_to_ranks(adj))


#%%


# pairplot(embedding[0], labels=meta_df["cat_id"].values, palette="tab20")


#%%

from src.visualization import set_theme

set_theme()

concat_embedding = np.concatenate(embedding, axis=1)

umapper = UMAP(min_dist=0.7, metric="cosine")
umap_embedding = umapper.fit_transform(concat_embedding)

plot_df = pd.DataFrame(
    data=umap_embedding,
    columns=[f"umap_{i}" for i in range(umap_embedding.shape[1])],
    index=meta_df.index,
)
plot_df["cat_id"] = meta_df["cat_id"]

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=plot_df,
    x="umap_0",
    y="umap_1",
    s=20,
    alpha=0.7,
    hue="cat_id",
    palette="tab20",
    ax=ax,
)
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.axis("off")


# %%

from graspologic.utils import to_laplace
from graspologic.embed import AdjacencySpectralEmbed

L = to_laplace(pass_to_ranks(adj), form="R-DAD")
X = covariate_df.values


def build_case_matrix(L, X, alpha, method="assort"):
    if method == "assort":
        L_case = L + alpha * X @ X.T
    elif method == "nonassort":
        L_case = L @ L.T + alpha * X @ X.T
    elif method == "cca":
        L_case = L @ X
    return L_case


alphas = [0, 0.025, 0.05, 0.1]
methods = ["assort", "nonassort"]
case_by_params = {}
umap_by_params = {}
for method in methods:
    for alpha in alphas:
        L_case = build_case_matrix(L, X, alpha, method=method)
        case = AdjacencySpectralEmbed(
            n_components=8, check_lcc=False, diag_aug=False, concat=True
        )
        case_embedding = case.fit_transform(L_case)
        umapper = UMAP(min_dist=0.7, metric="cosine")
        umap_embedding = umapper.fit_transform(case_embedding)
        case_by_params[(method, alpha)] = case_embedding
        umap_by_params[(method, alpha)] = umap_embedding

#%%

import os
from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


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
axs[0, 0].set_ylabel(f"CASE (assortative)")
axs[1, 0].set_ylabel(f"CASE (non-assortative)")
stashfig("casc-umaps")
