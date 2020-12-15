#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


data_dir = Path("maggot_models/data/raw/drosophila-connectome-embeddings/data")
AA = pd.read_csv(data_dir / "connectome-g1-LSE-n2965-dhat11x2-unnorm.csv", index_col=0)

AD = pd.read_csv(data_dir / "connectome-g2-LSE-n2965-dhat11x2-unnorm.csv", index_col=0)

DA = pd.read_csv(data_dir / "connectome-g3-LSE-n2965-dhat11x2-unnorm.csv", index_col=0)

DD = pd.read_csv(data_dir / "connectome-g4-LSE-n2965-dhat11x2-unnorm.csv", index_col=0)

meta = pd.read_csv(data_dir / "connectome-meta-n2695x16.csv", index_col=0)
meta["mbin"] = meta["Class4"] == "MBIN"

#%%
umap = UMAP(n_neighbors=20, n_components=2, metric="cosine", min_dist=0.7)

graph_names = ["AA", "AD", "DA", "DD"]
umap_embeddings = {}
plot_embeddings = {}
for embedding, graph_name in zip([AA, AD, DA, DD], graph_names):
    umap_embedding = umap.fit_transform(embedding.values)
    umap_embeddings[graph_name] = umap_embedding
    plot_df = pd.DataFrame(
        data=umap_embedding, columns=["umap_0", "umap_1"], index=meta.index
    )
    plot_df = pd.concat((plot_df, meta), axis=1)
    plot_embeddings[graph_name] = plot_df

#%%
# color mapping
colors = sns.color_palette("deep", 10)
red = colors[3]
grey = colors[7]
palette = {True: red, False: grey}

title_fontsize = 22.5

sns.set()
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

for i, (graph_name, plot_df) in enumerate(plot_embeddings.items()):
    ax = axs[i]
    sns.scatterplot(
        data=plot_df[~plot_df["mbin"]],
        x="umap_0",
        y="umap_1",
        hue="mbin",
        ax=ax,
        s=5,
        linewidth=0,
        alpha=0.5,
        palette=palette,
    )
    sns.scatterplot(
        data=plot_df[plot_df["mbin"]],
        x="umap_0",
        y="umap_1",
        hue="mbin",
        ax=ax,
        s=20,
        linewidth=0,
        alpha=1,
        zorder=100,
        palette=palette,
    )
    ax.get_legend().remove()
    ax.axis("off")
    ax.set_title(graph_name, fontsize=title_fontsize)

stashfig("hayden-MILP-umap")

#%%

import numpy as np

a = np.array([[0.8, 0.1], [0.1, 0.8]])
labels = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]])
a[(labels, labels.T)]
