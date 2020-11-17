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

#%%
X = sequencing_df.T.fillna(0).values
#%%
y = scrna_meta["Neuron_type"].values
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(sparse=False)
one_hot_y = one_hot.fit_transform(y.reshape(-1, 1))

#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, one_hot_y, test_size=0.5, stratify=y
)


from sklearn.linear_model import (
    Lasso,
    ElasticNet,
    LogisticRegression,
    SGDClassifier,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
)

# model = ElasticNet()

# LogisticRegression took a really long time
# model = LogisticRegression(
#     penalty="elasticnet", solver="saga", multi_class="multinomial", l1_ratio=0.5
# )
# also took a while
# model = SGDClassifier(loss="log", penalty="elasticnet", l1_ratio=0.85)
# currtime = time.time()
# model.fit(X_train, y_train)

# print(accuracy_score(y_test, model.predict(X_test)))
# model.fit(X_train, y_train.toarray())
# print(f"{time.time() - currtime} elapsed")

currtime = time.time()
model = MultiTaskElasticNetCV(normalize=True, l1_ratio=[1, 0.99, 0.95, 0.9])
model.fit(X_train, y_train)
print(f"{time.time() - currtime} elapsed")

#%%
coefs = model.coef_
abs_coefs = np.abs(coefs).sum(axis=0)
#%%
X_subset = X.values[:, np.nonzero(abs_coefs)[0]]

#%%
from umap import UMAP
import colorcet as cc

umapper = UMAP()
embed_subset = umapper.fit_transform(X_subset)
plot_df = pd.DataFrame(
    data=embed_subset, index=scrna_meta.index, columns=["umap_0", "umap_1"]
)
plot_df["labels"] = scrna_meta["Neuron_type"]
palette = dict(zip(np.unique(plot_df["labels"]), cc.glasbey_light))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=plot_df, x="umap_0", y="umap_1", ax=ax, palette=palette, hue="labels"
)
