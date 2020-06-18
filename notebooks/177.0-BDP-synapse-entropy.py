# %% [markdown]
# ##
import pandas as pd
import numpy as np

connector_loc = "maggot_models/data/processed/2020-06-10/connectors.csv"
connectors = pd.read_csv(connector_loc, index_col=0)

sizes = connectors.groupby(
    ["connector_id", "presynaptic_type", "postsynaptic_type"]
).size()
sizes.name = "count"

sizes = sizes.reset_index()

sizes["subconnector_type"] = (
    sizes["presynaptic_type"] + "-" + sizes["postsynaptic_type"]
)
n_postsynaptic = sizes.groupby("connector_id")["count"].sum()
sizes["n_postsynaptic"] = sizes["connector_id"].map(n_postsynaptic)
sizes["prop"] = sizes["count"] / sizes["n_postsynaptic"]

connector_entrop = {}
for connector_id, group in sizes.groupby("connector_id"):
    entrop = -np.sum(group["prop"] * np.log(group["prop"]) / np.log(4))
    connector_entrop[connector_id] = entrop

# %% [markdown]
# ##
entrop = pd.Series(connector_entrop)

import seaborn as sns

sns.distplot(entrop)

# %% [markdown]
# ##

sns.distplot(sizes["prop"])

(sizes["prop"] == 1).sum() / len(sizes)

# %% [markdown]
# ##

from src.data import load_metagraph

mg = load_metagraph("G")
mg.make_lcc()

graph_types = ["Gad", "Gaa", "Gdd", "Gda"]  # "Gs"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g)
    # this line is important, to make the graphs aligned
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

# %% [markdown]
# ##
adj_tensor = np.array(adjs)
sums = adj_tensor.sum(axis=0)
mask = sums > 0
edges_by_type = []
for a in adjs:
    edges = a[mask]
    edges_by_type.append(edges)
edges_by_type = np.array(edges_by_type)

# %% [markdown]
# ##
sums_by_edge = edges_by_type.sum(axis=0)
prop = edges_by_type / sums_by_edge[None, ...]
entrop_by_syn = -np.sum(np.multiply(prop, np.log(prop)) / np.log(4), axis=0)
sns.distplot(entrop_by_syn, bins=np.linspace(0, 1, 40))
# adj_tensor[mask[np.newaxis, ...]].shape()
# prop_tensor = adj_tensor / sums[np.newaxis, ...]

# %% [markdown]
# ##

from sklearn.decomposition import PCA

prop_latent = PCA(n_components=2).fit_transform(prop.T)
sns.scatterplot(x=prop_latent[:, 0], y=prop_latent[:, 1], s=10, alpha=0.05, linewidth=0)

# %% [markdown]
# ##

