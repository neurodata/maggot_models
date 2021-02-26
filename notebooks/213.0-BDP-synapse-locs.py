#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_metagraph

data_loc = Path("maggot_models/data/processed/2020-09-23/connectors.csv")

connectors = pd.read_csv(data_loc, index_col=0)

connectors = connectors[connectors["in_subgraph"]]
print(f"Number of connectors in induced subgraph: {len(connectors)}")

connectors["partners"] = list(
    zip(connectors["presynaptic_to"], connectors["postsynaptic_to"])
)
connectors.head()
#%% how common are cases where a pair of neurons is represented twice in a synapse?

grouped = connectors.groupby("connector_id")


def n_unique(series):
    return series.nunique()


n_unique_partners = grouped["partners"].agg(n_unique)
n_partners = grouped["partners"].count()

#%%
p_unique

# p_unique_partners = []
# for name, group in grouped:
#     p_unique_partners.append(group["partners"].nunique() / len(group))

#%%
uni_partners, counts = np.unique(connectors["partners"], return_counts=True)

#%%
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(
    x=counts,
    ax=ax,
    discrete=True,
    stat="probability",
)
ax.set(
    xlim=(0, 20),
    xticks=np.arange(1, 20),
    xlabel="Number of (i, j) edgelets",
)
# ax.bar(np.arange(len(counts)), counts)
#%%


mg = load_metagraph("G")
adj = mg.adj

n_ones = len(np.where(adj == 1)[0])
n_total = np.count_nonzero(adj)
print(n_ones / n_total)
#%%

#%%
for connector_id in connectors["connector_id"].unique()[:10]:
    print(connector_id)
    sub = connectors[connectors["connector_id"] == connector_id]
    print(sub["presynaptic_type"])
    print(sub["postsynaptic_type"])
#%%

connectors[connectors["connector_id"] == 38113][
    [
        "presynaptic_to",
        "postsynaptic_to",
        "x",
        "y",
        "z",
        "presynaptic_type",
        "postsynaptic_type",
    ]
]
