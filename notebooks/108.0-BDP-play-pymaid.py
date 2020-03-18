# %% [markdown]
# #
from src.pymaid import start_instance
from src.data import load_metagraph
import pymaid
from src.io import savecsv, savefig, saveskels

import os

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


VERSION = "2020-03-09"
print(f"Using version {VERSION}")

mg = load_metagraph("G", version=VERSION)
start_instance()

nl = pymaid.get_neurons([mg.meta.index[2]])

# Plot using default settings
fig, ax = nl.plot2d()

# %% [markdown]
# #
mg = mg.sort_values("Pair ID")
nl = pymaid.get_neurons(mg.meta[mg.meta["Merge Class"] == "sens-ORN"].index.values)
fig, ax = nl.plot2d()


# %%
nl = pymaid.get_neurons(mg.meta.index.values)
print(len(nl))
# %% [markdown]
# #
import pandas as pd
import seaborn as sns

connectors = pymaid.get_connectors(nl)
connectors.set_index("connector_id", inplace=True)
connectors.drop(
    [
        "confidence",
        "creation_time",
        "edition_time",
        "tags",
        "creator",
        "editor",
        "type",
    ],
    inplace=True,
    axis=1,
)
details = pymaid.get_connector_details(connectors.index.values)
details.set_index("connector_id", inplace=True)
connectors = pd.concat((connectors, details), ignore_index=False, axis=1)
connectors.reset_index(inplace=True)
# %% [markdown]
# #
pg = sns.PairGrid(data=connectors, x_vars=["x", "y", "z"], y_vars=["x", "y", "z"])
pg.map(sns.scatterplot, alpha=0.02, linewidth=0, s=1)

# %% [markdown]
# #

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False

mpl.rcParams["axes.spines.top"] = False


sns.set_context("talk")


def get_len(it):
    if isinstance(it, (list, tuple)):
        return len(it)
    else:
        return 0


connectors["n_post"] = connectors["postsynaptic_to"].map(get_len)
connectors = connectors[connectors["n_post"] != 0]

# %% [markdown]
# #

connectors = connectors[~connectors.isnull().any(axis=1)]

# %% [markdown]
# #
# %% [markdown]
# #

# #


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
bins = np.linspace(0.5, 10.5, 11)
sns.distplot(connectors["n_post"], ax=ax, bins=bins, kde=False, norm_hist=True)
ax.set_xticks(np.arange(1, 11))
ax.set_ylabel("Frequency")
ax.set_xlabel("# postsynaptic partners")
stashfig("n_postsynaptic")

connectors.to_csv("maggot_models/data/processed/2020-03-09/connectors.csv")
# %% [markdown]
# #
