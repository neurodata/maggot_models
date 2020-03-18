# %% [markdown]
# #
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval

import pymaid
from src.data import load_metagraph
from src.io import savecsv, savefig, saveskels
from src.pymaid import start_instance
from itertools import chain

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


VERSION = "2020-03-09"
print(f"Using version {VERSION}")

mg = load_metagraph("G", version=VERSION)

connectors = pd.read_csv(
    "maggot_models/data/processed/2020-03-09/connectors.csv",
    index_col=0,
    dtype={"presynaptic_to": int, "presynaptic_to_node": int},
    converters={"postsynaptic_to": literal_eval, "postsynaptic_to_node": literal_eval},
)

# %% [markdown]
# #


class HypergraphCascade:
    def __init__(self, connectors, store_fmt="hist", max_depth=10, p=0.01):
        self.connectors = connectors
        self.store_fmt = store_fmt
        self.max_depth = max_depth
        self.p = p

    def start_cascade(self, start_ids):
        self._active_ids = [start_ids]
        self.hops = 0

    def step_cascade(self):
        self.hops += 1
        connectors = self.connectors

        # TODO this should be a dictionary keyed by cell ID or index
        candidate_syns = connectors[connectors["presynaptic_to"].isin(self._active_ids)]
        n_syns = len(candidate_syns)

        # simulate a bernoulli for each synapse
        outcomes = np.random.uniform(size=n_syns)
        inds = np.zeros(n_syns, dtype=bool)
        inds[outcomes < self.p] = True

        self._active_syns = candidate_syns.iloc[inds]
        next_ids = self._active_syns["postsynaptic_to"]
        next_ids = list(chain.from_iterable(next_ids))
        next_ids = np.unique(next_ids)
        self._active_ids = next_ids


import seaborn as sns


def plot_syns(connector_df):
    pg = sns.PairGrid(data=connector_df, x_vars=["x", "y", "z"], y_vars=["x", "y", "z"])
    pg.map(sns.scatterplot, alpha=0.02, linewidth=0, s=1)


n_sims = 20
syn_lists = []
for i in range(n_sims):
    syns = []
    hgc = HypergraphCascade(connectors)
    hgc.start_cascade(3299214)
    hgc.step_cascade()
    print(hgc.hops)
    print(hgc._active_ids)
    syns.append(hgc._active_syns)

    hgc.step_cascade()
    print(hgc.hops)
    print(hgc._active_ids)
    syns.append(hgc._active_syns)

    hgc.step_cascade()
    print(hgc.hops)
    print(hgc._active_ids)
    syns.append(hgc._active_syns)
    syn_lists.append(syns)

    hgc.step_cascade()
    print(hgc.hops)
    print(hgc._active_ids)
    syns.append(hgc._active_syns)
    syn_lists.append(syns)

new_syns = []
for syns in syn_lists:
    for hops, syn in enumerate(syns):
        syn["hops"] = hops + 1
        new_syns.append(syn)

syn_df = pd.concat(new_syns, axis=0, ignore_index=True)
pg = sns.PairGrid(
    data=syn_df,
    x_vars=["x", "y", "z"],
    y_vars=["x", "y", "z"],
    hue="hops",
    palette="plasma",
)
pg.map_offdiag(sns.scatterplot, alpha=1, linewidth=0, s=10)


# %% [markdown]
# # The plan - general cascade model
# base eclass
