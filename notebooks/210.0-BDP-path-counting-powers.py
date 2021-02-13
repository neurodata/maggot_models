#%%
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import pymaid
from graspologic.utils import binarize, remove_loops
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.pymaid import start_instance
from src.visualization import set_theme

start_instance()


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


set_theme()


#%%

mg = load_metagraph("Gad")
mg = mg.make_lcc()
meta = mg.meta


def apply_annotation(meta, annotation, key):
    ids = pymaid.get_skids_by_annotation(annotation)
    ids = np.intersect1d(ids, meta.index)
    meta[key] = False
    meta.loc[ids, key] = True


apply_annotation(meta, "mw ipsilateral axon", "ipsilateral_axon")
apply_annotation(meta, "mw contralateral axon", "contralateral_axon")
apply_annotation(meta, "mw bilateral axon", "bilateral_axon")


def _source_mapper(name):
    return "source_" + name


def _target_mapper(name):
    return "target_" + name


meta = mg.meta
g = mg.g
# extract edgelist from the graph
edgelist_df = nx.to_pandas_edgelist(g)
# get metadata for the nodes and rename them based on source/target
sources = edgelist_df["source"].values.astype("int64")
targets = edgelist_df["target"].values.astype("int64")
source_meta = meta.loc[sources]
source_meta.index = pd.RangeIndex(len(source_meta))
target_meta = meta.loc[targets]
target_meta.index = pd.RangeIndex(len(target_meta))
source_meta.rename(_source_mapper, axis=1, inplace=True)
target_meta.rename(_target_mapper, axis=1, inplace=True)
# append to the columns of edgelist
edgelist_df = pd.concat(
    (edgelist_df, source_meta, target_meta), axis=1, ignore_index=False
)

#%%


def edgelist_to_mg(edgelist, meta, weight="weight"):
    g = nx.from_pandas_edgelist(edgelist, edge_attr=True, create_using=nx.DiGraph)
    nx.set_node_attributes(g, meta.to_dict(orient="index"))
    mg = MetaGraph(g, weight=weight)
    return mg


def compute_adjacency_powers(adj, max_hops=6):
    power_adjs = {1: adj}
    power_adj = adj
    for k in np.arange(2, max_hops):
        power_adj = power_adj @ adj
        power_adjs[k] = power_adj
    return power_adjs


def fetch_annotation_ids(meta, annotation):
    ids = pymaid.get_skids_by_annotation(annotation)
    ids = np.intersect1d(ids, meta.index)
    return ids


max_hops = 7
rows = []
cuts = [
    "none",
    "source_contralateral_axon",
    "source_bilateral_axon",
]
cut_names = []
for cut in cuts:
    for random in [True, False]:
        if cut == "none" and random:
            continue
        if random:
            n_repeats = 10
        else:
            n_repeats = 1
        for i in range(n_repeats):
            print(i)
            # perform the cut
            if cut != "none":
                cut_edgelist = edgelist_df[~edgelist_df[cut]]
                if random:
                    n_removed = len(edgelist_df) - len(cut_edgelist)
                    keep_inds = np.random.choice(
                        len(edgelist_df),
                        size=len(edgelist_df) - n_removed,
                        replace=False,
                    )
                    cut_edgelist = edgelist_df.iloc[keep_inds]
            else:
                cut_edgelist = edgelist_df
            cut_mg = edgelist_to_mg(cut_edgelist, meta, weight="weight")

            # get the adjacency
            adj = cut_mg.adj
            meta = cut_mg.meta
            adj = binarize(adj)
            adj = remove_loops(adj)

            # powers
            power_adjs = compute_adjacency_powers(adj, max_hops=max_hops)

            # select I/O nodes
            meta["inds"] = range(len(meta))
            # TODO not sure why this is breaking
            # TODO add ascending
            # source_nodes = fetch_annotation_ids(meta, "mw brain inputs")
            # target_nodes = fetch_annotation_ids(meta, "mw brain outputs")
            source_nodes = meta[meta["input"]].index
            target_nodes = meta[meta["output"]].index
            source_inds = meta.loc[source_nodes, "inds"]
            target_inds = meta.loc[target_nodes, "inds"]

            # compute the number of paths for each hop length
            for power, power_adj in power_adjs.items():
                sub_adj = power_adj[np.ix_(source_inds, target_inds)]
                n_paths = sub_adj.sum()
                rows.append(
                    {
                        "cut": cut,
                        "hops": power,
                        "n_paths": n_paths,
                        "repeat": i,
                        "random": random,
                    }
                )

results = pd.DataFrame(rows)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="hops", y="n_paths", hue="cut")
ax.set_yscale("log")
ax.get_legend().remove()
ax.set(xticks=np.arange(1, max_hops))
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Cut")
stashfig("n_paths_by_cut")

# %%
normalizer = results[results["cut"] == "none"].set_index("hops")["n_paths"]
normalizer = results["hops"].map(normalizer)
results["normalized_paths"] = results["n_paths"] / normalizer
#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="hops", y="normalized_paths", hue="cut", style="random")
# sns.scatterplot(data=results, x="hops", y="normalized_paths", hue="cut", legend=False)
ax.set(xticks=np.arange(1, max_hops))
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
stashfig("prop_paths_by_cut")


# %%
