# %% [markdown]
# ##

import numpy as np
import pandas as pd

import pymaid
from src.data import load_metagraph
from src.pymaid import start_instance
from tqdm import tqdm

mg = load_metagraph("G")
meta = mg.meta

start_instance()

splits = pymaid.find_treenodes(tags="mw axon split")
splits = splits.set_index("skeleton_id")["treenode_id"].squeeze()


def get_connectors(nl):
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
    return connectors


def append_labeled_nodes(add_list, nodes, split_node, name):
    for node in nodes:
        add_list.append(
            {"treenode_id": node, "treenode_type": name, "split_at": split_node}
        )


def get_treenode_types(nl, splits):
    treenode_info = []
    print("Cutting neurons...")
    for i, n in enumerate(tqdm(nl)):
        skid = int(n.skeleton_id)
        if skid in splits.index:
            split_node = splits[skid]
            # order of output is axon, dendrite
            fragments = pymaid.cut_neuron(n, split_node)
            for f in fragments[:-1]:
                axon_treenodes = f.nodes.treenode_id.values
                append_labeled_nodes(treenode_info, axon_treenodes, split_node, "axon")
            dend_treenodes = fragments[-1].nodes.treenode_id.values
            append_labeled_nodes(treenode_info, dend_treenodes, split_node, "dendrite")
        else:  # unsplittable neuron
            unsplit_treenodes = n.nodes.treenode_id.values
            append_labeled_nodes(treenode_info, unsplit_treenodes, split_node, "unspli")
    treenode_df = pd.DataFrame(treenode_info)
    # a split node is included in pre and post synaptic fragments
    # here i am just removing, i hope there is never a synapse on that node...
    treenode_df = treenode_df[~treenode_df["treenode_id"].duplicated(keep=False)]
    treenode_series = treenode_df.set_index("treenode_id")["treenode_type"]
    return treenode_series


# %% [markdown]
# ##

# params
ids = meta.index.values
ids = [int(i) for i in ids]
nl = pymaid.get_neurons(ids)

# %% [markdown]
# ##
print("Getting connectors...")
connectors = get_connectors(nl)

explode_cols = ["postsynaptic_to", "postsynaptic_to_node"]
index_cols = np.setdiff1d(connectors.columns, explode_cols)

print("Exploding connector DataFrame...")
# explode the lists within the connectors dataframe
connectors = (
    connectors.set_index(list(index_cols)).apply(pd.Series.explode).reset_index()
)
# TODO figure out these nans
connectors = connectors[~connectors.isnull().any(axis=1)]
connectors = connectors.astype(
    {
        "presynaptic_to": "int64",
        "presynaptic_to_node": "int64",
        "postsynaptic_to": "int64",
        "postsynaptic_to_node": "int64",
    }
)


print("Getting treenode compartment types...")
treenode_types = get_treenode_types(nl, splits)


print("Applying treenode types to connectors")
connectors["presynaptic_type"] = connectors["presynaptic_to_node"].map(treenode_types)
connectors["postsynaptic_type"] = connectors["postsynaptic_to_node"].map(treenode_types)

connectors["in_subgraph"] = connectors["presynaptic_to"].isin(ids) & connectors[
    "postsynaptic_to"
].isin(ids)

print("Saving...")
out_path = "maggot_models/data/processed/2020-05-08/"
connectors.to_csv(out_path + "connectors.csv")

