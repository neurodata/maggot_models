#%% Imports and file loading
from pathlib import Path
import glob
import json
from os import listdir
from operator import itemgetter

import pandas as pd
import networkx as nx
import numpy as np
from graspy.plot import gridplot
from src.data import load_networkx

# File locations
base_path = Path("./maggot_models/data/raw/Maggot-Brain-Connectome/")

data_path = base_path / "4-color-matrices_Brain"

data_date_graphs = "2020-02-26"  # this is for the graph, not the annotations

graph_types = ["axon-axon", "axon-dendrite", "dendrite-axon", "dendrite-dendrite"]

data_date_groups = "2020-03-01"  # this is for the annotations

class_data_folder = base_path / f"neuron-groups/{data_date_groups}"

all_neuron_file = "all-neurons-with-sensories-2020-01-14.json"
left_file = "hemisphere-L-2020-2-26.json"
right_file = "hemisphere-R-2020-2-26.json"

input_counts_file = "input_counts"

pair_file = base_path / "pairs/pairs-2020-02-19.csv"

output_path = Path(f"maggot_models/data/processed/{data_date_groups}")  # CHANGEME

skeleton_data_file = (
    data_path / Path(data_date_graphs) / "skeleton_id_vs_neuron_name.csv"
)

lineage_file = data_path / Path(data_date_graphs) / "skeleton_id_vs_lineage.csv"


def df_to_nx(df, meta_data_dict):
    c = df.columns.values
    c = c.astype(int)
    r = df.index.values
    df.columns = c
    if not (c == r).all():
        raise ValueError("Mismatching df indexing")
    graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    nx.set_node_attributes(graph, meta_data_dict)
    return graph


def extract_ids(lod):
    out_list = []
    for d in lod:
        skel_id = d["skeleton_id"]
        out_list.append(skel_id)
    return out_list


def remove_date(string):
    datestrings = ["-2019", "-2020"]
    for d in datestrings:
        ind = string.find(d)
        if ind != -1:
            return string[:ind]
    print(f"Could not remove date from string {string}")
    return -1


def append_class(df, id, col, name):
    try:
        if df.loc[i, col] == "":
            df.loc[i, col] += name
        elif df.loc[i, col] == "unk":  # always replace "unk"
            df.loc[i, col] = name
        elif not df.loc[i, col]:
            df.loc[i, col] = name
        else:
            df.loc[i, col] += ";" + name
        return 0
    except KeyError:
        print()
        print(f"Skeleton ID {id} from metadata JSON not in graph")
        print(f"Skeleton class was {name}")
        return 1


# # Begin main script

meta_data_df = pd.read_csv(skeleton_data_file, delimiter=",", usecols=range(2))
meta_data_df.rename(lambda x: x.strip(" "), axis=1, inplace=True)
meta_data_df.head()

meta_data_df.set_index("skeleton_id", inplace=True)
skeleton_ids = meta_data_df.index.values
print(f"There are {len(skeleton_ids)} possible nodes in the graph")

# %% [markdown]
# # Load initial files

# append new cell type classes
group_files = listdir(class_data_folder)

remove_files = [all_neuron_file, left_file, right_file]
[group_files.remove(rf) for rf in remove_files]

new_group_files = []
for f in group_files:
    if f.endswith(".json"):
        new_group_files.append(f)
group_files = new_group_files

# %% [markdown]
# # Iterate over all class and subclasses, put into dicts
print("\n\n\n\nAll classes\n\n")
names = []
group_map = {}
subgroup_map = {}
for f in group_files:
    if "CAT" not in f:  # skip categorical ones here
        name = remove_date(f)
        print()
        print(f"Processing metadata JSON for {name}")
        with open(class_data_folder / f, "r") as json_file:
            temp_dict = json.load(json_file)
            temp_ids = extract_ids(temp_dict)
            if "subclass_" in name:
                ind = name.find("subclass_")
                temp_name = name[ind + len("subclass_") :]  # only keep things after
                subgroup_map[temp_name] = temp_ids
            else:
                group_map[name] = temp_ids

# %% [markdown]
# #
meta_data_df["Class 1"] = "unk"
num_missing = 0
for name, ids in group_map.items():
    for i in ids:
        num_missing += append_class(meta_data_df, i, "Class 1", name)

print("\n\n\n\n")
print(f"{num_missing} skeleton IDs missing from graph")
print("\n\n\n\n")

meta_data_df["Class 2"] = ""
for name, ids in subgroup_map.items():
    for i in ids:
        num_missing += append_class(meta_data_df, i, "Class 2", name)

# Merge class (put class 1 and class 2 together as a column)
meta_data_df["Merge Class"] = ""
for i in meta_data_df.index.values:
    merge_class = meta_data_df.loc[i, "Class 1"]
    if meta_data_df.loc[i, "Class 2"] != "":
        merge_class += "-" + meta_data_df.loc[i, "Class 2"]
    meta_data_df.loc[i, "Merge Class"] = merge_class

#%% manage the "Categorical" (true/false) labels
print("\n\n\n\n All true/false labels")
for f in group_files:
    if "CAT" in f:
        name = remove_date(f)
        print()
        print(f"Processing metadata JSON for {name}")
        name = name.replace("CAT-", "")
        meta_data_df[name] = False
        with open(class_data_folder / f, "r") as json_file:
            temp_dict = json.load(json_file)
            temp_ids = extract_ids(temp_dict)
            for i in temp_ids:
                append_class(meta_data_df, i, name, True)

# %% [markdown]
# # Add hemisphere labels
meta_data_df["Hemisphere"] = None
for f in [left_file, right_file]:
    name = remove_date(f)
    name = name.replace("hemisphere-", "")
    print(name)
    with open(class_data_folder / f, "r") as json_file:
        temp_dict = json.load(json_file)
        temp_ids = extract_ids(temp_dict)
        for i in temp_ids:
            append_class(meta_data_df, i, "Hemisphere", name)
meta_data_df.loc[17068730, "Hemisphere"] = "C"
meta_data_df.loc[3813487, "Hemisphere"] = "C"
print("\n\n\n\nMissing Hemisphere annotation for")
print(meta_data_df[meta_data_df["Hemisphere"].isnull()])

# %% [markdown]
# # Pairs

# Pairs (NOTE this file has some issues where some ids are repeated in multiple pairs)
pair_df = pd.read_csv(pair_file, usecols=range(2))
pair_df["pair_id"] = range(len(pair_df))

uni_left, left_counts = np.unique(pair_df["leftid"], return_counts=True)
uni_right, right_counts = np.unique(pair_df["rightid"], return_counts=True)

dup_left_inds = np.where(left_counts != 1)[0]
dup_right_inds = np.where(right_counts != 1)[0]
dup_left_ids = uni_left[dup_left_inds]
dup_right_ids = uni_right[dup_right_inds]

print("\n\n\n\n")
if len(dup_left_inds) > 0:
    print("Duplicate pairs left:")
    print(dup_left_ids)
if len(dup_right_inds) > 0:
    print("Duplicate pairs right:")
    print(dup_right_ids)
print("\n\n\n\n")

drop_df = pair_df[
    pair_df["leftid"].isin(dup_left_ids) | pair_df["rightid"].isin(dup_right_ids)
]
print("\n\n\n\n")
print("Dropping pairs:")
print(drop_df)
print("\n\n\n\n")

pair_df.drop(drop_df.index, axis=0, inplace=True)

pair_ids = np.concatenate((pair_df["leftid"].values, pair_df["rightid"].values))
meta_ids = meta_data_df.index.values
in_meta_ids = np.isin(pair_ids, meta_ids)
drop_ids = pair_ids[~in_meta_ids]
pair_df = pair_df[~pair_df["leftid"].isin(drop_ids)]
pair_df = pair_df[~pair_df["rightid"].isin(drop_ids)]

left_to_right_df = pair_df.set_index("leftid")
right_to_left_df = pair_df.set_index("rightid")
right_to_left_df.head()

meta_data_df["Pair"] = -1
meta_data_df["Pair ID"] = -1
meta_data_df.loc[left_to_right_df.index, "Pair"] = left_to_right_df["rightid"]
meta_data_df.loc[right_to_left_df.index, "Pair"] = right_to_left_df["leftid"]

meta_data_df.loc[left_to_right_df.index, "Pair ID"] = left_to_right_df["pair_id"]
meta_data_df.loc[right_to_left_df.index, "Pair ID"] = right_to_left_df["pair_id"]

#%% Fix places where L/R labels are not the same
print("\n\n\n\nFinding asymmetric L/R labels")
for i in range(len(meta_data_df)):
    my_id = meta_data_df.index[i]
    my_class = meta_data_df.loc[my_id, "Class 1"]
    partner_id = meta_data_df.loc[my_id, "Pair"]
    if partner_id != -1:
        partner_class = meta_data_df.loc[partner_id, "Class 1"]
        if partner_class != "unk" and my_class == "unk":
            print(f"{my_id} had asymmetric class label {partner_class}, fixed")
            meta_data_df.loc[my_id, "Class 1"] = partner_class
        elif (partner_class != my_class) and (partner_class != "unk"):
            print(
                f"{meta_data_df.index[i]} and partner {partner_id} have different labels"
            )
print()

#%% lineagees
lineage_df = pd.read_csv(lineage_file)
lineage_df = lineage_df.set_index("skeleton_id")
lineage_df = lineage_df.fillna("unk")
# ignore lineages for nonexistent skeletons
lineage_df = lineage_df[lineage_df.index.isin(meta_data_df.index)]
print(f"Missing lineage info for {len(meta_data_df) - len(lineage_df)} skeletons")
print(
    meta_data_df[~meta_data_df.index.isin(lineage_df.index) & meta_data_df["is_brain"]]
)
print("\n\n\n\n")


def filter(string):
    string = string.replace("akira", "")
    string = string.replace("Lineage", "")
    string = string.replace("*", "")
    string = string.strip("_")
    string = string.strip(" ")
    string = string.replace("_r", "")
    string = string.replace("_l", "")
    return string


lineages = lineage_df["lineage"]
lineages = np.vectorize(filter)(lineages)
meta_data_df["lineage"] = "unk"
meta_data_df.loc[lineage_df.index, "lineage"] = lineages
nulls = meta_data_df[meta_data_df.isnull().any(axis=1)]

input_counts_path = data_path / data_date_graphs / (input_counts_file + ".csv")
input_counts_df = pd.read_csv(input_counts_path, index_col=0)
cols = input_counts_df.columns.values
cols = [str(c).strip(" ") for c in cols]
input_counts_df.columns = cols

meta_data_df.loc[input_counts_df.index, "dendrite_input"] = input_counts_df[
    "dendrite_inputs"
]
meta_data_df.loc[input_counts_df.index, "axon_input"] = input_counts_df["axon_inputs"]

meta_data_dict = meta_data_df.to_dict(orient="index")


#%% Import the raw graphs
nx_graphs_raw = {}
df_graphs_raw = {}
for graph_type in graph_types:
    print(graph_type)
    edgelist_path = data_path / data_date_graphs / (graph_type + ".csv")
    adj = pd.read_csv(edgelist_path, index_col=0)
    graph = df_to_nx(adj, meta_data_dict)
    nx_graphs_raw[graph_type] = graph
    df_graphs_raw[graph_type] = adj
    print()


#%% Normalize weights for the raw graphs
df_graphs_norm = {}
nx_graphs_norm = {}
print("Checking normalized weights")
input_counts = input_counts_df["axon_inputs"].values

input_counts[input_counts == 0] = 1
for graph_type in ["axon-axon", "dendrite-axon"]:
    print(graph_type)
    df_adj_raw = df_graphs_raw[graph_type]
    if (input_counts_df.index.values == adj.index.values).all():
        print("Same indexing!")
    else:
        raise ValueError("Indexing of input counts file not the same!")
    adj_raw = df_adj_raw.values
    adj_norm = adj_raw / input_counts[np.newaxis, :]
    print(adj_norm.sum(axis=0).max())
    df_adj_norm = pd.DataFrame(
        index=df_adj_raw.index, columns=df_adj_raw.columns, data=adj_norm
    )
    df_graphs_norm[graph_type] = df_adj_norm
    graph = df_to_nx(df_adj_norm, meta_data_dict)
    nx_graphs_norm[graph_type] = graph
    print()

input_counts = input_counts_df["dendrite_inputs"].values
input_counts[input_counts == 0] = 1
for graph_type in ["axon-dendrite", "dendrite-dendrite"]:
    print(graph_type)
    df_adj_raw = df_graphs_raw[graph_type]
    if (input_counts_df.index.values == adj.index.values).all():
        print("Same indexing!")
    adj_raw = df_adj_raw.values
    adj_norm = adj_raw / input_counts[np.newaxis, :]
    print(adj_norm.sum(axis=0).max())
    df_adj_norm = pd.DataFrame(
        index=df_adj_raw.index, columns=df_adj_raw.columns, data=adj_norm
    )
    df_graphs_norm[graph_type] = df_adj_norm
    graph = df_to_nx(df_adj_norm, meta_data_dict)
    nx_graphs_norm[graph_type] = graph
    print()

#%%

print("\n\n\n\nChecking for rows with Nan values")
missing_na = []
nan_df = meta_data_df[meta_data_df.isna().any(axis=1)]
for row in nan_df.index:
    na_ind = nan_df.loc[row].isna()
    print(nan_df.loc[row][na_ind])
    missing_na.append(row)
print()
print("These skeletons have missing values in the metadata")
print(missing_na)
print("\n\n\n\n")

#%% All-all graph
total_input = (
    input_counts_df["dendrite_inputs"].values + input_counts_df["axon_inputs"].values
)
total_input[total_input == 0] = 1

all_adj_raw = np.zeros_like(adj_norm)
for graph_type in graph_types:
    all_adj_raw += df_graphs_raw[graph_type].values

df_all_raw = pd.DataFrame(
    index=df_adj_raw.index, columns=df_adj_raw.columns, data=all_adj_raw
)

nx_all_raw = df_to_nx(df_all_raw, meta_data_dict)

all_adj_norm = all_adj_raw / total_input[np.newaxis, :]
df_all_norm = pd.DataFrame(
    index=df_adj_raw.index, columns=df_adj_raw.columns, data=all_adj_norm
)

nx_all_norm = df_to_nx(df_all_norm, meta_data_dict)

#%% Save

out_graphs = []
[out_graphs.append(i) for i in nx_graphs_raw.values()]
[print(i) for i in nx_graphs_raw.keys()]
save_names = ["Gaa", "Gad", "Gda", "Gdd"]
[out_graphs.append(i) for i in nx_graphs_norm.values()]
[print(i) for i in nx_graphs_norm.keys()]
save_names += ["Gaan", "Gdan", "Gadn", "Gddn"]
out_graphs.append(nx_all_raw)
save_names.append("G")
out_graphs.append(nx_all_norm)
save_names.append("Gn")

for name, graph in zip(save_names, out_graphs):
    nx.write_graphml(graph, output_path / (name + ".graphml"))

meta_data_df.to_csv(output_path / "meta_data.csv")

#%% verify things are right
print("\n\n\n\nChecking graphs are the same when saved")
for name, graph_wrote in zip(save_names, out_graphs):
    print(name)
    graph_read = nx.read_graphml(output_path / (name + ".graphml"))
    adj_read = nx.to_numpy_array(graph_read)
    adj_wrote = nx.to_numpy_array(graph_wrote)
    print(np.array_equal(adj_read, adj_wrote))
    graph_loader = load_networkx(name, version=data_date_graphs)
    adj_loader = nx.to_numpy_array(graph_loader)
    print(np.array_equal(adj_wrote, adj_loader))
    print()

