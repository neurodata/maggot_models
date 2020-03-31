#%% Imports and file loading
import glob
import json
import pprint
import sys
from operator import itemgetter
from os import listdir
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import pymaid
from graspy.plot import gridplot
from src.data import load_networkx
from src.pymaid import start_instance

# File locations
base_path = Path("./maggot_models/data/raw/Maggot-Brain-Connectome/")

data_path = base_path / "4-color-matrices_Brain"

data_date_graphs = "2020-03-06"  # this is for the graph, not the annotations

graph_types = ["axon-axon", "axon-dendrite", "dendrite-axon", "dendrite-dendrite"]

input_counts_file = "input_counts"

pair_file = base_path / "pairs/pairs-2020-02-19.csv"

output_name = "2020-03-26"
output_path = Path(f"maggot_models/data/processed/{output_name}")

sys.stdout = open(f"maggot_models/data/logs/{output_name}.txt", "w")


print(output_path)


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


priority_map = {
    "MBON": 1,
    "MBIN": 1,
    "KC": 1,
    "uPN": 1,
    "tPN": 1,
    "vPN": 1,
    "mPN": 1,
    "sens": 1,
    "APL": 1,
    "LHN": 2,
    "CN": 2,
    "dVNC": 2,
    "dSEZ": 2,
    "RG": 2,
    "dUnk": 2,
    "FBN": 3,
    "FAN": 3,
    "LHN2": 4,
    "CN2": 5,
    "FB2N": 3,
    "FFN": 3,
}


def priority(name):
    if name in priority_map:
        return priority_map[name]
    else:
        return 1000


check_priority = np.vectorize(priority)


def get_single_class(classes):
    single_class = classes[0]
    for c in classes[1:]:
        single_class += ";" + c
    return str(single_class)


def get_classes(meta, class_cols, fill_unk=False):
    all_class = []
    single_class = []
    n_class = []
    for index, row in meta.iterrows():
        classes = class_cols[row[class_cols].astype(bool)]
        all_class.append(str(classes))
        n_class.append(int(len(classes)))
        if len(classes) > 0:
            priorities = check_priority(classes)
            inds = np.where(priorities == priorities.min())[0]
            sc = get_single_class(classes[inds])
        else:
            if fill_unk:
                sc = "unk"
            else:
                sc = ""
        single_class.append(sc)
    return single_class, all_class, n_class


# %% [markdown]
# ##
print("Loading annotations:\n")

start_instance()
annot_df = pymaid.get_annotated("mw neuron groups")

series_ids = []
for annot_name in annot_df["name"]:
    print(annot_name)
    ids = pymaid.get_skids_by_annotation(annot_name)
    name = annot_name.replace("mw ", "")
    name = name.replace(" ", "_")
    indicator = pd.Series(
        index=ids, data=np.ones(len(ids), dtype=bool), name=name, dtype=bool
    )
    series_ids.append(indicator)
    print()


# %% [markdown]
# ##
meta = pd.concat(series_ids, axis=1, ignore_index=False)
meta.fillna(False, inplace=True)

class1_name_map = {
    "APL": "APL",
    "dSEZ": "dSEZ",
    "dVNC": "dVNC",
    "RG": "RG",
    "picky_LN": "pLN",
    "choosy_LN": "cLN",
    "broad_LN": "bLN",
    "CN": "CN",
    "CN2": "CN2",
    "CX": "CX",
    "FAN": "FAN",
    "FB2N": "FB2N",
    "FBN": "FBN",
    "KC": "KC",
    "keystone": "keystone",
    "LHN": "LHN",
    "LHN2": "LHN2",
    "LON": "LON",
    "MBIN": "MBIN",
    "MBON": "MBON",
    "motor": "motor",
    "mPN": "mPN",
    "dUnk": "dUnk",
    "sens": "sens",
    "tPN": "tPN",
    "uPN": "uPN",
    "vPN": "vPN",
}


meta.rename(class1_name_map, axis=1, inplace=True)


# %% [markdown]
# ##
class1_cols = np.array(list(class1_name_map.values()))


single_class1, all_class1, n_class1 = get_classes(meta, class1_cols, fill_unk=True)

meta["class1"] = single_class1
meta["all_class1"] = all_class1
meta["n_class1"] = n_class1


# %% [markdown]
# ##
class2_cols = []
for c in meta.columns.values:
    if "subclass" in c:
        class2_cols.append(c)
class2_cols = np.array(class2_cols)


single_class2, all_class2, n_class2 = get_classes(meta, class2_cols)


def remove_subclass(string):
    ind = string.find("subclass_")
    return string[ind + len("subclass_") :]


class2_name_map = {
    "appetitive": "app",
    "aversive": "av",
    "neither": "neith",
    "olfactory": "olfac",
}


def name_mapper(string, name_map):
    if string in name_map:
        return name_map[string]
    else:
        return string


single_class2 = np.vectorize(remove_subclass)(single_class2)
single_class2 = np.vectorize(lambda x: name_mapper(x, class2_name_map))(single_class2)

meta["class2"] = single_class2
meta["all_class2"] = all_class2
meta["n_class2"] = n_class2

# %% [markdown]
# ##
print()
print("Class 1 unique values:")
pprint.pprint(dict(zip(*np.unique(all_class1, return_counts=True))))
print()
print("Class 2 unique values:")
pprint.pprint(dict(zip(*np.unique(all_class2, return_counts=True))))
print()

# %% [markdown]
# ## Hemisphere
meta["hemisphere"] = "C"  # default is center
left_meta = meta[meta["left"]]
meta.loc[left_meta.index, "hemisphere"] = "L"
right_meta = meta[meta["right"]]
meta.loc[right_meta.index, "hemisphere"] = "R"

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

print("\n\n")
if len(dup_left_inds) > 0:
    print("Duplicate pairs left:")
    print(dup_left_ids)
if len(dup_right_inds) > 0:
    print("Duplicate pairs right:")
    print(dup_right_ids)
print("\n\n")

drop_df = pair_df[
    pair_df["leftid"].isin(dup_left_ids) | pair_df["rightid"].isin(dup_right_ids)
]
print("\n\n")
print("Dropping pairs:")
print(drop_df)
print("\n\n")

pair_df.drop(drop_df.index, axis=0, inplace=True)

pair_ids = np.concatenate((pair_df["leftid"].values, pair_df["rightid"].values))
meta_ids = meta.index.values
in_meta_ids = np.isin(pair_ids, meta_ids)
drop_ids = pair_ids[~in_meta_ids]
pair_df = pair_df[~pair_df["leftid"].isin(drop_ids)]
pair_df = pair_df[~pair_df["rightid"].isin(drop_ids)]

left_to_right_df = pair_df.set_index("leftid")
right_to_left_df = pair_df.set_index("rightid")
right_to_left_df.head()

meta["Pair"] = -1
meta["Pair ID"] = -1
meta.loc[left_to_right_df.index, "Pair"] = left_to_right_df["rightid"]
meta.loc[right_to_left_df.index, "Pair"] = right_to_left_df["leftid"]

meta.loc[left_to_right_df.index, "Pair ID"] = left_to_right_df["pair_id"]
meta.loc[right_to_left_df.index, "Pair ID"] = right_to_left_df["pair_id"]

#%% Fix places where L/R labels are not the same
print("\n\nFinding asymmetric L/R labels")
for i in range(len(meta)):
    my_id = meta.index[i]
    my_class = meta.loc[my_id, "class1"]
    partner_id = meta.loc[my_id, "Pair"]
    if partner_id != -1:
        partner_class = meta.loc[partner_id, "class1"]
        if partner_class != "unk" and my_class == "unk":
            print(f"{my_id} had asymmetric class label {partner_class}, fixed")
            meta.loc[my_id, "class1"] = partner_class
        elif (partner_class != my_class) and (partner_class != "unk"):
            msg = (
                f"{meta.index[i]} and partner {partner_id} have different labels"
                + f", labels are {my_class}, {partner_class}"
            )
            print(msg)
print()

# %% [markdown]
# #

# Merge class (put class 1 and class 2 together as a column)
meta["merge_class"] = ""
for i in meta.index.values:
    merge_class = meta.loc[i, "class1"]
    if meta.loc[i, "class2"] != "":
        merge_class += "-" + meta.loc[i, "class2"]
    meta.loc[i, "merge_class"] = merge_class

print()
print("Merge class unique values:")
pprint.pprint(dict(zip(*np.unique(meta["merge_class"], return_counts=True))))
print()
#%% lineages
lineage_df = pd.read_csv(lineage_file)
lineage_df = lineage_df.set_index("skeleton_id")
lineage_df = lineage_df.fillna("unk")
# ignore lineages for nonexistent skeletons
lineage_df = lineage_df[lineage_df.index.isin(meta.index)]
print(f"Missing lineage info for {len(meta) - len(lineage_df)} skeletons")
print()
print(f"Brain neurons without entry in lineage file:")
print(meta[~meta.index.isin(lineage_df.index) & meta["brain_neurons"]])
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
meta["lineage"] = "unk"
meta.loc[lineage_df.index, "lineage"] = lineages
nulls = meta[meta.isnull().any(axis=1)]

input_counts_path = data_path / data_date_graphs / (input_counts_file + ".csv")
input_counts_df = pd.read_csv(input_counts_path, index_col=0)
cols = input_counts_df.columns.values
cols = [str(c).strip(" ") for c in cols]
input_counts_df.columns = cols

meta.loc[input_counts_df.index, "dendrite_input"] = input_counts_df["dendrite_inputs"]
meta.loc[input_counts_df.index, "axon_input"] = input_counts_df["axon_inputs"]


#%% Import the raw graphs
print("Importing raw adjacency matrices:\n")
nx_graphs_raw = {}
df_graphs_raw = {}
for graph_type in graph_types:
    print(graph_type)
    edgelist_path = data_path / data_date_graphs / (graph_type + ".csv")
    adj = pd.read_csv(edgelist_path, index_col=0)
    meta = meta.reindex(adj.index)
    meta_data_dict = meta.to_dict(orient="index")
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

print("\n\nChecking for rows with Nan values")
missing_na = []
nan_df = meta[meta.isna().any(axis=1)]
for row in nan_df.index:
    na_ind = nan_df.loc[row].isna()
    print(nan_df.loc[row][na_ind])
    missing_na.append(row)
print()
print("These skeletons have missing values in the metadata")
print(missing_na)
print("\n\n")


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

print("Saving graphs:\n")
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

meta.to_csv(output_path / "meta_data.csv")

#%% verify things are right
print("\n\nChecking graphs are the same when saved")
print(output_path)
for name, graph_wrote in zip(save_names, out_graphs):
    print(name)
    graph_read = nx.read_graphml(output_path / (name + ".graphml"))
    adj_read = nx.to_numpy_array(graph_read)
    adj_wrote = nx.to_numpy_array(graph_wrote)
    print(np.array_equal(adj_read, adj_wrote))
    graph_loader = load_networkx(name, version=output_name)
    adj_loader = nx.to_numpy_array(graph_loader)
    print(np.array_equal(adj_wrote, adj_loader))
    print()

print("Done!")
sys.stdout.close()
