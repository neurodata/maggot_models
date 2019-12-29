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

base_path = Path("./maggot_models/data/raw/Maggot-Brain-Connectome/")

data_path = base_path / "4-color-matrices_Brain"

data_date = "2019-09-18-v2"  # this is for the graph, not the annotations

graph_types = ["axon-axon", "axon-dendrite", "dendrite-axon", "dendrite-dendrite"]

meta_data_file = "brain_meta-data"

data_date_groups = "2019-12-18"

class_data_folder = base_path / f"neuron-groups/{data_date_groups}"

input_counts_file = "input_counts"

output_path = Path(f"maggot_models/data/processed/{data_date_groups}")

meta_data_path = data_path / data_date / (meta_data_file + ".csv")
meta_data_df = pd.read_csv(meta_data_path, index_col=0)

meta_data_df["Old Class"] = meta_data_df["Class"]
meta_data_df.drop("Class", inplace=True, axis=1)


def extract_ids(lod):
    out_list = []
    for d in lod:
        skel_id = d["skeleton_id"]
        out_list.append(skel_id)
    return out_list


# append new cell type classes
group_files = listdir(class_data_folder)
group_files.remove("usplit-2019-12-20.json")  # add these back in later
group_files.remove("LNp-2019-12-20.json")
# group_files.remove("LNp-2019-12-20.json")
# group_files.remove("LNp-2019-12-20.json")

names = []
group_map = {}
subgroup_map = {}
for f in group_files:
    name = f.replace("-2019-12-9.json", "")
    name = name.replace("-2019-12-18.json", "")
    name = name.replace("-2019-12-20.json", "")
    names.append(name)

    with open(class_data_folder / f, "r") as json_file:
        temp_dict = json.load(json_file)
        temp_ids = extract_ids(temp_dict)
        if "subclass" in name:
            temp_name = name.replace("subclass_", "")
            subgroup_map[temp_name] = temp_ids
        else:
            group_map[name] = temp_ids

meta_data_df["Class 1"] = ""

kc_inds = meta_data_df[meta_data_df["Old Class"] == "KCs"].index.values
meta_data_df.loc[kc_inds, "Class 1"] = "KC"

num_missing = 0
for name, ids in group_map.items():
    for i in ids:
        try:
            if meta_data_df.loc[i, "Class 1"] == "":
                meta_data_df.loc[i, "Class 1"] += name
            else:
                meta_data_df.loc[i, "Class 1"] += "/" + name
        except KeyError:
            print(f"Skeleton ID {i} not in graph")
            num_missing += 1
print()
print(f"{num_missing} skeleton IDs missing from graph")
print()
meta_data_df.head()

meta_data_df["Class 2"] = ""
num_missing = 0
for name, ids in subgroup_map.items():
    for i in ids:
        try:
            if meta_data_df.loc[i, "Class 2"] == "":
                meta_data_df.loc[i, "Class 2"] += name
            else:
                meta_data_df.loc[i, "Class 2"] += "/" + name
        except KeyError:
            print(f"Skeleton ID {i} not in graph")
            num_missing += 1
print()
print(f"{num_missing} skeleton IDs missing from graph")
print()
meta_data_df.head()

# Do some name remapping
print(np.unique(meta_data_df["Class 1"]))

class_labels = meta_data_df["Class 1"].values
name_map = {
    "": "Unk",
    "APL": "APL",
    "FAN": "FAN",
    "FB2N": "FB2N",
    "FBN": "FBN",
    "FFN": "FFN",
    "KC": "KC",
    "MBIN": "MBIN",
    "MBON": "MBON",
    "O_IPC": "O_IPC",
    "O_ITP": "O_ITP",
    "O_ITP/O_dSEZ": "O_ITP/dSEZ",
    "O_dSEZ": "O_dSEZ",
    "dSEZ": "dSEZ",
    "O_dSEZ/FB2N": "O_dSEZ/FB2N",
    "O_dSEZ/FFN": "O_dSEZ/FFN",
    "O_dSEZ/O_CA-LP": "O_dSEZ/CA-LP",
    "O_dVNC": "O_dVNC",
    "bLN": "bLN",
    "cLN": "cLN",
    "mPN": "mPN",
    "mPN/FFN": "mPN/FFN",
    "pLN": "pLN",
    "tPN": "tPN",
    "uPN": "uPN",
    "vPN": "vPN",
}
class_labels = np.array(itemgetter(*class_labels)(name_map))
meta_data_df["Class 1"] = class_labels

print(np.unique(meta_data_df["Class 2"]))

class_labels = meta_data_df["Class 2"].values
name_map = {
    "": "",
    "DAN": "DAN",
    "OAN": "OAN",
    "multimodal": "m",
    "olfactory": "o",
    "Duet": "D",
    "Trio": "T",
}
class_labels = np.array(itemgetter(*class_labels)(name_map))
meta_data_df["Class 2"] = class_labels


meta_data_df["Merge Class"] = ""
for i in meta_data_df.index.values:
    merge_class = meta_data_df.loc[i, "Class 1"]
    if meta_data_df.loc[i, "Class 2"] != "":
        merge_class += "-" + meta_data_df.loc[i, "Class 2"]
    meta_data_df.loc[i, "Merge Class"] = merge_class

side_labels = meta_data_df["Hemisphere"].values
name_map = {" mw right": "R", " mw left": "L"}
side_labels = np.array(itemgetter(*side_labels)(name_map))
meta_data_df["Hemisphere"] = side_labels

#%% show that there are some duplicated
all_labeled_ids = []
for name, vals in group_map.items():
    all_labeled_ids += vals

labels, counts = np.unique(all_labeled_ids, return_counts=True)


#%%
meta_data_dict = meta_data_df.to_dict(orient="index")
print(meta_data_df.head())

input_counts_path = data_path / data_date / (input_counts_file + ".csv")
input_counts_df = pd.read_csv(input_counts_path, index_col=0)
cols = input_counts_df.columns.values
cols = [str(c).strip(" ") for c in cols]
input_counts_df.columns = cols
print(input_counts_df.head())


def df_to_nx(df, meta_data_dict):
    c = df.columns.values
    c = c.astype(int)
    r = df.index.values
    df.columns = c
    if not (c == r).all():
        raise ValueError("Mismatching df indexing")
    graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
    nx.set_node_attributes(graph, meta_data_dict)
    return graph


#%% Import the raw graphs
nx_graphs_raw = {}
df_graphs_raw = {}
for graph_type in graph_types:
    print(graph_type)
    edgelist_path = data_path / data_date / (graph_type + ".csv")
    adj = pd.read_csv(edgelist_path, index_col=0)
    graph = df_to_nx(adj, meta_data_dict)
    nx_graphs_raw[graph_type] = graph
    df_graphs_raw[graph_type] = adj

    gridplot([adj.values], title=graph_type)
    print()


#%% Normalize weights for the raw graphs
df_graphs_norm = {}
nx_graphs_norm = {}

input_counts = input_counts_df["axon_inputs"].values
input_counts[input_counts == 0] = 1
for graph_type in ["axon-axon", "dendrite-axon"]:
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
    gridplot([df_adj_norm.values], title=graph_type)
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
    gridplot([df_adj_norm.values], title=graph_type)
    nx_graphs_norm[graph_type] = graph
    print()


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

#%% verify things are right
for name, graph_wrote in zip(save_names, out_graphs):
    print(name)
    graph_read = nx.read_graphml(output_path / (name + ".graphml"))
    adj_read = nx.to_numpy_array(graph_read)
    adj_wrote = nx.to_numpy_array(graph_wrote)
    print(np.array_equal(adj_read, adj_wrote))
    graph_loader = load_networkx(name)
    adj_loader = nx.to_numpy_array(graph_loader)
    print(np.array_equal(adj_wrote, adj_loader))
    print()

