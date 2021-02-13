#%% Imports and file loading
import pprint
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import pymaid
from src.data import load_networkx
from src.io import savecsv
from src.pymaid import start_instance

output_name = "2021-01-29"
output_path = Path(f"maggot_models/data/processed/{output_name}")

# toggle for logging output to another file
sys.stdout = open(f"maggot_models/data/logs/{output_name}.txt", "w")

print(f"Data will save to: {output_path}")


# File locations
base_path = Path("./maggot_models/data/raw/Maggot-Brain-Connectome/")

data_path = base_path / "4-color-matrices_Brain"

data_date_graphs = "2021-01-29"
print(f"Data for adjacency matrices: {data_date_graphs}")

graph_types = ["axon-axon", "axon-dendrite", "dendrite-axon", "dendrite-dendrite"]

input_counts_file = "input_counts"

pair_file = "pairs/pairs-2020-05-08.csv"
print(f"Data for pairs: {pair_file}")
pair_file = base_path / pair_file

use_super = False

print()


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
    "RGN": 2,
    "dUnk": 2,
    "FBN": 3,
    "FAN": 3,
    "LHN2": 5,  # used to be 4
    "CN2": 6,  # used to be 5
    "FB2N": 3,
    "FFN": 4,  # used to be 4
    "MN2": 3,
    "AN2": 3,
    "vtd2": 3,
    "A00c": 1,
}


def priority(name, priority_map=priority_map):
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


def get_classes(meta, class_cols, fill_unk=False, check_priority=check_priority):
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


def get_indicator_from_annotation(annot_name, filt=None):
    ids = pymaid.get_skids_by_annotation(annot_name.replace("*", "\*"))
    if filt is not None:
        name = filt(annot_name)
    else:
        name = annot_name
    indicator = pd.Series(
        index=ids, data=np.ones(len(ids), dtype=bool), name=name, dtype=bool
    )
    return indicator


def df_from_meta_annotation(key, filt=None):
    print(f"Getting annotations under {key}:\n")
    annot_df = pymaid.get_annotated(key)

    series_ids = []

    for annot_name in annot_df["name"]:
        print("\t" + annot_name)
        indicator = get_indicator_from_annotation(annot_name, filt=filt)
        series_ids.append(indicator)
    print()
    return pd.concat(series_ids, axis=1, ignore_index=False)


# %% [markdown]
# ## load main groups as boolean columns


start_instance()  # creates a pymaid instance


def filt(name):
    name = name.replace("mw ", "")
    name = name.replace(" ", "_")
    return name


meta = df_from_meta_annotation("mw neuron groups", filt=filt)

output_meta = df_from_meta_annotation("mw brain outputs", filt=filt)
is_output = output_meta.any(axis=1)
meta.loc[is_output.index, "output"] = is_output

input_meta = df_from_meta_annotation("mw brain inputs", filt=filt)
is_input = input_meta.any(axis=1)
meta.loc[is_input.index, "input"] = is_input


# %% [markdown]
# ##
meta.fillna(False, inplace=True)

class1_name_map = {
    "APL": "APL",
    "dSEZ": "dSEZ",
    "dVNC": "dVNC",
    "RGN": "RGN",
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
    # "vtd_2ndOrder": "vtd2",
    "AN_2nd_order": "AN2",
    "MN_2nd_order": "MN2",
    "A00c": "A00c",
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


def indicator_union(annot_names, filt=None):
    indicators = []
    for annot_name in annot_names:
        indicator = get_indicator_from_annotation(annot_name)
        indicators.append(indicator)
    annot_indicator_df = pd.concat(indicators, axis=1, ignore_index=False)
    return annot_indicator_df.any(axis=1)  # going to be all true regardless, oh well


simple_class_map = {
    "sensory": [
        "mw ORN",
        "mw thermosensories",
        "mw photoreceptors",
        "mw AN sensories",
        "mw MN sensories",
        "mw v'td",
        "mw A00c",
    ],
    "PN": [
        "mw AN 2nd_order PN",
        "mw MN 2nd_order PN",
        "mw vtd 2nd_order PN",
        "mw thermo 2nd_order PN",
        "mw photo 2nd_order PN",
        "mw A00c 2nd_order PN",
        "mw ORN 2nd_order PN",
    ],
    "LN": [
        "mw MN 2nd_order LN",
        "mw ORN 2nd_order LN",
        "mw ORN_AN_MN 2nd_order LN",
        "mw photo 2nd_order LN",
    ],
    "LHN": ["mw LHN"],
    "MBIN": ["mw MBIN"],
    "KC": ["mw KC"],
    "MBON": ["mw MBON"],
    "FBN": ["mw FBN", "mw FB2N", "mw FAN"],
    "CN": ["mw CN"],
    "motorneuron": ["mw motor"],
    "pre-outputs": ["mw pre-dVNC", "mw pre-dSEZ", "mw pre-RGN"],
    "outputs": ["mw dVNC", "mw dSEZ", "mw RGN"],
    "sensory_2nd_order": [
        "mw A00c 2nd_order",
        "mw AN 2nd_order",
        "mw MN 2nd_order",
        "mw ORN 2nd_order",
        "mw photo 2nd_order",
        "mw thermo 2nd_order",
        "mw vtd 2nd_order",
    ],
}

# print("hi")
cols = []
for key, val in simple_class_map.items():
    # print("hi")
    print(val)
    indicator = indicator_union(val, filt=filt)
    indicator.name = key
    cols.append(indicator)
simple_class_df = pd.DataFrame(cols).T
simple_class_df = simple_class_df.reindex(meta.index)
simple_class_df.fillna(False, inplace=True)

simple_name_map = {
    "sensory": "Sens",
    "motorneuron": "Motr",
    "pre-outputs": "PreO",
    "outputs": "Outs",
    "sensory_2nd_order": "Sens2o",
}
simple_class_df.rename(simple_name_map, axis=1, inplace=True)

priority_map = {
    "Sens": 1,
    "PN": 2,
    "MBIN": 3,
    "KC": 4,
    "MBON": 5,
    "Motr": 6,
    "LHN": 7,
    "FBN": 8,
    "CN": 9,
    "dVNC": 10,
    "dSEZ": 11,
    "RGN": 12,
}
priority_map = {
    "Sens": 1,
    "LN": 2,
    "PN": 3,
    "MBIN": 4,
    "KC": 5,
    "MBON": 6,
    "Motr": 7,
    "LHN": 8,
    "FBN": 9,
    "CN": 10,
    "PreO": 11,
    "Out": 12,
    "Sens2o": 13,
}

check_priority = np.vectorize(lambda x: priority(x, priority_map=priority_map))

simple_class, all_simple_class, n_simple_class = get_classes(
    simple_class_df,
    simple_class_df.columns.values,
    fill_unk=True,
    check_priority=check_priority,
)

meta["simple_class"] = simple_class
meta["all_simple_class"] = all_simple_class
meta["n_simple_class"] = n_simple_class
#%%
meta.sort_values("n_simple_class", ascending=False)[
    ["simple_class", "all_simple_class"]
].head(20)
# %% [markdown]
# ##
print()
print("Class 1 unique values:")
pprint.pprint(dict(zip(*np.unique(all_class1, return_counts=True))))
print()
print("Class 2 unique values:")
pprint.pprint(dict(zip(*np.unique(all_class2, return_counts=True))))
print()
print("Simple class unique values:")
pprint.pprint(dict(zip(*np.unique(all_simple_class, return_counts=True))))
print()

# %% [markdown]
# ## Hemisphere
meta["hemisphere"] = "C"  # default is center
left_meta = meta[meta["left"]]
meta.loc[left_meta.index, "hemisphere"] = "L"
right_meta = meta[meta["right"]]
meta.loc[right_meta.index, "hemisphere"] = "R"

print("\n\n")
print("Center neurons:")
print(meta[meta["hemisphere"] == "C"])
print("\n\n")


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

meta["pair"] = -1
meta["pair_id"] = -1
meta.loc[left_to_right_df.index, "pair"] = left_to_right_df["rightid"]
meta.loc[right_to_left_df.index, "pair"] = right_to_left_df["leftid"]

meta.loc[left_to_right_df.index, "pair_id"] = left_to_right_df["pair_id"]
meta.loc[right_to_left_df.index, "pair_id"] = right_to_left_df["pair_id"]

#%% Fix places where L/R labels are not the same
print("\n\nFinding asymmetric L/R labels")
for i in range(len(meta)):
    my_id = meta.index[i]
    my_class = meta.loc[my_id, "class1"]
    partner_id = meta.loc[my_id, "pair"]
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
# # create the merge class annotation

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

# %% [markdown]
# ## Load lineages


def filt(string):
    string = string.replace("akira", "")
    string = string.replace("Lineage", "")
    string = string.replace("lineage", "")
    string = string.replace("*", "")
    string = string.strip("_")
    string = string.strip(" ")
    string = string.replace("_r", "")
    string = string.replace("_l", "")
    string = string.replace("right", "")
    string = string.replace("left", "")
    string = string.replace("unknown", "unk")
    return string


lineage_df = df_from_meta_annotation("Volker", filt=filt)

lineage_df = lineage_df.fillna(False)
data = lineage_df.values
row_sums = data.sum(axis=1)
lineage_df.loc[row_sums > 1, :] = False
check_row_sums = lineage_df.values.sum(axis=1)
assert check_row_sums.max() == 1

columns = lineage_df.columns
lineages = []
for index, row in lineage_df.iterrows():
    lineage = columns[row].values
    if len(lineage) < 1:
        lineage = "unk"
    else:
        lineage = lineage[0]
    lineages.append(lineage)
lineage_series = pd.Series(index=lineage_df.index, data=lineages)
lineage_series = lineage_series[lineage_series.index.isin(meta.index)]
meta["lineage"] = "unk"
meta.loc[lineage_series.index, "lineage"] = lineage_series.values

# %% [markdown]
# ## check for lineage mismatches in pairs

pair_meta = meta[meta["pair"] != -1]
pair_meta = pair_meta.sort_values(["pair_id", "hemisphere"])

pair_unk = 0
unk = []
pair_mismatch = 0
mismatch = []
for p in pair_meta["pair_id"].unique():
    pm = pair_meta[pair_meta["pair_id"] == p]
    uni_lin = pm["lineage"].unique()
    if ("unk" in uni_lin) and len(uni_lin) > 1:
        print(str(uni_lin) + " unk")
        pair_unk += 1
        unk.append(pm.index.values)
    elif len(uni_lin) > 1:
        print(str(uni_lin) + " mismatch")
        pair_mismatch += 1
        mismatch.append(pm.index.values)

mismatch = pd.DataFrame(mismatch)
savecsv(mismatch, "mismatch")
unk = pd.DataFrame(unk)
savecsv(unk, "unk")

#%%
input_counts_path = data_path / data_date_graphs / (input_counts_file + ".csv")
input_counts_df = pd.read_csv(input_counts_path, index_col=0)
cols = input_counts_df.columns.values
cols = [str(c).strip(" ") for c in cols]
input_counts_df.columns = cols

meta.loc[input_counts_df.index, "dendrite_input"] = input_counts_df["dendrite_inputs"]
meta.loc[input_counts_df.index, "axon_input"] = input_counts_df["axon_inputs"]

# %% [markdown]
# ## Add names
name_map = pymaid.get_names(meta.index.values)

meta["name"] = meta.index.map(lambda name: name_map[str(name)])

# %% [markdown]
# ## Deal with the supernode graph

if use_super:
    empty_types = {
        np.bool: False,
        np.float64: 0.0,
        np.int64: 0,
        np.object: "",
        bool: False,
    }

    def get_empty_val(dtype):
        for key, val in empty_types.items():
            if dtype == key:
                return val

    def make_empty_df_from(df, new_index=None):
        n_rows = len(new_index)
        series_dict = {}
        for c in df.columns:
            dtype = df[c].dtype
            empty_val = get_empty_val(dtype)
            series_dict[c] = pd.Series(
                [empty_val] * n_rows, dtype=dtype, index=new_index
            )
        return pd.DataFrame(series_dict)

    graph_type = "supernodes"
    edgelist_path = data_path / data_date_graphs / (graph_type + ".csv")
    adj = pd.read_csv(edgelist_path, index_col=0)

    supernode_names = [
        "Brain Hemisphere left",
        "Brain Hemisphere right",
        "SEZ_left",
        "SEZ_right",
        "T1_left",
        "T1_right",
        "T2_left",
        "T2_right",
        "T3_left",
        "T3_right",
        "A1_left",
        "A1_right",
        "A2_left",
        "A2_right",
        "A3_left",
        "A3_right",
        "A4_left",
        "A4_right",
        "A5_left",
        "A5_right",
        "A6_left",
        "A6_right",
        "A7_left",
        "A7_right",
        "A8_left",
        "A8_right",
    ]
    supernode_ids = np.arange(-100, -100 - len(supernode_names), step=-1)
    supernode_name_map = dict(zip(supernode_names, supernode_ids))
    super_meta = make_empty_df_from(meta, supernode_ids)

    super_meta["name"] = supernode_names
    super_meta["left"] = super_meta["name"].map(lambda name: "left" in name)
    super_meta["right"] = super_meta["name"].map(lambda name: "right" in name)
    left_meta = super_meta[super_meta["left"]]
    super_meta.loc[left_meta.index, "hemisphere"] = "L"
    right_meta = super_meta[super_meta["right"]]
    super_meta.loc[right_meta.index, "hemisphere"] = "R"

    def pair_mapper(skid):
        if skid % 2 == 0:
            return skid - 1
        else:
            return skid + 1

    super_meta["pair"] = super_meta.index.map(pair_mapper)
    max_pair_id = meta["pair_id"].max() + 1
    super_pair_ids = np.arange(len(supernode_names) / 2) + max_pair_id
    super_pair_ids = super_pair_ids.astype(np.int64)
    super_pair_ids = np.repeat(super_pair_ids, 2)
    super_meta["pair_id"] = super_pair_ids
    super_meta["super"] = True
    super_meta["class1"] = "super"

    def class2_mapper(name):
        if "brain" in name.lower():
            return "brain"
        elif "sez" in name.lower():
            return "sez"
        else:
            return "vnc"

    super_meta["class2"] = super_meta["name"].map(class2_mapper)
    super_meta["n_class2"] = 1
    super_meta["n_class1"] = 1
    super_meta["merge_class"] = super_meta["class1"] + "-" + super_meta["class2"]
    super_meta["lineage"] = "none"

    meta["super"] = False
    meta = pd.concat((meta, super_meta), axis=0)

#%% Import the raw graphs
print("Importing raw adjacency matrices:\n")


def index_mapper(idx):
    if use_super and (idx in supernode_names):
        return supernode_name_map[idx]
    else:
        return idx


all_used_ids = []
for graph_type in graph_types:
    edgelist_path = data_path / data_date_graphs / (graph_type + ".csv")
    adj = pd.read_csv(edgelist_path, index_col=0)
    idx = adj.index.values.astype(int)
    all_used_ids.append(idx)

if use_super:
    all_used_ids.append(supernode_ids)

all_used_ids = np.concatenate(all_used_ids)
all_used_ids = np.unique(all_used_ids)
meta = meta.reindex(all_used_ids, axis=0)

nx_graphs_raw = {}
df_graphs_raw = {}
for graph_type in graph_types:
    print(graph_type)
    edgelist_path = data_path / data_date_graphs / (graph_type + ".csv")
    adj = pd.read_csv(edgelist_path, index_col=0)
    adj.rename(index=index_mapper, columns=index_mapper, inplace=True)
    adj.columns = adj.columns.astype(int)
    adj = adj.reindex(index=meta.index, columns=meta.index, fill_value=0.0)
    meta_data_dict = meta.to_dict(orient="index")
    graph = df_to_nx(adj, meta_data_dict)
    nx_graphs_raw[graph_type] = graph
    df_graphs_raw[graph_type] = adj
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

all_adj_raw = np.zeros_like(adj.values)
for graph_type in graph_types[:4]:  # ignore supernode graph
    all_adj_raw += df_graphs_raw[graph_type].values

df_all_raw = pd.DataFrame(index=adj.index, columns=adj.columns, data=all_adj_raw)

nx_all_raw = df_to_nx(df_all_raw, meta_data_dict)

# all_adj_norm = all_adj_raw / total_input[np.newaxis, :]
# df_all_norm = pd.DataFrame(index=adj.index, columns=adj.columns, data=all_adj_norm)

# nx_all_norm = df_to_nx(df_all_norm, meta_data_dict)

#%% Save

print("Saving graphs:\n")
out_graphs = []
[out_graphs.append(i) for i in nx_graphs_raw.values()]
[print(i) for i in nx_graphs_raw.keys()]
save_names = ["Gaa", "Gad", "Gda", "Gdd"]
if use_super:
    save_names.append("Gs")
# [out_graphs.append(i) for i in nx_graphs_norm.values()]
# [print(i) for i in nx_graphs_norm.keys()]
# save_names += ["Gaan", "Gdan", "Gadn", "Gddn"]
out_graphs.append(nx_all_raw)
save_names.append("G")
# out_graphs.append(nx_all_norm)
# save_names.append("Gn")

for name, graph in zip(save_names, out_graphs):
    nx.write_graphml(graph, output_path / (name + ".graphml"))

for name, graph in zip(save_names, out_graphs):
    nx.write_weighted_edgelist(graph, output_path / (name + ".edgelist"))

meta.to_csv(output_path / "meta_data.csv")
meta.to_csv(output_path / "meta_data.tsv", sep="\t")

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


# %% [markdown]
# ## Connectors
print("\n\n\n\n")
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
    for i, n in enumerate(nl):
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
print("Getting connectors\n")
connectors = get_connectors(nl)

explode_cols = ["postsynaptic_to", "postsynaptic_to_node"]
index_cols = np.setdiff1d(connectors.columns, explode_cols)

print("Exploding connector DataFrame\n")
# explode the lists within the connectors dataframe
connectors = (
    connectors.set_index(list(index_cols)).apply(pd.Series.explode).reset_index()
)
# TODO figure out these nans
connectors = connectors[~connectors.isnull().any(axis=1)]
bad_connectors = connectors[connectors.isnull().any(axis=1)]
print(f"Connectors with errors: {len(bad_connectors)}\n")
connectors = connectors.astype(
    {
        "presynaptic_to": "int64",
        "presynaptic_to_node": "int64",
        "postsynaptic_to": "int64",
        "postsynaptic_to_node": "int64",
    }
)


print("Getting treenode compartment types\n")
treenode_types = get_treenode_types(nl, splits)


print("Applying treenode types to connectors\n")
connectors["presynaptic_type"] = connectors["presynaptic_to_node"].map(treenode_types)
connectors["postsynaptic_type"] = connectors["postsynaptic_to_node"].map(treenode_types)

connectors["in_subgraph"] = connectors["presynaptic_to"].isin(ids) & connectors[
    "postsynaptic_to"
].isin(ids)

print("Saving")
out_path = f"maggot_models/data/processed/{output_name}/"
connectors.to_csv(out_path + "connectors.csv")
print()

print("Done!")
sys.stdout.close()
