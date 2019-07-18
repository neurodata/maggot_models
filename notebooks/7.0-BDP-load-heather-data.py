#%%
from pathlib import Path
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graspy.plot import heatmap
import pandas as pd

data_path = Path("maggot_models/data/raw/20190615_mw")
file_name = "mw_20190615_G.graphml"
file_path = data_path / file_name


graph = nx.read_graphml(file_path)
node_data = list(graph.nodes.data())
node_data
#%%
names, data = zip(*node_data)
meta_df = pd.DataFrame(data)
meta_df
meta_df.to_csv(data_path / "BP_20190615mw_full_meta.csv")
#%%
unique_classes = np.unique(meta_df["Class"].values)

mb_classes = []

name_map = {
    "CN": "C/LH",
    "DANs": "I",
    "KCs": "KC",
    "LHN": "C/LH",
    "LHN; CN": "C/LH",
    "MBINs": "I",
    "MBON": "O",
    "MBON; CN": "O",
    "OANs": "I",
    "ORN mPNs": "P",
    "ORN uPNs": "P",
    "tPNs": "P",
    "vPNs": "P",
    "Unidentified": "U",
    "Other": "Ot",
}
og_classes = meta_df["Class"]
my_classes = np.array(itemgetter(*og_classes)(name_map))
meta_df["simple_class"] = my_classes

mb_keys = np.array(["KC", "P", "O", "I"])

mb_df = meta_df[meta_df["simple_class"].isin(mb_keys)]
left_mb_df = mb_df[mb_df["Hemisphere"] == "left"]
right_mb_df = mb_df[mb_df["Hemisphere"] == "right"]

mb_df

#%%
left_names = left_mb_df["Name"]
right_names = right_mb_df["Name"]
left_inds = left_mb_df.index.values
right_inds = right_mb_df.index.values


def strip(string):
    string = string.lower()
    string = string.replace(";", "")
    string = string.replace(",", "")
    string = string.replace("left", "")
    string = string.replace("right", "")
    string = string.replace("?", "")
    string = string.replace(" (bala1/2)", "")
    string = string.replace(" r", "")
    string = string.replace(" l", "s")
    return string


left_names = np.array(list(map(strip, left_names)))
left_names

right_names = np.array(list(map(strip, right_names)))
right_names

in_both_names = np.intersect1d(left_names, right_names)
in_both_names
print("Number of names in both")
print(in_both_names.shape)

#%%
in_right_names = np.isin(left_names, right_names)
left_not_in_right = left_names[~in_right_names]
left_not_in_right
print(f"{len(left_not_in_right)} cells in left not in right")

in_left_names = np.isin(right_names, left_names)
right_not_in_left = right_names[~in_left_names]
right_not_in_left
print(f"{len(right_not_in_left)} cells in right not in left")


# unpaired_keys = [
#     "kc no pair",
#     "kc young",
#     "kc very young",
#     "kc young no claws",
#     "kc no claws",
#     "kc young no claw",
# ]
# in_unpaired = np.isin(left_not_in_right, unpaired_keys)
# print("Cells in left, not in right")
# print(left_not_in_right[~in_unpaired])
# print()
# print("Cells in right, not in left")
# in_unpaired = np.isin(right_not_in_left, unpaired_keys)
# print(right_not_in_left[~in_unpaired])


#%%
left_names[in_right_names]
right_names[in_left_names]
left_matched_inds = left_inds[in_right_names]
right_matched_inds = right_inds[in_left_names]

#%%
left_matched_df = meta_df.loc[left_matched_inds]
right_matched_df = meta_df.loc[right_matched_inds]


def format_columns(string, side="left"):
    string = side + "_" + string
    return string


def format_left(string):
    return format_columns(string, "left")


def format_right(string):
    return format_columns(string, "right")


left_matched_df.rename(columns=format_left, inplace=True)
right_matched_df.rename(columns=format_right, inplace=True)

merge_df = pd.concat(
    (left_matched_df.reset_index(drop=True), right_matched_df.reset_index(drop=True)),
    axis=1,
    join="outer",
    ignore_index=False,
)
merge_df
base_path = "/Users/bpedigo/JHU_code/maggot_models/maggot_models/notebooks/"
merge_df.to_csv(base_path + "matched.csv", index=False)
#%%
left_unmatched_inds = left_inds[~in_right_names]
right_unmatched_inds = right_inds[~in_left_names]
left_unmatched_df = meta_df.loc[left_unmatched_inds]
right_unmatched_df = meta_df.loc[right_unmatched_inds]
left_unmatched_df.to_csv(base_path + "left_unmatched.csv", index=False)
right_unmatched_df.to_csv(base_path + "right_unmatched.csv", index=False)
#%%
