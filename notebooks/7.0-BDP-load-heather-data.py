#%%
from pathlib import Path
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graspy.plot import heatmap
import pandas as pd

data_path = Path("maggot_models/data/raw/20190615_mw")
file_name = "graphs_mw_20190615.graphml"
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

in_left_names = np.isin(right_names, left_names)
right_not_in_left = right_names[~in_left_names]
right_not_in_left
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
