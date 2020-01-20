# %% [markdown]
# #
from pathlib import Path
import glob
import json
from os import listdir
from operator import itemgetter
import os
import random

import pandas as pd
import networkx as nx
import numpy as np
from graspy.plot import gridplot
from src.data import load_networkx
from src.io import saveskels

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

SAVESKELS = True


def r():
    return random.randint(0, 255)


def stashskel(name, ids, labels, colors=None, palette=None, **kws):
    saveskels(
        name,
        ids,
        labels,
        colors=colors,
        palette=None,
        foldername=FNAME,
        save_on=SAVESKELS,
        **kws,
    )


# File locations
base_path = Path("./maggot_models/data/raw/Maggot-Brain-Connectome/")

data_path = base_path / "4-color-matrices_Brain"

data_date_graphs = "2020-01-14"  # this is for the graph, not the annotations

graph_types = ["axon-axon", "axon-dendrite", "dendrite-axon", "dendrite-dendrite"]

data_date_groups = "2020-01-14"  # this is for the annotations

class_data_folder = base_path / f"neuron-groups/{data_date_groups}"

all_neuron_file = "all-neurons-with-sensories-2020-01-14.json"
left_file = "hemisphere-L-2020-1-14.json"
right_file = "hemisphere-R-2020-1-14.json"

input_counts_file = "input_counts"

pair_file = base_path / "pairs/bp-pairs-2020-01-13.csv"

output_path = Path(f"maggot_models/data/processed/{data_date_groups}")

skeleton_data_file = (
    data_path / Path(data_date_graphs) / "skeleton_id_vs_neuron_name.csv"
)

meta_df = pd.read_csv(output_path / "meta_data.csv", index_col=0)


# %%
mismatch_pair_left = []
mismatch_pair_right = []
for i in meta_df.index:
    if meta_df.loc[i, "Hemisphere"] == "L":
        pair_id = meta_df.loc[i, "Pair"]
        if pair_id != -1:
            self_class = meta_df.loc[i, "Merge Class"]
            other_class = meta_df.loc[pair_id, "Merge Class"]
            if self_class != other_class:
                print(f"Left ID is {i}")
                print(f"Right ID is {pair_id}")
                print(f"Left class is {self_class}")
                print(f"Right class is {other_class}")
                print()
                mismatch_pair_left.append(i)
                mismatch_pair_right.append(pair_id)

colors = []
ids = []
left_right_pairs = zip(mismatch_pair_left, mismatch_pair_right)
for (left, right) in left_right_pairs:
    hex_color = "#%02X%02X%02X" % (r(), r(), r())
    colors.append(hex_color)
    colors.append(hex_color)
    ids.append(left)
    ids.append(right)

stashskel("mismatched-pair-classes", ids, colors, colors=colors, palette=None)


# %%
