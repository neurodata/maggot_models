#%%
from pathlib import Path
from operator import itemgetter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.plot import heatmap
from graspy.utils import get_lcc


data_path = Path("./maggot_models/data/raw/20190424_mw_brain_neurons")
adj_df = pd.read_csv(data_path / "brain-connectivity-matrix.csv", index_col=0)
adj = adj_df.values

meta_df = pd.read_csv(data_path / "brain_meta-data.csv")
meta_df.set_index("ID", inplace=True)
adj_df.columns = adj_df.columns.values.astype(int)
meta_inds = meta_df.index.values
adj_inds = adj_df.index.values

# check that they are sorted the same
(meta_inds == adj_inds).all()

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
}
og_classes = meta_df["Class"]
my_classes = np.array(itemgetter(*og_classes)(name_map))
meta_df["simple_class"] = my_classes
meta_df

#%%
mb_keys = np.array(["KC", "P", "O", "I"])

mb_df = meta_df[meta_df["simple_class"].isin(mb_keys)]
left_mb_df = mb_df[mb_df["Hemisphere"] == "left"]
right_mb_df = mb_df[mb_df["Hemisphere"] == "right"]

mb_inds = mb_df.index
left_mb_inds = left_mb_df.index
right_mb_inds = right_mb_df.index

mb_adj_df = adj_df.loc[mb_inds, mb_inds]

left_mb_adj_df = adj_df.loc[left_mb_inds, left_mb_inds]
right_mb_adj_df = adj_df.loc[right_mb_inds, right_mb_inds]

#%%
save_path = Path("./maggot_models/data/processed/")
base_name = "BP_20190424mw"
meta_df.to_csv(save_path / f"{base_name}_meta.csv")
adj_df.to_csv(save_path / f"{base_name}_adj.csv")

left_mb_df.to_csv(save_path / f"{base_name}_left_mb_meta.csv")
left_mb_adj_df.to_csv(save_path / f"{base_name}_left_mb_adj.csv")

right_mb_df.to_csv(save_path / f"{base_name}_right_mb_meta.csv")
right_mb_adj_df.to_csv(save_path / f"{base_name}_right_mb_adj.csv")

