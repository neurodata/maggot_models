import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from src.data import load_metagraph
from src.io import savecsv

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


mg = load_metagraph("G", version="2020-05-08")
ids = mg.meta.index.values

# load connectors
connector_path = "maggot_models/data/processed/2020-05-08/connectors.csv"
connectors = pd.read_csv(connector_path)

compartment = "dendrite"
direction = "postsynaptic"
subsample = False


def filter_connectors(connectors, ids, direction, compartment):
    label_connectors = connectors[connectors[f"{direction}_to"].isin(ids)]
    label_connectors = label_connectors[
        label_connectors[f"{direction}_type"] == compartment
    ]
    label_connectors = label_connectors[
        ~label_connectors["connector_id"].duplicated(keep="first")
    ]
    return label_connectors


select_connectors = filter_connectors(connectors, ids, direction, compartment)
connector_ids = select_connectors["connector_id"].values

X = select_connectors[["x", "y", "z"]].values

if subsample != 0:
    inds = np.random.choice(len(X), size=subsample, replace=False)
    X = X[inds]
    connector_ids = connector_ids[inds]

pdists = pairwise_distances(X, n_jobs=-1)

pdist_df = pd.DataFrame(data=pdists, index=connector_ids, columns=connector_ids)
savename = f"all-pdists-compartment={compartment}-direction={direction}"
stashcsv(pdist_df, savename)
