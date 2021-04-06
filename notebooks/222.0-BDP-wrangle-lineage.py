#%%
import pymaid
import pandas as pd
import numpy as np
from src.pymaid import start_instance

start_instance()


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
    print(f"Applying annotations under meta annotation {key}...")
    annot_df = pymaid.get_annotated(key)

    series_ids = []

    for annot_name in annot_df["name"]:
        print("\t" + annot_name)
        indicator = get_indicator_from_annotation(annot_name, filt=filt)
        series_ids.append(indicator)
    return pd.concat(series_ids, axis=1, ignore_index=False)


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

print(f"Maximum number of lineage annotations: {lineage_df.sum(axis=1).max()}")
