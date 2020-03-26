import json
from os import listdir
from pathlib import Path

import numpy as np

import pymaid
from src.pymaid import start_instance


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


start_instance()
meta_annot_df = pymaid.get_annotated("mw neuron groups")


base_path = Path("./maggot_models/data/raw/Maggot-Brain-Connectome/")
data_date_groups = "2020-03-09"
class_data_folder = base_path / f"neuron-groups/{data_date_groups}"
raw_json_path = (
    "maggot_models/data/raw/Maggot-Brain-Connectome/neuron-groups/2020-03-09"
)
group_files = listdir(class_data_folder)
new_group_files = []
for f in group_files:
    if f.endswith(".json"):
        new_group_files.append(f)
group_files = new_group_files
group_files.remove("all-neurons-with-sensories-2020-01-14.json")

# only used to make the dict below by hand
names = [remove_date(f) for f in group_files]
new_names = meta_annot_df["name"].values


name_map = {
    "CAT-is_brain": "brain neurons",
    "O_dUnk": "dUnk",
    "sens_subclass_photoRh6": "sens subclass_photoRh6",
    "bLN": "broad LN",
    "mPN": "mPN",
    "KC_subclass_4claw": "KC_subclass_4claw",
    "KC": "KC",
    "sens_subclass_PaN": "sens subclass_PaN",
    "keystone": "keystone",
    "CAT-is_pdiff": "partially differentiated",
    "subclass_IPC": "RG subclass_IPC",
    "subclass_OAN": "MBIN subclass_OAN",
    "vPN": "vPN",
    "cLN": "choosy LN",
    "CAT-is_sink": "sink",
    "CAT-is_usplit": "unsplittable",
    "sens_subclass_AN": "sens subclass_AN",
    "subclass_DAN": "MBIN subclass_DAN",
    "FBN": "FBN",
    "sens_subclass_photoRh5": "sens subclass_photoRh5",
    "CN2": "CN2",
    "LON": "LON",
    "subclass_ITP": "RG subclass_ITP",
    "motor": "motor",
    "sens": "sens",
    "KC_subclass_1claw": "KC_subclass_1claw",
    "sens_subclass_ORN": "sens subclass_ORN",
    "hemisphere-L": "left",
    "subclass_olfac": "mPN subclass_olfactory",
    "LHN": "LHN",
    "KC_subclass_2claw": "KC_subclass_2claw",
    "FB2N": "FB2N",
    "O_RG": "RG",
    "CAT-is_LNpre": "preliminary LN",
    "MBIN": "MBIN",
    "KC_subclass_3claw": "KC_subclass_3claw",
    "sens_subclass_thermo": "sens subclass_thermo",
    "APL": "APL",
    "FAN": "FAN",
    "CAT_is_sink": "sink",
    "CX": "CX",
    "motor_subclass_mVAN": "motor subclass_VAN",
    "FFN": "FFN",
    "tPN": "tPN",
    "sens_subclass_MN": "sens subclass_MN",
    "subclass_CA-LP": "RG subclass_CA-LP",
    "LHN2": "LHN2",
    "subclass_Duet": "broad LN subclass_Duet",
    "motor_subclass_mMN": "motor subclass_MN",
    "subclass_multi": "mPN subclass_multi",
    "KC_subclass_6claw": "KC_subclass_6claw",
    "sens_subclass_vtd": "sens subclass_vtd",
    "hemisphere-R": "right",
    "MBON": "MBON",
    "O_dVNC": "dVNC",
    "O_dSEZ": "dSEZ",
    "uPN": "uPN",
    "motor_subclass_mAN": "motor subclass_AN",
    "KC_subclass_5claw": "KC_subclass_5claw",
    "motor_subclass_mPaN": "motor subclass_PaN",
    "subclass_Trio": "broad LN subclass_Trio",
    "CN": "CN",
    "pLN": "picky LN",
}


diffs = []
for i, (old_name, new_name) in enumerate(name_map.items()):
    f = group_files[i]
    print(old_name + " from file " + f)
    print(new_name)

    with open(class_data_folder / f, "r") as json_file:
        temp_dict = json.load(json_file)
        old_ids = extract_ids(temp_dict)

    new_ids = pymaid.get_skids_by_annotation("mw " + new_name)

    diff = np.setdiff1d(old_ids, new_ids)
    diffs.append(diff)
    print(f"{len(diff)} IDs disagree")

    print()

print("Number of annotations with disagreements:")
print(np.count_nonzero(diffs))
