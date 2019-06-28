#%%
from src.data import load_new_left, load_new_right
import numpy as np

_, _, left_names = load_new_left(return_names=True)
_, _, right_names = load_new_right(return_names=True)
left_names
right_names


def strip(string):
    string = string.lower()
    string = string.replace(";", "")
    string = string.replace(",", "")
    string = string.replace("left", "")
    string = string.replace("right", "")
    string = string.replace("?", "")
    return string


left_names = np.array(list(map(strip, left_names)))
left_names

right_names = np.array(list(map(strip, right_names)))
right_names

in_both_names = np.intersect1d(left_names, right_names)
in_both_names
print(in_both_names.shape)

in_right_names = np.isin(left_names, right_names)
left_not_in_right = left_names[~in_right_names]
left_not_in_right
left_not_in_right.shape

in_left_names = np.isin(right_names, left_names)
right_not_in_left = right_names[~in_left_names]
#%%
unpaired_keys = [
    "kc no pair",
    "kc young",
    "kc very young",
    "kc young no claws",
    "kc no claws",
    "kc young no claw",
]
in_unpaired = np.isin(left_not_in_right, unpaired_keys)
print(left_not_in_right[~in_unpaired])
print()
in_unpaired = np.isin(right_not_in_left, unpaired_keys)
print(right_not_in_left[~in_unpaired])
