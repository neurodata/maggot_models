import colorcet as cc
import numpy as np

CLASS_IND_DICT = {
    "KC": 0,
    "KC-1claw": 28,
    "KC-2claw": 32,
    "KC-3claw": 92,
    "KC-4claw": 91,
    "KC-5claw": 78,
    "KC-6claw": 61,
    "APL": 24,
    "MBIN": 121,
    "MBIN-DAN": 58,
    "MBIN-OAN": 5,
    "MBON": 172,  # 43,
    "sens-AN": 1,
    "sens-MN": 12,
    "sens-ORN": 51,
    "sens-PaN": 76,
    "sens-photoRh5": 84,
    "sens-photoRh6": 106,
    "sens-thermo;AN": 55,
    "sens-vtd": 145,
    "mPN-multi": 3,
    "mPN-olfac": 88,
    "mPN;FFN-multi": 3,
    "tPN": 186,
    "uPN": 36,
    "vPN": 225,
    "pLN": 57,
    "bLN-Duet": 216,
    "bLN-Trio": 8,
    "cLN": 232,
    "FAN": 2,
    "FB2N": 21,
    "FBN": 50,
    "FFN": 52,
    "O_dSEZ;FB2N": 21,
    "O_dSEZ;FFN": 52,
    "O_CA-LP": 191,
    "O_IPC": 42,
    "O_ITP": 211,
    "O_dSEZ": 26,
    "O_dVNC": 38,
    "unk": 190,
}

names = []
color_inds = []

for key, val in CLASS_IND_DICT.items():
    names.append(key)
    color_inds.append(val)

colors = np.array(cc.glasbey_light)[color_inds]
CLASS_COLOR_DICT = dict(zip(names, colors))
