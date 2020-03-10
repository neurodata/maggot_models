import colorcet as cc
import numpy as np

import matplotlib.pyplot as plt

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
    "sens-thermo": 55,
    "sens-vtd": 145,
    "mPN-multi": 3,
    "mPN-olfac": 88,
    "mPN;FFN-multi": 3,
    "tPN": 186,
    "uPN": 36,
    "pLN": 57,
    "bLN-Duet": 216,
    "bLN-Trio": 8,
    "keystone": 157,
    "cLN": 232,
    "FAN": 2,
    "FB2N": 21,
    "FBN": 50,
    "FFN": 52,
    "O_dSEZ;FB2N": 21,
    "O_dSEZ;FFN": 52,
    "O_RG-CA-LP": 42,
    "O_RG-IPC": 42,
    "O_RG-ITP": 42,
    "O_RG": 42,
    "O_dSEZ": 124,
    "O_dSEZ;CN": 124,
    "O_dSEZ;LHN": 124,
    "LHN;O_dSEZ": 124,
    "O_dVNC": 38,
    "O_dVNC;CN": 38,
    "O_dVNC;O_RG": 38,
    "O_RG;O_dVNC": 38,
    "O_dVNC;O_dSEZ": 38,
    "O_dUnk": 105,
    "unk": 190,
    "LHN": 123,
    "LHN2": 65,
    "CN": 206,
    "CN2": 95,
    "LHN;CN": 108,
    "vPN": 33,
    "LON": 159,
    "CX": 228,
}


names = []
color_inds = []

for key, val in CLASS_IND_DICT.items():
    names.append(key)
    color_inds.append(val)

colors = np.array(cc.glasbey_light)[color_inds]
CLASS_COLOR_DICT = dict(zip(names, colors))
CLASS_COLOR_DICT["motor-mVAN"] = "#000000"
CLASS_COLOR_DICT["motor-mAN"] = "#000000"
CLASS_COLOR_DICT["motor-mPaN"] = "#000000"
CLASS_COLOR_DICT["motor-mMN"] = "#000000"


def plot_colors():
    from src.visualization import palplot

    fig, axs = plt.subplots(1, 6, figsize=(6, 10))
    n_per_col = 40
    for i, ax in enumerate(axs):
        pal = cc.glasbey_light[i * n_per_col : (i + 1) * n_per_col]
        palplot(n_per_col, pal, figsize=(1, 10), ax=ax, start=i * n_per_col)


def plot_class_colormap():
    from src.visualization import palplot

    # names = []
    # color_inds = []

    # for key, val in CLASS_IND_DICT.items():
    #     names.append(key)
    #     color_inds.append(val)
    # print(color_inds)
    # fig, ax = plt.subplots(1, 1, figsize=(3, 15))
    # colors = np.array(cc.glasbey_light)[color_inds]
    # palplot(len(colors), colors, ax=ax)
    # ax.yaxis.set_major_formatter(plt.FixedFormatter(names))
    names = []
    colors = []
    for key, val in CLASS_COLOR_DICT.items():
        names.append(key)
        colors.append(val)
    fig, ax = plt.subplots(1, 1, figsize=(3, 15))
    palplot(len(colors), colors, ax=ax)
    ax.yaxis.set_major_formatter(plt.FixedFormatter(names))


if __name__ == "__main__":
    plot_colors()
    plot_class_colormap()
    plt.show()
