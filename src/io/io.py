import csv
import json
import os
import pickle
from operator import itemgetter
from pathlib import Path
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _handle_dirs(pathname, foldername, subfoldername):
    path = Path(pathname)
    if foldername is not None:
        path = path / foldername
        if not os.path.isdir(path):
            os.mkdir(path)
        if subfoldername is not None:
            path = path / subfoldername
            if not os.path.isdir(path):
                os.mkdir(path)
    return path


def savefig(
    name,
    format="png",
    dpi=300,
    foldername=None,
    subfoldername="figs",
    pathname="./maggot_models/notebooks/outs",
    bbox_inches="tight",
    pad_inches=0.5,
    save_on=True,
    transparent=False,
    print_out=True,
    **kws,
):
    if save_on:
        path = _handle_dirs(pathname, foldername, subfoldername)
        savename = path / str(name + "." + format)
        plt.savefig(
            savename,
            format=format,
            facecolor="white",
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            dpi=dpi,
            **kws,
        )
        if print_out:
            print(f"Saved figure to {savename}")


def saveobj(
    obj,
    name,
    foldername=None,
    subfoldername="objs",
    pathname="./maggot_models/notebooks/outs",
    save_on=True,
):
    if save_on:
        path = _handle_dirs(pathname, foldername, subfoldername)
        savename = path / str(name + ".pickle")
        with open(savename, "wb") as f:
            pickle.dump(obj, f)
            print(f"Saved object to {savename}")


def saveskels(
    name,
    ids,
    labels,
    colors=None,
    palette="tab10",
    foldername=None,
    subfoldername="jsons",
    pathname="./maggot_models/notebooks/outs",
    multiout=False,
    save_on=True,
    postfix="",
):
    """Take a list of skeleton ids and output as json file for catmaid

    Parameters
    ----------
    name : str
        filename to save output
    ids : list or array
        skeleton ids
    colors : list or array
        either a hexadecimal color for each skeleton or a label for each skeleton to be
        colored by palette
    palette : str or None, optional
        if not None, this is a palette specification to use to color skeletons
    """
    if save_on:
        uni_labels = np.unique(labels)
        n_labels = len(uni_labels)

        if colors is None:
            if isinstance(palette, str):
                pal = sns.color_palette(palette, n_colors=n_labels)
                pal = pal.as_hex()
            else:
                pal = palette
            # uni_labels = [int(i) for i in uni_labels]
            colormap = dict(zip(uni_labels, pal))
            colors = np.array(itemgetter(*colors)(colormap))

        opacs = np.array(len(ids) * [1])

        path = _handle_dirs(pathname, foldername, subfoldername)

        if multiout:

            for l in uni_labels:
                filename = path / str(name + "-" + str(l) + postfix + ".json")

                inds = np.where(labels == l)[0]

                spec_list = [
                    {"skeleton_id": int(i), "color": str(c), "opacity": float(o)}
                    for i, c, o in zip(ids[inds], colors[inds], opacs[inds])
                ]
                with open(filename, "w") as fout:
                    json.dump(spec_list, fout)
        else:
            spec_list = [
                {"skeleton_id": int(i), "color": str(c), "opacity": float(o)}
                for i, c, o in zip(ids, colors, opacs)
            ]
            filename = path / str(name + ".json")
            with open(filename, "w") as fout:
                json.dump(spec_list, fout)

        if palette is not None:
            # return (spec_list, colormap, pal)
            return spec_list
        else:
            return spec_list


def savecsv(
    df,
    name,
    foldername=None,
    subfoldername="csvs",
    pathname="./maggot_models/notebooks/outs",
    save_on=True,
):
    if save_on:
        path = _handle_dirs(pathname, foldername, subfoldername)
        savename = path / str(name + ".csv")
        df.to_csv(savename)
        print(f"Saved DataFrame to {savename}")


def savelol(
    lol,
    name,
    foldername=None,
    subfoldername="csvs",
    pathname="./maggot_models/notebooks/outs",
    save_on=True,
):
    path = _handle_dirs(pathname, foldername, subfoldername)
    savename = path / str(name + ".csv")
    with open(savename, "w") as f:
        wr = csv.writer(f)
        wr.writerows(lol)
    print(f"Saved list of lists to {savename}")


def readlol(
    name,
    foldername=None,
    subfoldername="csvs",
    pathname="./maggot_models/notebooks/outs",
):
    path = _handle_dirs(pathname, foldername, subfoldername)
    savename = path / str(name + ".csv")
    outer_list = []
    with open(savename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            outer_list.append([int(l) for l in line])
    return outer_list


def readcsv(
    name,
    foldername=None,
    subfoldername="csvs",
    pathname="./maggot_models/notebooks/outs",
    **kws,
):
    path = _handle_dirs(pathname, foldername, subfoldername)
    savename = path / str(name + ".csv")
    return pd.read_csv(savename, **kws)


def _write_walk_set(walk_set, f):
    for w in walk_set:
        str_walk = str(w)
        str_walk = str_walk.strip("[]")
        str_walk = str_walk.replace("'", "")
        str_walk = str_walk.replace(",", "")
        f.write(f"{str_walk}\n")
    f.write("\n")


def save_walks(
    walks, name="walks.txt", outpath=".", multidoc=False, shuffle_walks=True
):
    outpath = Path(outpath)
    outfile = outpath / name
    if not multidoc:
        walks = [walks]
    with open(outfile, "w") as f:
        for walk_set in walks:
            if shuffle_walks:
                shuffle(walk_set)
            _write_walk_set(walk_set, f)
    print(f"Saved walks to {outfile}")
