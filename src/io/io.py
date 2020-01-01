import matplotlib.pyplot as plt
import os
from pathlib import Path

# def stashfig(
#     name, foldername=None, default_fmt="png", default_dpi=150, save_on=True, **kws
# ):
#     if save_on:
#         savefig(name, foldername=foldername, fmt=default_fmt, dpi=default_dpi, **kws)


def savefig(
    name,
    fmt="png",
    dpi=150,
    foldername=None,
    subfoldername="figs",
    pathname="./maggot_models/notebooks/outs",
    bbox_inches="tight",
    pad_inches=0.5,
    save_on=True,
    **kws,
):
    if save_on:
        path = Path(pathname)
        if foldername is not None:
            path = path / foldername
            if not os.path.isdir(path):
                os.mkdir(path)
            if subfoldername is not None:
                path = path / subfoldername
                if not os.path.isdir(path):
                    os.mkdir(path)
        savename = path / str(name + "." + fmt)
        plt.savefig(
            savename,
            fmt=fmt,
            facecolor="w",
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            dpi=dpi,
            **kws,
        )
        print(f"Saved figure to {savename}")
