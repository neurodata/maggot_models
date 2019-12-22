from src.utils import savefig


def stashfig(
    name, foldername=None, default_fmt="png", default_dpi=150, save_on=True, **kws
):
    if save_on:
        savefig(name, foldername=foldername, fmt=default_fmt, dpi=default_dpi, **kws)
