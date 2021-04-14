import seaborn as sns
import matplotlib as mpl

# plotting settings
rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "black",
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}


def set_theme(rc_dict=rc_dict, **kws):
    for key, val in rc_dict.items():
        mpl.rcParams[key] = val
    context = sns.plotting_context(context="talk", rc=rc_dict, **kws)
    sns.set_context(context)
