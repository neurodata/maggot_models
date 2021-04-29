import seaborn as sns
from .visualize import add_connections
import matplotlib.pyplot as plt
import pandas as pd
import colorcet as cc
import numpy as np


def plot_pairs(
    X,
    labels,
    n_show=8,
    model=None,
    left_pair_inds=None,
    right_pair_inds=None,
    equal=False,
    palette=None,
):
    """Plots pairwise dimensional projections, and draws lines between known pair neurons

    Parameters
    ----------
    X : [type]
        [description]
    labels : [type]
        [description]
    model : [type], optional
        [description], by default None
    left_pair_inds : [type], optional
        [description], by default None
    right_pair_inds : [type], optional
        [description], by default None
    equal : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if n_show is not None:
        n_dims = n_show
    else:
        n_dims = X.shape[1]

    if palette is None:
        uni_labels = np.unique(labels)
        palette = dict(zip(uni_labels, cc.glasbey_light))

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X[:, :n_dims], columns=[str(i) for i in range(n_dims)])
    data["label"] = labels

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=str(j),
                    y=str(i),
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=palette,
                )
                if left_pair_inds is not None and right_pair_inds is not None:
                    add_connections(
                        data.iloc[left_pair_inds, j],
                        data.iloc[right_pair_inds, j],
                        data.iloc[left_pair_inds, i],
                        data.iloc[right_pair_inds, i],
                        ax=ax,
                    )

    plt.tight_layout()
    return fig, axs