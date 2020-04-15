import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def cluster_pairplot(X, model, labels=None, equal=False):
    k = model.n_components
    n_dims = X.shape[1]

    if colors is None:
        colors = sns.color_palette("tab10", n_colors=k, desat=0.7)

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels  #
    pred = predict(X, left_inds, right_inds, model, relabel=False)
    data["pred"] = pred

    print(len(np.unique(pred)))

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=CLASS_COLOR_DICT,
                )
                make_ellipses(model, ax, i, j, colors, fill=False, equal=equal)
                if left_pair_inds is not None and right_pair_inds is not None:
                    add_connections(
                        data.iloc[left_pair_inds.values, j],
                        data.iloc[right_pair_inds.values, j],
                        data.iloc[left_pair_inds.values, i],
                        data.iloc[right_pair_inds.values, i],
                        ax=ax,
                    )

            if i > j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="pred",
                    palette=colors,
                )
                make_ellipses(model, ax, i, j, colors, fill=True, equal=equal)

    plt.tight_layout()
    return fig, axs
