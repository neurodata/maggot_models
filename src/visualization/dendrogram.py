import matplotlib.pyplot as plt
import numpy as np

from .manual_colors import CLASS_COLOR_DICT
from .visualize import palplot, remove_spines


def get_mid_map(full_meta, leaf_key=None, bilat=False, gap=10):
    if not bilat:
        meta = full_meta[full_meta["hemisphere"] == "L"].copy()
    else:
        meta = full_meta.copy()

    sizes = meta.groupby([leaf_key, "merge_class"], sort=False).size()

    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    left_mid_map = dict(zip(uni_labels, mids))
    if bilat:
        first_mid_map = {}
        for k in left_mid_map.keys():
            left_mid = left_mid_map[k]
            first_mid_map[k + "-"] = left_mid
        return first_mid_map

    # right
    meta = full_meta[full_meta["hemisphere"] == "R"].copy()

    sizes = meta.groupby([leaf_key, "merge_class"], sort=False).size()

    # uni_labels = np.unique(labels)
    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    right_mid_map = dict(zip(uni_labels, mids))

    keys = list(set(list(left_mid_map.keys()) + list(right_mid_map.keys())))
    first_mid_map = {}
    for k in keys:
        left_mid = -1
        right_mid = -1
        if k in left_mid_map:
            left_mid = left_mid_map[k]
        if k in right_mid_map:
            right_mid = right_mid_map[k]

        first_mid_map[k + "-"] = max(left_mid, right_mid)
    return first_mid_map


def calc_bar_params(sizes, label, mid, palette=None):
    if palette is None:
        palette = CLASS_COLOR_DICT
    heights = sizes.loc[label]
    n_in_bar = heights.sum()
    offset = mid - n_in_bar / 2
    starts = heights.cumsum() - heights + offset
    colors = np.vectorize(palette.get)(heights.index)
    return heights, starts, colors


def get_last_mids(label, last_mid_map):
    last_mids = []
    if label + "-" in last_mid_map:
        last_mids.append(last_mid_map[label + "-"])
    if label + "-0" in last_mid_map:
        last_mids.append(last_mid_map[label + "-0"])
    if label + "-1" in last_mid_map:
        last_mids.append(last_mid_map[label + "-1"])
    if label in last_mid_map:
        last_mids.append(last_mid_map[label])
    if len(last_mids) == 0:
        print(label + " has no anchor in mid-map")
    return last_mids


def draw_bar_dendrogram(
    meta,
    ax,
    first_mid_map,
    lowest_level=7,
    width=0.5,
    draw_labels=False,
    color_key="merge_class",
    color_order="sf",
):
    meta = meta.copy()
    last_mid_map = first_mid_map
    line_kws = dict(linewidth=1, color="k")
    for level in np.arange(lowest_level + 1)[::-1]:
        x = level
        # mean_in_cluster = meta.groupby([f"lvl{level}_labels", color_key])["sf"].mean()
        meta = meta.sort_values(
            [f"lvl{level}_labels", f"{color_key}_{color_order}_order", color_key],
            ascending=False,
        )
        sizes = meta.groupby([f"lvl{level}_labels", color_key], sort=False).size()

        uni_labels = sizes.index.unique(level=0)  # these need to be in the right order

        mids = []
        for ul in uni_labels:
            if not isinstance(ul, str):
                ul = str(ul)  # HACK
            last_mids = get_last_mids(ul, last_mid_map)
            grand_mid = np.mean(last_mids)

            heights, starts, colors = calc_bar_params(sizes, ul, grand_mid)

            minimum = starts[0]
            maximum = starts[-1] + heights[-1]
            mid = (minimum + maximum) / 2
            mids.append(mid)

            # draw the bars
            for i in range(len(heights)):
                ax.bar(
                    x=x,
                    height=heights[i],
                    width=width,
                    bottom=starts[i],
                    color=colors[i],
                )
                if (level == lowest_level) and draw_labels:
                    ax.text(
                        x=lowest_level + 0.5, y=mid, s=ul, verticalalignment="center"
                    )

            # draw a horizontal line from the middle of this bar
            if level != 0:  # dont plot dash on the last
                ax.plot([x - 0.5 * width, x - width], [mid, mid], **line_kws)

            # line connecting to children clusters
            if level != lowest_level:  # don't plot first dash
                ax.plot(
                    [x + 0.5 * width, x + width], [grand_mid, grand_mid], **line_kws
                )

            # draw a vertical line connecting the two child clusters
            if len(last_mids) == 2:
                ax.plot([x + width, x + width], last_mids, **line_kws)

        last_mid_map = dict(zip(uni_labels, mids))
    remove_spines(ax)


def draw_leaf_dendrogram(meta, ax, lowest_level=7, width=0.5, draw_labels=False):
    leaf_sizes = meta.groupby(
        [f"lvl{lowest_level}_labels", "merge_class"], sort=False
    ).size()
    uni_labels = leaf_sizes.index.unique(0)
    # uni_labels = "u"

    first_mid_map = dict(zip(uni_labels, np.arange(len(uni_labels)) + 0.5))
    last_mid_map = first_mid_map
    line_kws = dict(linewidth=1, color="k")
    for level in np.arange(lowest_level + 1)[::-1]:
        x = level
        sizes = meta.groupby([f"lvl{level}_labels", "merge_class"], sort=False).size()

        uni_labels = sizes.index.unique(0)  # these need to be in the right order

        mids = []
        for ul in uni_labels:
            last_mids = get_last_mids(ul, last_mid_map)
            grand_mid = np.mean(last_mids)

            heights, starts, colors = calc_bar_params(sizes, ul, grand_mid)

            minimum = starts[0]
            maximum = starts[-1] + heights[-1]
            mid = (minimum + maximum) / 2
            mids.append(mid)

            if level == lowest_level:
                # draw the bars
                for i in range(len(heights)):
                    draw_heights = heights / heights.sum()
                    draw_starts = starts / heights.sum()
                    ax.barh(
                        y=mid,
                        width=draw_heights[i],
                        height=0.5,
                        left=draw_starts[i] - draw_starts[0] + level,
                        color=colors[i],
                    )
                    if draw_labels:
                        ax.text(
                            x=lowest_level + 0.5,
                            y=mid,
                            s=ul,
                            verticalalignment="center",
                        )

            # draw a horizontal line from the middle of this bar
            if level != 0:  # dont plot dash on the last
                # ax.plot([x - 0.5 * width, x - width], [mid, mid], **line_kws)
                ax.plot([x, x - width], [mid, mid], **line_kws)

            # line connecting to children clusters
            if level != lowest_level:  # don't plot first dash
                # ax.plot(
                #     [x + 0.5 * width, x + width], [grand_mid, grand_mid], **line_kws
                # )
                ax.plot([x, x + width], [grand_mid, grand_mid], **line_kws)

            # draw a vertical line connecting the two child clusters
            if len(last_mids) == 2:
                ax.plot([x + width, x + width], last_mids, **line_kws)

        last_mid_map = dict(zip(uni_labels, mids))
    remove_spines(ax)
    ax.set_yticks([])
    ax.set_xticks(np.arange(lowest_level + 1))
    return first_mid_map


def plot_single_dendrogram(
    meta, ax, lowest_level=7, gap=10, width=0.5, draw_labels=False, bars=True
):
    leaf_key = f"lvl{lowest_level}_labels"
    n_leaf = meta[leaf_key].nunique()
    n_cells = len(meta)

    first_mid_map = get_mid_map(meta, leaf_key=leaf_key, bilat=True)

    ax.set_ylim((-gap, (n_cells + gap * n_leaf)))
    ax.set_xlim((-0.5, lowest_level + 1 + 0.5))

    if bars:
        draw_bar_dendrogram(
            meta, ax, first_mid_map, lowest_level=lowest_level, draw_labels=draw_labels
        )
    else:
        draw_leaf_dendrogram(
            meta, ax, first_mid_map, lowest_level=7, draw_labels=draw_labels
        )

    ax.set_yticks([])
    ax.set_xticks(np.arange(lowest_level + 1))
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Level")

    # add a scale bar in the bottom left
    ax.bar(x=0, height=100, bottom=0, width=width, color="k")
    ax.text(x=0.35, y=0, s="100 neurons")

    return first_mid_map


def plot_double_dendrogram(
    meta, axs, lowest_level=7, gap=10, width=0.5, color_order="sf", make_flippable=False
):
    leaf_key = f"lvl{lowest_level}_labels"
    n_leaf = meta[f"lvl{lowest_level}_labels"].nunique()
    n_pairs = len(meta) // 2

    first_mid_map = get_mid_map(meta, leaf_key=leaf_key, bilat=False, gap=gap)

    # left side
    left_meta = meta[meta["hemisphere"] == "L"].copy()

    ax = axs[0]
    ax.set_title("Left", fontsize="x-large")
    ax.set_ylim((-gap, (n_pairs + gap * n_leaf)))
    ax.set_xlim((-0.5, lowest_level + 0.5))

    draw_bar_dendrogram(left_meta, ax, first_mid_map, color_order=color_order)

    ax.set_yticks([])
    ax.set_xticks(np.arange(lowest_level + 1))
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    xlabel = ax.set_xlabel("Level")
    # HACK big hackeroni alert
    if make_flippable:
        xlabel.set_rotation(180)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.tick

    # add a scale bar in the bottom left
    ax.bar(x=0, height=100, bottom=0, width=width, color="k")
    ax.text(x=0.35, y=0, s="100 neurons")

    # right side
    right_meta = meta[meta["hemisphere"] == "R"].copy()

    ax = axs[1]
    ax.set_title("Right", fontsize="x-large")
    ax.set_ylim((-gap, (n_pairs + gap * n_leaf)))
    ax.set_xlim((lowest_level + 0.5, -0.5))  # reversed x axis order to make them mirror

    draw_bar_dendrogram(right_meta, ax, first_mid_map, color_order=color_order)

    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    xlabel = ax.set_xlabel("Level")
    ax.set_xticks(np.arange(lowest_level + 1))
    if make_flippable:
        xlabel.set_rotation(180)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)


def plot_color_labels(meta, ax, color_order="sf", color_key="merge_class"):
    meta = meta.copy()
    # meta = meta.sort_values([f"merge_class_{color_order}_order"], ascending=False)
    meta = meta.sort_values(
        [f"{color_key}_{color_order}_order", color_key],
        ascending=False,
    )
    sizes = meta.groupby(["merge_class"], sort=False).size()
    uni_class = sizes.index.unique()
    counts = sizes.values
    count_map = dict(zip(uni_class, counts))
    names = []
    colors = []
    for key, val in count_map.items():
        names.append(f"{key} ({count_map[key]})")
        colors.append(CLASS_COLOR_DICT[key])
    colors = colors[::-1]  # reverse because of signal flow sorting
    names = names[::-1]
    palplot(len(colors), colors, ax=ax)
    ax.yaxis.set_major_formatter(plt.FixedFormatter(names))
