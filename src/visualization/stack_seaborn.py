from __future__ import division
from textwrap import dedent
import colorsys
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import warnings
from six import string_types
from six.moves import range

from seaborn import utils
from seaborn.axisgrid import FacetGrid
from seaborn.categorical import _BarPlotter, _CategoricalPlotter
from seaborn.categorical import factorplot as _factorplot


__all__ = ["countplot", "factorplot", "freqplot"]


class _StackBarPlotter(_BarPlotter):
    """ Stacked Bar Plotter
 
    A modification of the :mod:`seaborn._BarPlotter` object with the added ability of
    stacking bars either verticaly or horizontally. It takes the same arguments
    as :mod:`seaborn._BarPlotter` plus the following:
 
    Arguments
    ---------
    stack : bool
        Stack bars if true, otherwise returns equivalent barplot as
        :mod:`seaborn._BarPlotter`.
    """

    def draw_bars(self, ax, kws):
        """Draw the bars onto `ax`."""
        # Get the right matplotlib function depending on the orientation
        barfunc = ax.bar if self.orient == "v" else ax.barh
        barpos = np.arange(len(self.statistic))

        if self.plot_hues is None:

            # Draw the bars
            barfunc(
                barpos,
                self.statistic,
                self.width,
                color=self.colors,
                align="center",
                **kws,
            )

            # Draw the confidence intervals
            errcolors = [self.errcolor] * len(barpos)
            self.draw_confints(
                ax, barpos, self.confint, errcolors, self.errwidth, self.capsize
            )
        else:
            # Stack by hue
            for j, hue_level in enumerate(self.hue_names):

                barpos_prior = None if j == 0 else np.sum(self.statistic[:, :j], axis=1)

                # Draw the bars
                if self.orient == "v":
                    barfunc(
                        barpos,
                        self.statistic[:, j],
                        self.nested_width,
                        bottom=barpos_prior,
                        color=self.colors[j],
                        align="center",
                        label=hue_level,
                        **kws,
                    )
                elif self.orient == "h":
                    barfunc(
                        barpos,
                        self.statistic[:, j],
                        self.nested_width,
                        left=barpos_prior,
                        color=self.colors[j],
                        align="center",
                        label=hue_level,
                        **kws,
                    )

                # Draw the confidence intervals
                if self.confint.size:
                    confint = (
                        self.confint[:, j]
                        if j == 0
                        else np.sum(self.confint[:, :j], axis=1)
                    )
                    errcolors = [self.errcolor] * len(barpos)
                    self.draw_confints(
                        ax, barpos, confint, errcolors, self.errwidth, self.capsize
                    )


def countplot(
    x=None,
    y=None,
    hue=None,
    data=None,
    order=None,
    hue_order=None,
    orient=None,
    color=None,
    palette=None,
    saturation=0.75,
    dodge=True,
    stack=False,
    ax=None,
    **kwargs,
):
    """ Show the count of observations in each categorical bin using bars.
 
    The count plot is a normalization of a histogram across categories, as opposed
    to quantitative variables. The basic API and options are identical to those for
    :func:`barplot`, so you can compare counts across nested variables.
 
    Parameters
    ----------
    x, y, hue : str or array-like, optional
        Inputs for plotting long-form data.
    data : DataFrame, array, or list of arrays, optional
        Dataset for plotting. If `x` and `y` are absent, this is interpreted as wide-form.
        Otherwise, data is expected to be long-form.
    order, hue_order : list of str, optional
        Order to plot the categorical levels, otherwise the levels are inferred from the
        data object.
    orient : {"v", "h"}, optional
        Whether to plot bars vertically ("v") or horizontally ("h"). This can also be
        inferred from the dtype of the input variables, but can be used to specify when the
        "categorical" variable is a numeric or when plotting wide-form data.
    color : matplotlib color, optional
        Color for all of the elemnts, or seed for a gradient palette.
    palette : palette name, list, or dict, optional
        Colors to use for the different levels of the `hue` variable. Should be somthing that
        can be interpreted by `color_palette()` or a dictionary mapping hue levels to
        matplotlib colors.
    saturation :  float, optional
        Proportion of the original saturation to draw colors. Large patches often look better
        with slighlty desaturated colors, but set this to `1` if you want the plot colorss to
        perfectly match the input color spec.
    dodge : bool, optional
        When hue nesting is used, whether elements should be shifted along the categorical axis.
    stack : bool, optional
        When hue nesting is used, whether elements should be stacked ontop of each other. Note,
        dodge is set to False when stack is True.
    ax : matplotlib.axes, optional
        Axes object to draw the plot onto, otherwise uses the current axes.
    **kwargs : Other keyword arguments are passed through to `plt.bar` at draw time
 
    Examples
    --------
    .. plot::
        :context: close-figs
 
        >>> import schmeaborn as sns
        >>> titanic = sns.load_dataset("titanic")
        >>> ax = sns.freqplot(x="class", data=titanic)
 
    Show frequencies for two categorical variables:
 
    .. plot::
        :context: close-figs
 
        >>> ax = sns.freqplot(x="class", hue="who", data=titanic)
 
    Plot the bars horizontally:
 
    .. plot::
        :context: close-figs
 
        >>> ax = sns.freqplot(y="class", hue="who", data=titanic)
 
    Plot categories stacked:
 
    .. plot::
        :context: close-figs
 
        >>> ax = sns.freqplot(x="class", hue="who", stack=True, data=titanic)
    """

    # Define parameters for barplot
    if stack:
        dodge = False
    estimator = len
    ci = None
    n_boot = 0
    units = None
    errcolor = None
    errwidth = None
    capsize = None

    # Check orientation by input
    if x is None and y is not None:
        orient = "h"
        x = y
    elif y is None and x is not None:
        orient = "v"
        y = x
    elif x is not None and y is not None:
        raise TypeError("Cannot pass values for both `x` and `y`")
    else:
        raise TypeError("Must pass values for either `x` or `y`")

    bar_plot_func = _StackBarPlotter if stack else _BarPlotter
    plotter = bar_plot_func(
        x,
        y,
        hue,
        data,
        order,
        hue_order,
        estimator,
        ci,
        n_boot,
        units,
        orient,
        color,
        palette,
        saturation,
        errcolor,
        errwidth,
        capsize,
        dodge,
    )

    plotter.value_label = "count"

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax


def freqplot(
    x=None,
    y=None,
    hue=None,
    data=None,
    order=None,
    hue_order=None,
    orient=None,
    color=None,
    palette=None,
    saturation=0.75,
    dodge=True,
    stack=False,
    ax=None,
    **kwargs,
):
    """ Show the frequency of observations in each categorical bin using bars.
 
    The frequency plot is a normalization of a histogram across categories, as opposed
    to quantitative variables. The basic API and options are identical to those for
    :func:`barplot`, so you can compare counts across nested variables.
 
    Parameters
    ----------
    x, y, hue : str or array-like, optional
        Inputs for plotting long-form data.
    data : DataFrame, array, or list of arrays, optional
        Dataset for plotting. If `x` and `y` are absent, this is interpreted as wide-form.
        Otherwise, data is expected to be long-form.
    order, hue_order : list of str, optional
        Order to plot the categorical levels, otherwise the levels are inferred from the
        data object.
    orient : {"v", "h"}, optional
        Whether to plot bars vertically ("v") or horizontally ("h"). This can also be
        inferred from the dtype of the input variables, but can be used to specify when the
        "categorical" variable is a numeric or when plotting wide-form data.
    color : matplotlib color, optional
        Color for all of the elemnts, or seed for a gradient palette.
    palette : palette name, list, or dict, optional
        Colors to use for the different levels of the `hue` variable. Should be somthing that
        can be interpreted by `color_palette()` or a dictionary mapping hue levels to
        matplotlib colors.
    saturation :  float, optional
        Proportion of the original saturation to draw colors. Large patches often look better
        with slighlty desaturated colors, but set this to `1` if you want the plot colorss to
        perfectly match the input color spec.
    dodge : bool, optional
        When hue nesting is used, whether elements should be shifted along the categorical axis.
    stack : bool, optional
        When hue nesting is used, whether elements should be stacked ontop of each other. Note,
        dodge is set to False when stack is True.
    ax : matplotlib.axes, optional
        Axes object to draw the plot onto, otherwise uses the current axes.
    **kwargs : Other keyword arguments are passed through to `plt.bar` at draw time
 
    Examples
    --------
    .. plot::
        :context: close-figs
 
        >>> import schmeaborn as sns
        >>> titanic = sns.load_dataset("titanic")
        >>> ax = sns.freqplot(x="class", data=titanic)
 
    Show frequencies for two categorical variables:
 
    .. plot::
        :context: close-figs
 
        >>> ax = sns.freqplot(x="class", hue="who", data=titanic)
 
    Plot the bars horizontally:
 
    .. plot::
        :context: close-figs
 
        >>> ax = sns.freqplot(y="class", hue="who", data=titanic)
 
    Plot categories stacked:
 
    .. plot::
        :context: close-figs
 
        >>> ax = sns.freqplot(x="class", hue="who", stack=True, data=titanic)
    """

    # Define parameters for barplot
    if stack:
        dodge = False
    estimator = len
    ci = None
    n_boot = 0
    units = None
    errcolor = None
    errwidth = None
    capsize = None

    # Check orientation by input
    if x is None and y is not None:
        orient = "h"
        x = y
    elif y is None and x is not None:
        orient = "v"
        y = x
    elif x is not None and y is not None:
        raise TypeError("Cannot pass values for both `x` and `y`")
    else:
        raise TypeError("Must pass values for either `x` or `y`")

    bar_plot_func = _StackBarPlotter if stack else _BarPlotter
    plotter = bar_plot_func(
        x,
        y,
        hue,
        data,
        order,
        hue_order,
        estimator,
        ci,
        n_boot,
        units,
        orient,
        color,
        palette,
        saturation,
        errcolor,
        errwidth,
        capsize,
        dodge,
    )

    # Safely calculate frequencies: NaN counts replaced by 0
    plotter.statistic = np.nan_to_num(plotter.statistic)

    if plotter.statistic.ndim == 1:
        # Normalize statistic
        plotter.statistic = plotter.statistic / np.nansum(plotter.statistic)

        # Safety Check for proper normalization
        err = f"Frequencies not properly normalized. \n {plotter.statistic} \n"
        assert np.allclose(np.nansum(plotter.statistic), 1, rtol=1e-6), err
    elif plotter.statistic.ndim > 1:
        # Normalize row-stochastic
        plotter.statistic = (
            plotter.statistic / np.nansum(plotter.statistic, axis=1)[:, None]
        )

        # Safely check for proper normalization (ignore where full row is null)
        sum_stats = np.nansum(plotter.statistic, axis=1)

        # Safety Check for proper normalization
        err = f"Frequencies not properly normalized. \n {plotter.statistic} \n"
        assert np.allclose(sum_stats, 1, rtol=1e-6), err
    else:
        raise ValueError("Unable to count the combination of x and hue.")

    plotter.value_label = "frequency"

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax
