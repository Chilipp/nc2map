# -*- coding: utf-8 -*-
"""_cbar_manager module of the nc2map module.

This module contains the definition of the CbarManager which aims at managing
the colorbars of multiple MapBase instances at once and the show_colormaps
function to visualize available colormaps, as well some predefined colormaps"""
import logging
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from difflib import get_close_matches


cmapnames = {  # names of self defined colormaps (see get_cmap function below)
    'red_white_blue': [  # symmetric water fluxes
        (1, 0, 0), (1, 0.5, 0), (1, 1, 0), (1, 1., 1),
        (1, 1., 1), (0, 1, 1), (0, 0.5, 1), (0, 0, 1)],
    'blue_white_red': [  # symmetric temperature
        (0, 0, 1), (0, 0.5, 1), (0, 1, 1), (1, 1., 1),
        (1, 1., 1), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)],
    'white_blue_red': [  # temperature
        (1, 1., 1), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)],
    'white_red_blue': [  # water fluxes
        (1, 1., 1), (1, 0, 0), (1, 1, 0), (0, 1, 1), (0, 0, 1)],
    }
for key, val in cmapnames.items():
    cmapnames[key + '_r'] = val[::-1]

def get_cmap(name, lut=None):
    """Returns the specified colormap. If name is one of
    nc2map._cmap_ops.cmapnames, a new instance of
    mpl.colors.LinearSegmentedColormap will the given lut will be created.
    Otherwise if name is a colormap, it will be returned unchanged
    """
    if name in cmapnames:
        lut = lut or 10
        return mpl.colors.LinearSegmentedColormap.from_list(
            name=name, colors=cmapnames[name], N=lut)
    elif name in plt.cm.datad:
        return plt.get_cmap(name, lut)
    else:
        return name


def show_colormaps(*args, **kwargs):
    """Script to show standard colormaps from pyplot. Taken from
    http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    and adapted in November 2014.
    *args may be any names as strings of standard colorbars (e.g. 'jet',
    'Greens', etc.) or a colormap instance suitable with pyplot.
    Parameters
    ----------
    *args
        any name as strings of standard colorbars (e.g. 'jet', 'Greens', etc.)
        or a colormap instance suitable with pyplot
    N: int, optional
        Default: 11. The number of increments in the colormap.
    show: bool, optional
        Default: True. If True, show the created figure at the end with
        pyplot.show(block=False)"""
    # This example comes from the Cookbook on www.scipy.org.  According to the
    # history, Andrew Straw did the conversion from an old page, but it is
    # unclear who the original author is."""
    import logging
    import matplotlib.pyplot as plt
    import numpy as np
    from difflib import get_close_matches
    a = np.linspace(0, 1, 256).reshape(1, -1)
    a = np.vstack((a, a))
    # Get a list of the colormaps in matplotlib.  Ignore the ones that end with
    # '_r' because these are simply reversed versions of ones that don't end
    # with '_r'
    logger = logging.getLogger(__name__)
    for arg in (arg for arg in args
                if arg not in plt.cm.datad.keys() + cmapnames.keys()):
        if isinstance(arg, str):
            similarkeys = get_close_matches(
                arg, plt.cm.datad.keys()+cmapnames.keys())
        if similarkeys != []:
            logger.warning(
                "Colormap %s not found in standard colormaps.\n"
                "Similar colormaps are %s.", arg, ', '.join(similarkeys))
        else:
            logger.warning(
                "Colormap %s not found in standard colormaps.\n"
                "Run function without arguments to see all colormaps", arg)
    if args == ():
        maps = sorted(m for m in plt.cm.datad.keys()+cmapnames.keys()
                      if not m.endswith("_r"))
    else:
        maps = sorted(m for m in plt.cm.datad.keys()+cmapnames.keys()
                      if m in args) + [m for m in args
                                       if not isinstance(m, str)]
    nmaps = len(maps) + 1
    fig = plt.figure(figsize=(5, 10))
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
    N = kwargs.pop('N', 11)
    show = kwargs.pop('show', True)
    for i, m in enumerate(maps):
        ax = plt.subplot(nmaps, 1, i+1)
        plt.axis("off")
        plt.imshow(a, aspect='auto', cmap=get_cmap(m, N),
                   origin='lower')
        pos = list(ax.get_position().bounds)
        fig.text(pos[0] - 0.01, pos[1], m, fontsize=10,
                 horizontalalignment='right')
    fig.canvas.set_window_title("Figure %i: Predefined colormaps" % fig.number)
    if show:
        plt.show(block=False)
