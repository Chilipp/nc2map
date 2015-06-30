# -*- coding: utf-8 -*-
"""Module containing the SimplePlot and ViolinPlot class.

The SimplePlot class is the basis for all simple visualizations (i.e.
everything that is not controlled by a mpl_toolkits.basemap.Basemap
instance, i.e. everything that is not plotted on the Earth.
The ViolinPlot is used by the ViolinEvaluator to make violin plots"""
import six
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import DateFormatter, AutoDateFormatter
import datetime as dt
import numpy as np
from copy import deepcopy
from itertools import izip
from difflib import get_close_matches
from _base_plot import BasePlot
from ..warning import warn, critical
from ..formatoptions._simple_fmt import SimpleFmt


def ticks(x, axis, ticks_type, doc):
    """Property calculating ticks and setting the ticks for SimplePlot
    instances.
    Axis is the axis ('x' or 'y'), tick_type is 'major' or 'minor'"""
    def getx(self):
        return getattr(self, '_'+x)

    def setx(self, value):
        axiso = getattr(self.ax, axis + 'axis')
        tt = ticks_type
        minor = True if tt == 'minor' else False
        self.logger.debug('Set %s ticks of %s axis...', tt, axis)
        if value is None:
            if minor and axis == 'x':
                self.logger.debug("    Disable minor %s ticks" % axis)
                self.ax.tick_params(axis=axis, which=tt, bottom=False,
                                    top=False)
            elif minor and axis == 'y':
                self.logger.debug("    Disable minor %s ticks" % axis)
                self.ax.tick_params(axis=axis, which=tt, right=False,
                                    left=False)
            self.logger.debug(
                '    Set to default value with %s',
                self._axis_locators[axis][tt])
            getattr(axiso, 'set_%s_locator' % tt)(
                self._axis_locators[axis][tt])
            setattr(self, '_'+x, value)
            return
        if isinstance(value, int):
            self.logger.debug('    Found integer %i', value)
            if not minor:
                locator = self._axis_locators[axis][tt]
                self.logger.debug(
                    '    --> reset to default locator %s', locator)
                getattr(axiso, 'set_%s_locator' % tt)(locator)
            ticks = getattr(self.ax, 'get_%sticks' % axis)(
                minor=minor)[::value]
        else:
            self.logger.debug("    Set ticks to whatever got as value")
            try:
                ticks = value[:]
            except TypeError:
                ticks = None
        if ticks is not None:
            setattr(self, '_'+x, ticks)
        self.logger.debug("    --> Set ticks to %s", str(ticks))
        getattr(self.ax, 'set_%sticks' % axis)(ticks, minor=minor)
        return

    def delx(self):
        delattr(self, '_'+x)

    return property(getx, setx, delx, doc)


def ticklabels(x, axis, tick_type, doc):
    """Property setting up ticklabels to the cbars in get_cbars method
    Axis is the axis ('x' or 'y'), tick_type is 'major' or 'minor'"""
    def getx(self):
        return getattr(self, '_'+x)

    def setx(self, value):
        axiso = getattr(self.ax, axis + 'axis')
        ticks = getattr(self.ax, 'get_%sticks' % axis)()
        tt = tick_type
        minor = True if tt == 'minor' else False
        self.logger.debug('Set %s ticklabels of %s axis...', tt, axis)
        if isinstance(value, six.string_types):
            # assume format string or dictionary with major and minor
            default_formatter = self._axis_formatters[axis]
            if isinstance(default_formatter, AutoDateFormatter):
                getattr(axiso, 'set_%s_formatter' % tt)(
                    DateFormatter(value))
            else:
                getattr(axiso, 'set_%s_formatter' % tt)(
                    FormatStrFormatter(value))
            setattr(self, '_'+x, value)
        elif value is not None:
            if getattr(self, axis + 'ticks') is None:
                self.logger.debug('    Ticks are None --> setting ticks')
                axiso.set_ticks(ticks, minor=minor)
            self.logger.debug("    Set ticklables %s" % str(value))
            if len(ticks) != len(value):
                warn("Length of ticks (%i) and ticklabels (%i)"
                     "do not match!" % (len(ticks), len(value)),
                     logger=self.logger)
            axiso.set_ticklabels(value, minor=minor)
        else:
            self.logger.debug("    Set with default formatter %s",
                self._axis_formatters[axis])
            getattr(axiso, 'set_%s_formatter' % tt)(
                self._axis_formatters[axis])
        setattr(self, '_'+x, value)
        try:
            fontsize = self.fmt._tickops['fontsize'][tt]
        except TypeError:
            fontsize = self.fmt._tickops['fontsize']
        try:
            fontweight = self.fmt._tickops['fontweight'][tt]
        except TypeError:
            fontweight = self.fmt._tickops['fontweight']
        try:
            rotation = getattr(self.fmt, axis + 'rotation')[tt]
        except TypeError:
            rotation = getattr(self.fmt, axis + 'rotation')
        if not minor and getattr(self, x + '_minor') is not None:
            try:
                pad = self.fmt._tickops['pad']
                self.ax.tick_params(axis=axis, which='major', pad=pad)
            except KeyError:
                pass
        for text in axiso.get_ticklabels(minor=minor):
            text.set_size(fontsize)
            text.set_weight(fontweight)
            text.set_rotation(rotation)
        return

    def delx(self):
        delattr(self, '_'+x)

    return property(getx, setx, delx, doc)


class SimplePlot(BasePlot):
    """Basic class for all simple visualizations (i.e. everything that is not
    controlled by a mpl_toolkits.basemap.Basemap instance, i.e. everything
    that is not plotted on the Earth (e.g. a line plot).
    This class itself does not contain any plotting features."""
    xticks = ticks(
        'xticks', 'x', 'major', """
        Ticks of the x-axis. Set it with integer, 'bounds', array or None,
        receive array of ticks.""")
    xticklabels = ticklabels(
        'xticklabels', 'x', 'major', """
        Ticklabels of the x-axis. Set it with an array of strings.""")
    xticks_minor = ticks(
        'xticks_minor', 'x', 'minor', """
        Minor ticks of the x-axis. Set it with integer, 'bounds', array or
        None, receive array of ticks.""")
    xticklabels_minor = ticklabels(
        'xticklabels_minor', 'x', 'minor', """
        Minor ticklabels of the x-axis. Set it with an array of strings.""")
    yticks = ticks(
        'yticks', 'y', 'major', """
        Ticks of the y-axis. Set it with integer, 'bounds', array or None,
        receive array of ticks.""")
    yticklabels = ticklabels(
        'yticklabels', 'y', 'major', """
        Ticklabels of the y-axis. Set it with an array of strings.""")
    yticks_minor = ticks(
        'yticks_minor', 'y', 'minor', """
        Ticks of the y-axis. Set it with integer, 'bounds', array or None,
        receive array of ticks.""")
    yticklabels_minor = ticklabels(
        'yticklabels_minor', 'y', 'minor', """
        Ticklabels of the y-axis. Set it with an array of strings.""")

    def __init__(self, fmt={}, name='line', ax=None, mapsin=None,
                 figsize=None, meta={}):
        super(SimplePlot, self).__init__(name=name, mapsin=mapsin, ax=ax,
                                         figsize=figsize, meta=meta)
        if isinstance(fmt, SimpleFmt):
            self.fmt = fmt
        else:
            self.fmt = SimpleFmt(**fmt)
        self._axis_locators = {}
        self._axis_formatters = {}

    def _configureaxes(self):
        super(SimplePlot, self)._configureaxes()
        ax = self.ax
        self._set_default_formatters()
        self._set_default_locators()
        # configure scaling of x-axis
        if self.fmt.scale is not None and self.fmt.scale.lower() in ['logx',
                                                                     'logxy']:
            ax.set_xscale("log")
        else:
            ax.set_xscale('linear')
        # configure scaling of y-axis
        if self.fmt.scale is not None and self.fmt.scale.lower() in ['logy',
                                                                     'logxy']:
            ax.set_yscale("log")
        else:
            ax.set_yscale('linear')
        plt.figure(self.ax.get_figure().number)
        # draw the figure to make sure that the axis are formatted correctly
        plt.draw()
        # modify xticks
        plt.minorticks_on()
        self.xticks_minor = self.fmt.xticks['minor']
        self.xticks = self.fmt.xticks['major']
        self.xticklabels_minor = self.fmt.xticklabels['minor']
        self.xticklabels = self.fmt.xticklabels['major']
        self.yticks_minor = self.fmt.yticks['minor']
        self.yticks = self.fmt.yticks['major']
        self.yticklabels_minor = self.fmt.yticklabels['minor']
        self.yticklabels = self.fmt.yticklabels['major']

        if self.fmt.xlabel is not None:
            self.ax.set_xlabel(self._replace(self.fmt.xlabel),
                               **self.fmt._labelops)
        else:
            self.ax.set_xlabel('')
        if self.fmt.ylabel is not None:
            self.ax.set_ylabel(self._replace(self.fmt.ylabel),
                               **self.fmt._labelops)
        else:
            self.ax.set_ylabel('')

    def make_plot(self):
        # return if not enabled
        if not self.fmt.enable:
            return
        ax = self.ax
        plt.sca(self.ax)

        self._configureaxes()
        plt.draw()

    def update(self, **kwargs):
        self.ax.clear()

        self.make_plot()

    def _set_default_formatters(self, force=False):
        if not force:
            self._axis_formatters.setdefault(
                'x', self.ax.xaxis.get_major_formatter())
            self._axis_formatters.setdefault(
                'y', self.ax.yaxis.get_major_formatter())
        else:
            self._axis_formatters = {
                'x': self.ax.xaxis.get_major_formatter(),
                'y': self.ax.yaxis.get_major_formatter(),
                }

    def _set_default_locators(self, force=False):
        if not force:
            self._axis_locators.setdefault('x', {})
            self._axis_locators['x'].setdefault(
                'major', self.ax.xaxis.get_major_locator())
            self._axis_locators['x'].setdefault(
                'minor', self.ax.xaxis.get_minor_locator())
            self._axis_locators.setdefault('y', {})
            self._axis_locators['y'].setdefault(
                'major', self.ax.yaxis.get_major_locator())
            self._axis_locators['y'].setdefault(
                'minor', self.ax.yaxis.get_minor_locator())
        else:
            self._axis_locators['x'] = {
                'major': self.ax.xaxis.get_major_locator(),
                'minor': self.ax.xaxis.get_minor_locator()}
            self._axis_locators['y'] = {
                'major': self.ax.yaxis.get_major_locator(),
                'minor': self.ax.yaxis.get_minor_locator()}
    def close(self, num=0):
        """Closes the SimplePlot instance (*args do not have any effect)"""
        if not num % 2:
            self.ax.clear()
        if not num % 3:
            plt.close(self.ax.get_figure())
        if not num % 5:
            try:
                del self.data
            except AttributeError:
                pass
        if not num % 7:
            try:
                self.reader.close()
            except AttributeError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
    

class ViolinPlot(SimplePlot):
    """Class to make a violinplot"""
    @property
    def meta(self):
        meta = self._meta.copy()
        meta['name'] = self.name
        return meta
    def __init__(self, plotdata, fmt={}, name='violin', ax=None,
                 mapsin=None, snskwargs={}, meta={}):
        """Initialization method of the ViolinPlot class

        Input:
            - plotdata: must be the data suitable for the sns.violinplot
                function
            - fmt: dictionary suitable for the nc2map.formatoptions.SimpleFmt
                class.
            - name: Name of the ViolinPlot instance
            - ax: axes instance where to plot on (if None, a new figure will
                be created)
            - snskwargs: Further keywords that are passed to the sns.violinplot
                function
            - meta: Meta information of the ViolinPlot
        """
        import pandas as pd
        self.name = name
        self.set_logger()
        super(ViolinPlot, self).__init__(name=name, mapsin=mapsin, ax=ax,
                                         fmt=fmt, meta=meta)
        self.plotdata = pd.DataFrame(plotdata)
        self.snskwargs = snskwargs.copy()
        self.make_plot()

    def make_plot(self):
        """Make the violin plot with self.plotdata and self.snskwargs and
        configure the subplot according to self.fmt"""
        import seaborn as sns
        plt.sca(self.ax)
        sns.violinplot(self.plotdata, **self.snskwargs)
        self._configureaxes()
        plt.draw()

    def update(self, fmt={}, snskwargs={}, **kwargs):
        """Update the ViolinPlot.
        Input:
            - fmt: dictionary suitable for the nc2map.formatoptions.SimpleFmt
                class.
            - snskwargs: Further keywords that are passed to the sns.violinplot
                function
        Further keyword arguments (**kwargs) are also seen as formatoption
        keywords or snskwargs."""
        fmt = deepcopy(fmt)
        snskwargs = deepcopy(snskwargs)
        self.ax.clear()
        for key, val in kwargs.items():
            if key in self.fmt._default:
                self.fmt.update(**{key: val})
            else:
                self.snskwargs.update({key: val})
        self.make_plot()

    def __str__(self):
        return "nc2map.mapos.%s %s" % (self.__class__.__name__, self.name)
