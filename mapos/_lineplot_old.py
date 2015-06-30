# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from itertools import izip
from difflib import get_close_matches
from _xyplot import XyPlot
from _simple_plot import SimplePlot


class LinePlot(SimplePlot):
    """class to plot lineplots"""

    def __init__(self, ydata, xdata=None, fmt={}, name='line', ax=None,
                 mapsin=None, meta={}):
        """Initialization method of LinePlot class

        Input:
          - ydata: Dictionary or array. y-data for the plot. Can be one of the
              following types:
              -- array with shape (n, m) where n is the number of lines and m
                  the number of data points per line (Note: m has not to be
                  the same for all lines)
              -- dictionary {'label': [y1, y2, ...],
                             'label2': [...]}
                  where the keys (e.g. 'label') is the label for the legend
                  and the value the 1-dimensional data array for each line.
              -- dictionary {'label': {'data': [y1, y2, ...],
                                       'fill': fill values for error range,
                                       'fill...': value for fill_between
                                       any other key-value pair for plt.plot
                                       },

                            'label2': {...}, ...}
                  As an explanation:
                    --- Each subdictionary of ydata will be interpreted as one
                        line.
                    --- 'label' is the key which will then be used in the
                        legend (if not 'label' is set in the inner dictionary)
                    --- The 'data' key stands for the one dimensional array
                        for the plot.
                    --- The 'fill' value indicates the data used for the
                        error range.
                    --- keywords starting with 'fill' will be used for the
                        fill_between method (e.g. fillalpha=0.5 will result
                        in plt.fill_between(..., alpha = 0.5)
                    --- any other key-value pair (e.g. color='r') will be used
                        in the plotting call (i.e. plt.plot(..., color='r')
                        for this line
          - xdata: Dictionary, Array or None.
              -- If None, the xdata will be range(n) where n is the length of
                  the corresponding ydata.
              -- If one-dimensional array, this array will be used for all
                  ydata lines
              -- The rest is the same as for ydata
          - fmt: Dictionary with formatoption keys for XyFmt class (see below)
          - name: name of the line plot instance
          - ax: Axes where to plot on. If None, a new axes will be created
          - mapsin: nc2map.MapsManager instance the line plot belongs to
          - meta: Dictionary containing meta informations that can be used for
              texts (e.g. title, etc.)
        """
        super(LinePlot, self).__init__(name=name, mapsin=mapsin, ax=ax,
                                       fmt=fmt, meta=meta)
        xdata, ydata = self._setlines(xdata, ydata, fmt=fmt)
        self.xdata = xdata
        self.ydata = ydata
        self.lines = {}
        self.make_plot()

    def make_plot(self):
        """Method that makes the plot"""
        # return if not enabled
        if not self.fmt.enable:
            return
        ax = self.ax
        plt.sca(ax)
        color_cycle = ax._get_lines.color_cycle

        if self.fmt.grid is not None:
            plt.grid()

        if self.fmt.xlim is None:
            xlim = (min(min(val['data']) for key, val in self.xdata.items()),
                    max(max(val['data']) for key, val in self.xdata.items()))
        else:
            xlim = self.fmt.xlim
        plt.xlim(*xlim)
        if self.fmt.ylim is not None:
            plt.ylim(self.fmt.ylim)
        self.xlim = xlim

        # make plotting
        for line, data in self.ydata.items():
            plotdata = data.pop('data')
            # y fill data
            try:
                yfilldata = np.array(data.pop('fill'))
                yfill_kwargs = {}
                for key, val in data.items():
                    if key[:-4] == 'fill':
                        yfill_kwargs[key[4:]] = data.pop(key)
            except KeyError:
                yfilldata = None
            if yfilldata is not None and yfilldata.ndim == 1:
                yfilldata = np.array([data['data'] - yfilldata,
                                      data['data'] + yfilldata])

            # x fill data
            try:
                xfilldata = np.array(self.xdata[line].pop('fill'))
                xfill_kwargs = {}
                for key, val in data.items():
                    if key[:-4] == 'fill':
                        xfill_kwargs[key[4:]] = data.pop(key)
            except KeyError:
                xfilldata = None
            if xfilldata is not None and xfilldata.ndim == 1:
                xfilldata = np.array([self.xdata[line]['data'] - xfilldata,
                                      self.xdata[line]['data'] + xfilldata])
            data.setdefault('label', line)
            self.lines[line] = ax.plot(
                self.xdata[line]['data'], plotdata, **data)
            lcolor = self.lines[line][0].get_color()
            if yfilldata is not None:
                y_fill_kwargs.setdefault('color', lcolor)
                self.lines[line].append(ax.fill_between(
                    self.xdata[line]['data'], yfilldata[0], yfilldata[1],
                    **yfill_kwargs))
            if xfilldata is not None:
                x_fill_kwargs.setdefault('color', lcolor)
                self.lines[line].append(ax.fill_betweenx(
                    plotdata, xfilldata[0], xfilldata[1],
                    **xfill_kwargs))

        if self.fmt.legend is not None:
            self.legend = plt.legend(**self.fmt.legend)

        self._configureaxes()
        plt.draw()

    def update(self, ydata={}, xdata={}, lines=None, **kwargs):
        """Update method of LinePlot class

        Input:
          - ydata: Dictionary {'label': {'key': 'val', ...},
                               'key': 'val', ...}
              where 'label' may be one of the line labels and ('key', 'val')
              any value pair which is also possible in __init__ method
              If set in the outer dictionary (i.e. not in the inner 'label'
              dictionary) they are considered as default items for all lines
          - xdata: Dictionary (same structure as ydata)
          - lines: List of strings. The strings must correspond to the 'labels'
              of the lines as used in self.ydata.keys(). This defines the
              which lines to update. If None, all lines will be updated.
        Further keywords may be any formatoption keyword from the XyFmt class.
        Note: To add new lines, use the addline method instead.
        """
        self.ax.clear()
        if hasattr(self, 'legend'):
            self.legend.remove()
            del self.legend
        self.fmt.update(kwargs)
        self.fmt.update(**{key: val for key, val in ydata.items() if key in
                           self.fmt._default})
        if lines is None:
            lines = self.ydata.keys()
        for line in lines:
            self.ydata[line].update(
                {key: val for key, val in ydata.items() if key not in lines})
            self.ydata[line].update(ydata.get(line, {}))
            self.xdata[line].update(
                {key: val for key, val in xdata.items() if key not in lines})
            self.xdata[line].update(xdata.get(line, {}))
        self.make_plot()

    def _setlines(self, xdata, ydata, fmt=None):
        if not isinstance(ydata, dict):
            try:
                iter(ydata[0])
            except TypeError:
                ydata = [ydata]
            if not hasattr(self, 'ydata'):
                n = 0
            else:
                n = len(self.lines)
            ydata = OrderedDict([('line%i' % i, {'data': ydata[i-n]})
                     for i in xrange(n, len(ydata)+n)])
        else:
            for line, data in ydata.items():
                if not isinstance(data, dict):
                    ydata[line] = {'data': data}
        if xdata is None:
            xdata = {key: {'data': range(len(val['data']))}
                     for key, val in ydata.items()}
        elif not isinstance(xdata, dict):
            keys = sorted(ydata.keys())
            xdata = {key: {'data': xdata} for key in keys}
        else:
            for line, data in xdata.items():
                if not isinstance(data, dict):
                    xdata[line] = {'data': data}
        return xdata, ydata

    def addline(self, ydata, xdata=None):
        xdata, ydata = self._setlines(xdata, ydata)
        self.xdata.update(xdata)
        self.ydata.update(ydata)
        self.ax.clear()
        if hasattr(self, 'legend'):
            self.legend.remove()
            del self.legend
        self.make_plot()

    def show(self):
        plt.show(block=False)

    def close(self):
        plt.close(self.ax.get_figure())
        del self.xdata
        del self.ydata
