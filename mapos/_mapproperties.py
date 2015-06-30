# -*- coding: utf-8 -*-
"""Module containing commonly used property definitions in the nc2map module

This module contains the MapProperties class, a container whose methods defines
properties that are commonly used throughout the classes in the
nc2map.mapos module."""
import glob
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ..warning import warn, critical
from .._axes_wrapper import subplots as make_subplots


class MapProperties(object):
    """class containg property definitions of class MapBase, and subclasses"""
    def default(self, x, doc):
        """default property"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)

        def delx(self):
            delattr(self, '_'+x)

        return property(getx, setx, delx, doc)

    def dim(self, x, doc):
        """Dimension property which directs to self.dims dictionary"""
        def getx(self):
            value = self.dims.get(x, 0)
            try:
                if value < 0:
                    var = self.get_var()
                    ntime = list(var.dimensions).index(self.reader.timenames)
                    value += ntime
            except TypeError:
                pass
            return value

        def setx(self, value):
            try:
                if value < 0:
                    var = self.get_var()
                    ntime = list(var.dimensions).index(self.reader.timenames)
                    value += ntime
            except TypeError:
                pass
            self.update(**{x: value})
            if self._mapsin is not None:
                self._mapsin._set_window_title(self.ax.get_figure())
            plt.sca(self.ax)
            plt.draw()

        def delx(self):
            del self.dims[x]

        return property(getx, setx, delx, doc)

    def speed(self, x, doc):
        def getx(self):
            if hasattr(self, '_'+x):
                return getattr(self, '_'+x)

            return self.calc_speed()

        def setx(self, value):
            setattr(self, '_'+x, value)

        def delx(self):
            delattr(self, '_'+x)

        return property(getx, setx, delx, doc)

    def weights(self, x, doc):
        """property to calculate weights"""
        def getx(self):
            self.logger.debug('Getting weights...')
            if hasattr(self, '_'+x):
                self.logger.debug('    Found attribute...')
                return getattr(self, '_'+x)
            return self.gridweights()

        def setx(self, value):
            setattr(self, '_'+x, value)

        def delx(self):
            delattr(self, '_'+x)

        return property(getx, setx, delx, doc)

    def ticks(self, x, doc):
        """Property calculating ticks and setting the ticks to the cbars in
        get_cbars method"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            self.logger.debug('Set ticks...')
            if isinstance(value, int):
                self.logger.debug('    Found integer %i', value)
                # only BoundaryNorm is used --> locator is FixedLocator
                # see method hmatplotlib.ticker.ColorbarBase._ticker
                locator = mpl.ticker.FixedLocator(self._bounds, nbins=10)
                ticks = locator()[::value]
            elif np.all(value == 'bounds'):
                self.logger.debug('    Set ticks equal to bounds')
                ticks = self._bounds
            elif np.all(value == 'mid'):
                self.logger.debug('    Set ticks into the middle of colorbars')
                bounds = np.array(self._bounds)
                ticks = (bounds[1:] + bounds[:-1])/2.
            else:
                self.logger.debug("    Set ticks to whatever got as value")
                ticks = value
            self.logger.debug("    --> Set ticks to %s", str(ticks))
            setattr(self, '_'+x, ticks)
            for cbar in self.get_cbars():
                cbar.set_ticks(ticks)

        def delx(self):
            delattr(self, '_'+x)

        return property(getx, setx, delx, doc)

    def ticklabels(self, x, doc):
        """Property setting up ticklabels to the cbars in get_cbars method"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            self.logger.debug('Set ticklabels...')
            if value is not None:
                if self.ticks is None:
                    self.logger.debug('    Ticks are None --> setting ticks')
                    locator = mpl.ticker.FixedLocator(self._bounds, nbins=10)
                    self.ticks = locator()
                self.logger.debug("    Set ticklables %s" % str(value))
                if len(self.ticks) != len(value):
                    warn("Length of ticks (%i) and ticklabels (%i)"
                         "do not match!" % (len(self.ticks), len(value)),
                         logger=self.logger)
            for cbar in self.get_cbars():
                if value is not None:
                    cbar.set_ticklabels(value)
                cbar.ax.tick_params(labelsize=self.fmt._ctickops['fontsize'])
                if cbar.orientation == 'horizontal':
                    axis = 'xaxis'
                else:
                    axis = 'yaxis'
                for text in getattr(cbar.ax, axis).get_ticklabels():
                    text.set_weight(self.fmt._ctickops['fontweight'])

        def delx(self):
            delattr(self, '_'+x)

        return property(getx, setx, delx, doc)

    def ax(self, x, doc):
        """default property"""
        def getx(self):
            if getattr(self, '_'+x) is None:
                fig, ax = make_subplots()
                try:
                    fig.canvas.set_window_title(
                        'Figure %i: %s' % (fig.number, self.name))
                except AttributeError:
                    pass
                setattr(self, '_' + x, ax)
                try:
                    setattr(self.wind, x, ax)
                except AttributeError:
                    pass
            ax = getattr(self, '_'+x)
            try:
                return ax._AxesWrapper__ax
            except AttributeError:
                return ax

        def setx(self, value):
            setattr(self, '_' + x, value)
            try:
                setattr(self.wind, x, value)
            except AttributeError:
                pass

        def delx(self):
            delattr(self, '_'+x)

        return property(getx, setx, delx, doc)
