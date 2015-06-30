#!/usr/bin/env python

from copy import deepcopy
from _simple_plot import SimplePlot
import matplotlib.pyplot as plt


class ViolinPlot(SimplePlot):
    """Class to make a violinplot"""
    @property
    def meta(self):
        meta = self._meta.copy()
        meta['name'] = self.name
        return meta
    def __init__(self, plotdata, fmt={}, name='violin', ax=None,
                 mapsin=None, snskwargs={}, meta={}):
        self.name = name
        self.set_logger()
        super(ViolinPlot, self).__init__(name=name, mapsin=mapsin, ax=ax,
                                         fmt=fmt, meta=meta)
        self.plotdata = plotdata
        self.snskwargs = snskwargs.copy()
        self.make_plot()
    
    def make_plot(self):
        import seaborn as sns
        plt.sca(self.ax)
        sns.violinplot(self.plotdata, **self.snskwargs)
        self._configureaxes()
        plt.draw()
        
    def update(self, fmt={}, snskwargs={}, **kwargs):
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
        return repr(self)[1:-1]

    def __repr__(self):
        return "<nc2map.mapos.%s %s>" % (self.__class__.__name__, self.name)
