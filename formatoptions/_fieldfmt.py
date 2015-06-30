# -*- coding: utf-8 -*-
import numpy as np
from _fmt_base import FmtBase, props
from _windfmt import WindFmt
from ..defaults import FieldFmt as default
from ..warning import warn, critical, warnings, Nc2MapWarning


class FieldFmt(FmtBase):
    """Class to control the formatoptions of a FieldPlot instance. See function
    show_fmtkeys for formatoption keywords.
    """

    # masking properties
    maskless = props.maskprop(
        'maskless', """
        Float (Default: %s). All data less than this value is masked (see
        also maskleq)""" %
        default['maskless'])
    maskleq = props.maskprop(
        'maskleq', """
        Float (Default: %s). All data less or equal than this value is masked
        (see also maskless)""" % default['maskleq'])
    maskgreater = props.maskprop(
        'maskgreater', """
        Float (Default: %s). All data greater than this value is masked (see
        also maskgeq)""" %
        default['maskgreater'])
    maskgeq = props.maskprop(
        'maskgeq', """
        Float (Default: %s). All data greater or equal than this value is
        masked (see also maskgreater)""" %
        default['maskgeq'])
    maskbetween = props.maskprop(
        'maskbetween', """
        Tuple or list (Default: %s). Pair (min, max) between which
        the data shall be masked""" % default['maskbetween'])
    plottype = props.default(
        'plottype', """
        string {'quad', 'tri'} (Default: %s). If 'tri', a triangular grid
        is assumed and the matplotlib.pyplot.tripcolor function is used for
        visualization. Otherwise the matplotlib.pyplot.pcolormesh function is
        used.""" % default['plottype'])
    grid = props.default(
        'grid', """
        color or linestyle (depending on `plottype`). Sets the color and
        linestyle for the grid (not the axes but the grid of the data) if you
        you want to show it.
        If `plottype` is 'quad', it can be a color, if `plottype` is 'tri',
        it can be a dictionary suitable for the plt.triplot function or a
        linestyle (e.g. 'k-' for black continuous lines).""")

    # wind plot property
    windplot = props.default(
        'windplot', """
        WindFmt instance. (Default initialized by {}). Defines the properties
        of the wind plot. Can be set either directly via a WindFmt instance or
        with a dictionary containing the formatoptions  (see
        show_fmtkeys('wind', 'windonly')""")

    def __init__(self, **kwargs):
        """initialization and setting of default values. Key word arguments may
        be any names of a property. Use show_fmtkeys for possible keywords and
        their documentation"""
        self.set_logger()
        super(FieldFmt, self).__init__()
        self._default.update(default)

        # set default values
        for key, val in default.items():
            setattr(self, key, val)

        # set WindFmt with options stored in 'windplot' and general options as
        # defined in kwargs
        self.windplot = WindFmt()
        self.update(**kwargs)

    def update(self, updatewind=True, **kwargs):
        """Update formatoptions property by the keywords defined in **kwargs.
        All key words of initialization are possible."""
        # set WindFmt with options stored in 'windplot' and general options as
        # defined in kwargs
        self._updating = 1
        kwargs = self._removeoldkeys(kwargs)
        if 'windplot' in kwargs:
            windops = kwargs.pop('windplot')
        else:
            windops = {}
        for key, val in kwargs.items():
            self.check_key(key)
            setattr(self, key, val)
        if updatewind and kwargs != {}:
            windops.update({key: kwargs[key] for key in self._general
                            if key in kwargs})
            windops.update({'lineshapes': self._lineshapes})
            windops.update({'lonlatbox': self.lonlatbox})
            calc_shapes = self.windplot._calc_shapes
            self.windplot._calc_shapes = 0
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", '|tight|', Nc2MapWarning,
                    'nc2map.formatoptions._base_fmt', 0)
                self.windplot.update(**windops)
            self.windplot._calc_shapes = calc_shapes
        self._updating = 0

    def asdict(self, wind=False):
        """Returns the non-default FmtBase instance properties as a
        dictionary"""
        fmt = {key[1:]: val for key, val in self.__dict__.items()
               if (key[1:] in self._default.keys()
                   and np.all(val != self._default[key[1:]])
                   and key[1:] != 'windplot')}
        if wind:
            fmt.update({'windplot': self.windplot.asdict(False)})
        return fmt
