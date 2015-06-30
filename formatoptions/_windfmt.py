# -*- coding: utf-8 -*-
from difflib import get_close_matches
import numpy as np
from _fmt_base import FmtBase, props
from ..defaults import WindFmt as default
from ..warning import warn, critical, warnings

class WindFmt(FmtBase):
    """Class to control the formatoptions of a WindPlot instance. See function
    show_fmtkeys for formatoption keywords.
    """

    # ------------------ define properties here -----------------------
    # general
    proj = props.proj(
        'proj', FmtBase.proj.__doc__ + """
        NOTE: 'robin', 'northpole' and 'southpole' projections do not work for
        WindPlot and will be replaced by 'cyl'""")

    # Arrow properties
    arrowsize = props.streamplotops(
        'arrowsize', """
        float (Default: %s). Defines the size of the arrows""" %
        default['arrowsize'])
    arrowstyle = props.streamplotops(
        'arrowstyle', """
        string (Default: %s). Defines the style of the arrows (See
        :class:`~matplotlib.patches.FancyArrowPatch`)""" %
        default['arrowstyle'])
    linewidth = props.windplotops(
        'linewidth', """
        float, string ('absolute', 'u' or 'v') or 2D-array (Default: %s).
        Defines the linewidth behaviour. Possible types are
        - float: give the linewidth explicitly
        - 2D-array (which has to match the shape of of u and v): The values
            determine the linewidth according to the given numbers
        - 'absolute', 'u' or 'v': a normalized 2D-array is computed and makes
            the colorcode corresponding to the absolute flow of u or v. A
            further scaling can be done via the 'scale' key (see above). Higher
            'scale' corresponds to higher linewidth.""" % default['linewidth'])
    density = props.streamplotops(
        'density', """
        Float or tuple (Default: %s). Value scaling the density of the arrows
        (1 means no density scaling)
          - If float, this is the value for longitudes and latitudes.
          - If tuple (x, y), x scales the longitudes and y the latitude.
        Please note that for quiver plots (i.e. streamplot=False)
        densities > 1 are not possible. Densities of quiver plots are scaled
        using the weighted mean. Densities of streamplots are scaled using
        the density keyword of the pyplot.streamplot function.
        See also reduceabove for quiver plots.""" % default['density'])
    scale = props.quiverops(
        'scale', """
        Float (Default: %s). Scales the length of the arrows.
        Affects only quiver plots (i.e. streamplot=False).""" % (
            default['scale']))
    lengthscale = props.default(
        'lengthscale', """
        String (Default: %s). If 'log' the length of the quiver plot arrows
        are scaled logarithmically via speed=sqrt(log(u)^2+log(v)^2).
        This affects only quiver plots (i.e. streamplot=False).""" % (
            default['lengthscale']))

    color = props.windplotops(
        'color', """
        string ('absolute', 'u' or 'v'), matplotlib color code or 2D-array
        (Default: %s). Defines the color behaviour.
        Possible types are
        - 2D-array (which has to match the shape of of u and v): The values
            determine the colorcoding according to 'cmap'
        - 'absolute', 'u' or 'v': a color coding 2D-array is computed and make
            the colorcode corresponding to the absolute flow or u or v.
        - single letter ('b': blue, 'g': green, 'r': red, 'c': cyan, 'm':
            magenta, 'y': yellow, 'k': black, 'w': white): Color for all arrows
        - float between 0 and 1 (defines the greyness): Color for all arrows
        - html hex string (e.g. '#eeefff'): Color for all arrows""" %
        default['color'])
    cmap = props.windplotops('cmap', FmtBase.cmap.__doc__)
    rasterized = props.windplotops('rasterized', FmtBase.rasterized.__doc__)

    # masking properties
    reduceabove = props.default(
        'reduceabove', """
        Tuple or list (perc, pctl) with floats. (Default: %s). Reduces the
        resolution to 'perc' of the original resolution if in the area defined
        by 'perc' average speed is higher than the pctl-th percentile.
        'perc' can be a float 0<=f<=1 or a tuple (x, y) in this range. If
        float, this is the value for longitudes and latitudes. If tuple (x, y),
        x scales the longitudes and y the latitude. This defines the scaling of
        the density (see also density keyword).
        pctl can be a float between 0 and 100.
        This formatoption is for quiver plots (i.e. streamplot=False) only. To
        reduce the resolution of streamplots, use density keyword.""" % (
            default['reduceabove']))

    # style and additional labeling properties
    streamplot = props.streamplot(
        'streamplot', """
        Boolean (Default: %s). If True, a pyplot.streamplot() will be
        used instead of a pyplot.quiver()""" % default['streamplot'])
    # legend = props.default(  # currently not implemented
        # 'legend', """
        # Float or list of floats (Default: %s). Draws quiverkeys over the
        # plot""" % default['legend'])

    @property
    def windonly_keys(self):
        """Returns the formatoption keys that are specific for WindFmt
        instances but not FieldFmt"""
        return list(frozenset(self._default) - set(
            self._general + self._cmapprops))

    # initialization
    def __init__(self, **kwargs):
        """initialization and setting of default values. Key word arguments may
        be any names of a property. Use show_fmtkeys for possible keywords and
        their documentation"""
        self.set_logger()
        super(WindFmt, self).__init__()
        self._default.update(default)
        # Option dictionaries
        self._windplotops = {}

        # set default values
        for key, val in default.items():
            setattr(self, key, val)

        # add cmap to _cmapprops (not done anymore since cmap is windplotops
        self._cmapprops.append('cmap')

        for key, val in kwargs.items():
            self.check_key(key)
            setattr(self, key, val)

    def asdict(self, general=True):
        """Returns the non-default FmtBase instance properties as a
        dictionary.
        general may be True or False. If False, formatoption keywords from
        MapBase Base class are not returned"""
        if general:
            fmt = {key[1:]: val for key, val in self.__dict__.items()
                   if (key[1:] in self._default.keys()
                       and np.all(val != self._default[key[1:]]))}
        else:
            fmt = {key[1:]: val for key, val in self.__dict__.items()
                   if (key[1:] in self._default.keys()
                       and np.all(val != self._default[key[1:]])
                       and key[1:] not in self._general)}
        return fmt

    def get_fmtkeys(self, *args):
        """Function which returns a dictionary containing all possible
        formatoption settings as keys and their documentation as value.
        Use as args 'windonly' if only those keywords specific to the
        WindFmt object shall be returned."""
        # filter 'wind' out
        args = [arg for arg in args if arg != 'wind']
        if args == []:
            return sorted(self._default.keys())
        elif args == ['windonly']:
            return self.windonly_keys
        elif 'windonly' in args:
            args.remove('windonly')
            for arg in (arg for arg in args
                        if arg not in self._default.keys()):
                self.check_key(arg, raise_error=False)
                args.remove(arg)
            return [arg for arg in args if arg in self.windonly_keys]
        else:
            return super(WindFmt, self).get_fmtkeys(*args)

    def show_fmtkeys(self, *args):
        super(WindFmt, self).show_fmtkeys(*args)

    def _removeoldkeys(self, entries):
        return entries

    def _removeoldkeys(self, entries):
        """Method to remove wrong keys and modify them"""
        entries = super(WindFmt, self)._removeoldkeys(entries)
        if (not isinstance(entries.get('proj'), dict) and
                entries.get('proj') in ['northpole', 'southpole',
                                        'robin']):
            warn(
                "Stereographic and Robinson projections are known to "
                "produce errors with wind plots. --> I replace proj keyword "
                " argument with 'cyl' (cylindrical projection)")
            entries['proj'] = 'cyl'
        return entries


    # ------------------ modify docstrings here --------------------------
    show_fmtkeys.__doc__ = FmtBase.show_fmtkeys.__doc__ + """
        Use as 'windonly' in args if only those keywords specific to the
        WindFmt instance shall be shown."""
