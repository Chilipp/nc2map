# -*- coding: utf-8 -*-
"""_cbar_manager module of the nc2map module.

This module contains the definition of the CbarManager which aims at managing
the colorbars of multiple MapBase instances at once and the show_colormaps
function to visualize available colormaps, as well some predefined colormaps"""
from copy import deepcopy
from itertools import chain
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from formatoptions import FmtBase
from _maps_manager import MapsManager
from mapos._map_base import _props, returnbounds
from ._cmap_ops import get_cmap

class CbarManager(MapsManager):
    """Class to control a single colorbar of multiple MapBase instances.
    It is not recommended to use the methods of this class but rather
    use the update_cbar method from the Maps instance the CbarManager
    instance belongs to.
    See function __init__ for a description of the properties
    """

    ticks = _props.ticks(
        'ticks', """
        Ticks of the colorbars. Set it with integer, 'bounds', array or None,
        receive array of ticks.""")
    ticklabels = _props.ticklabels(
        'ticklabels', """
        Ticklabels of the colorbars. Set it with an array of strings.""")

    def __init__(self, maps, fig, cbar, fmt={}, mapsin=None, _bounds=None,
                 _cmap=None, _norm=None, wind=False, bottom_adjust=0.2,
                 right_adjust=0.8):
        """initialize the class.
        Input:
        - maps: List of MapBase instances which are controlled by the
            cbarmanager
        - fig: list of figures in which the MapBase instances are
        - cbar: dictionary. Keys are positions of the colorbar ('b',
            'r','sh' and 'sv') and values are again dictionaries with
            the figure of the colorbar instance as key and the colorbar
            instance as value.
        - fmt: a FmtBase instance to control the formatoptions
        - mapsin: the Maps instance the cbarmanager belongs to
        - _bounds are the bounds as calculated from the settings in fmt
        - _cmap is the colormap as defined from fmt
        - _norm is a normalization instance
        - wind is True, if the MapBase instances are WindPlot instances
            plotted over FieldPlot instances
        """
        self.set_logger()
        super(CbarManager, self).__init__()
        self.maps = maps
        self.cbar = cbar
        if isinstance(fmt, FmtBase):
            self.fmt = fmt
        else:
            fmt.setdefault('cmap', self.maps[0].fmt.cmap)
            fmt.setdefault('bounds', self.maps[0].fmt.bounds)
            fmt.setdefault('clabel', self.maps[0].fmt.clabel)
            self.fmt = FmtBase(**fmt)
        self.mapsin = mapsin
        self._bounds = _bounds
        self._cmap = _cmap
        self._norm = _norm
        self.wind = wind
        self.bottom_adjust = bottom_adjust
        self.right_adjust = right_adjust

    def update(self, fmt={}, todefault=False, plot=True, **kwargs):
        # delete colorbars
        self.logger.debug('Updating...')
        self.logger.debug('Plot after finishing: %s', plot)
        fmt = deepcopy(fmt)
        fmt.update(kwargs)
        possible_keys = self.fmt._default.keys()
        for key in fmt:
            self.fmt.check_key(key, possible_keys=possible_keys)
        if not todefault:
            self_fmt = self.fmt.asdict()
            fmt = {key: value for key, value in fmt.items()
                   if np.all(value != self_fmt.get(
                       key, self.fmt._default.get(key)))}
        else:
            self.logger.debug('Update to default...')
            fmt = {
                key: fmt.get(key, value)
                for key, value in self.fmt._default.items()
                if (key not in fmt and
                    np.all(value != getattr(self.fmt, key)))
                or (key in fmt and
                    np.all(fmt[key] != getattr(self.fmt, key)))}

        self.logger.debug("Update to ")
        try:
            max_kwarg_len = max(map(len, kwargs.keys())) + 3
        except ValueError:
            max_kwarg_len = 0
        for key, val in kwargs.items():
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              key, str(val))

        if 'plotcbar' in fmt:
            if fmt['plotcbar'] in [False, None]:
                fmt['plotcbar'] = ''
            if fmt['plotcbar'] is True:
                fmt['plotcbar'] = 'b'
            self._removecbar([cbarpos for cbarpos in self.fmt.plotcbar
                                if cbarpos not in fmt['plotcbar']])
        if ('extend' in fmt
                or fmt.get('ticklabels', False) is None):
            self.logger.debug(
                "Found 'extend' or 'ticklabels' in cbarops --> remove "
                "cbars")
            self._removecbar()
        self.fmt.update(**fmt)
        boundsnames = ['rounded', 'sym', 'minmax', 'roundedsym']
        if (self.fmt.bounds[0] in boundsnames
                and len(self.fmt.bounds) == 2):
            self._bounds = returnbounds(
                map(lambda x: (np.min(x), np.max(x)),
                    (mapo.data[:] for mapo in self.maps)),
                self.fmt.bounds)
        elif (self.fmt.bounds[0] in boundsnames
                and len(self.fmt.bounds) == 3):
            self._bounds = returnbounds(
                np.ma.concatenate(
                    tuple(mapo.data[:] for mapo in self.maps)),
                self.fmt.bounds)
        else:
            self._bounds = self.fmt.bounds
        if self.fmt.norm == 'bounds':
            self._cmap = get_cmap(self.fmt.cmap, len(self._bounds)-1)
            self._norm = mpl.colors.BoundaryNorm(self._bounds, self._cmap.N)
        else:
            self._cmap = get_cmap(self.fmt.cmap)
            if self.fmt.norm is None:
                self._norm = mpl.colors.Normalize()
                self._norm.vmin = self._bounds.min()
                self._norm.vmax = self._bounds.max()
            else:
                self._norm = self.fmt.norm
        if self.fmt.opacity is not None:
            self._calculate_opacity()
        for mapo in self.maps:
            mapo.fmt._enablebounds = False
            mapo._bounds = self._bounds
            mapo._cmap = self._cmap
            mapo._norm = self._norm
            mapo._make_plot = 1
            if plot:
                mapo.make_plot()
        if plot:
            self._draw_colorbar()

    def _draw_colorbar(self, draw=True):
        """draws the colorbars specified in self.fmt.plotcbar"""
        for cbarpos in self.fmt.plotcbar:
            if cbarpos in self.cbar:
                for fig in self.cbar[cbarpos]:
                    plt.figure(fig.number)
                    self.cbar[cbarpos][fig].set_cmap(self._cmap)
                    self.cbar[cbarpos][fig].set_norm(self._norm)
                    self.cbar[cbarpos][fig].draw_all()
                    if fig not in self.get_figs():
                        plt.draw()
            else:
                orientations = ['horizontal', 'vertical',
                                'horizontal', 'vertical']
                cbarlabels = ['b', 'r', 'sh', 'sv']
                if cbarpos not in cbarlabels:
                    try:
                        raise KeyError((
                            'Unknown position option %s! '
                            'Please use one of %s') % (
                                str(cbarpos), ', '.join(cbarlabels) + '.'))
                    except KeyError:
                        raise KeyError((
                            'Unknown position option for the colorbar! '
                            'Please use one of %s') % (
                                ', '.join(cbarlabels) + '.'))
                self.cbar[cbarpos] = {}
                orientation = orientations[cbarlabels.index(cbarpos)]
                if cbarpos in ['b', 'r']:
                    for fig in self.get_figs():
                        if cbarpos == 'b':
                            fig.subplots_adjust(bottom=self.bottom_adjust)
                            ax = fig.add_axes([0.125, 0.135, 0.775, 0.05])
                        elif cbarpos == 'r':
                            fig.subplots_adjust(right=self.right_adjust)
                            ax = fig.add_axes([0.825, 0.25, 0.035, 0.6])
                        self.cbar[cbarpos][fig] = mpl.colorbar.ColorbarBase(
                            ax, cmap=self._cmap, norm=self._norm,
                            orientation=orientation, extend=self.fmt.extend)
                elif cbarpos in ['sh', 'sv']:
                    if cbarpos == 'sh':
                        fig = plt.figure(figsize=(8, 1))
                        ax = fig.add_axes([0.05, 0.5, 0.9, 0.3])
                    elif cbarpos == 'sv':
                        fig = plt.figure(figsize=(1, 8))
                        ax = fig.add_axes([0.3, 0.05, 0.3, 0.9])
                    self._set_window_title(fig)
                    self.cbar[cbarpos][fig] = mpl.colorbar.ColorbarBase(
                        ax, cmap=self._cmap, norm=self._norm,
                        orientation=orientation, extend=self.fmt.extend)

            if self.fmt.clabel is not None:
                for fig in self.cbar[cbarpos]:
                    label = self._replace(txt=self.fmt.clabel, delimiter=', ')
                    self.cbar[cbarpos][fig].set_label(
                        label, **self.fmt._labelops)
            else:
                self.cbar[cbarpos][fig].set_label('')

        # set ticks
        self.ticks = self.fmt.ticks
        # set ticklabels
        self.ticklabels = self.fmt.ticklabels

        if draw:
            for fig in set(chain(*(
                    val.keys() for key, val in self.cbar.items()))):
                plt.figure(fig.number)
                plt.draw()

    def _calculate_opacity(self):
        opacity = np.array(self.fmt.opacity, dtype=float)
        if opacity.ndim == 0:
            if opacity < 0 or opacity > 1:
                raise ValueError(
                    "Float for opacity must be between 0 and 1, "
                    "not %s" % opacity)
            maxN = int(round(self._cmap.N*opacity.tolist()))
            alpha = np.linspace(0, 1., maxN)
            colors = self._cmap(range(self._cmap.N))
            colors[:maxN-self._cmap.N, -1] = alpha
            self._cmap = self._cmap.from_list(
                self._cmap.name + '_opa', colors, self._cmap.N)
        elif opacity.ndim == 1:
            if np.any(opacity > 1):
                raise ValueError(
                    "Opacities must be between 0 and 1! Not %s" % (
                        opacity.max()))
            if np.any(opacity < 0):
                raise ValueError(
                    "Opacities must be between 0 and 1! Not %s" % (
                        opacity.min()))
            x = range(self._cmap.N)
            xp = np.linspace(0, self._cmap.N, len(opacity),
                                endpoint=True)
            alpha = np.interp(x, xp, opacity)
            colors = self._cmap(range(self._cmap.N))
            colors[:, -1] = alpha
            self._cmap = self._cmap.from_list(
                self._cmap.name + '_opa', colors, self._cmap.N)
        elif opacity.ndim == 2:
            # assume second axis being tuple of (data, alpha)
            if not opacity.shape[-1] == 2:
                raise ValueError(
                    "Two-dimensional opacity settings must be "
                    "of shape (N, 2), not %s!" % str(
                        opacity.shape))
            if np.any(opacity[:, 1] > 1):
                raise ValueError(
                    "Opacities must be between 0 and 1! Not %s" % (
                        opacity.max()))
            if np.any(opacity[:, 1] < 0):
                raise ValueError(
                    "Opacities must be between 0 and 1! Not %s" % (
                        opacity.min()))
            cN = self._cmap.N
            bN = len(self._bounds) - 1
            if cN != bN:
                data = np.interp(
                    np.linspace(0, bN, cN, endpoint=True),
                    range(bN),
                    (self._bounds[:-1]+self._bounds[1:])*0.5)
            else:
                data = (self._bounds[:-1]+self._bounds[1:])*0.5
            alpha = np.interp(data, opacity[:, 0], opacity[:, 1],
                                left=0, right=1)
            colors = self._cmap(range(self._cmap.N))
            colors[:, -1] = alpha
            self._cmap = self._cmap.from_list(
                self._cmap.name + '_opa', colors, self._cmap.N)
        else:
            raise ValueError(
                "opacity formatoption keyword can at maximum be "
                "2-dimensional! Found %i dimensions!" % (
                    opacity.ndim))

    def asdict(self):
        return self.fmt.asdict()

    def _removecbar(self, positions='all'):
        """removes the colorbars with the defined positions (either 'b','r',
        'sh' or 'sv')"""
        if not hasattr(self, 'cbar'):
            return
        for mapo in self.maps:
            mapo.fmt._enablebounds = True
            mapo._make_plot = 1
            try:
                plt.sca(mapo.ax)
            except ValueError:  # map has no axis (probably because closed)
                continue
            mapo.update()
        if positions == 'all':
            positions = self.cbar.keys()
        for cbarpos in positions:
            if cbarpos in self.cbar:
                for fig in self.cbar[cbarpos]:
                    if cbarpos in ['b', 'r'] and cbarpos in self.cbar:
                        plt.figure(fig.number)
                        fig.delaxes(self.cbar[cbarpos][fig].ax)
                        if cbarpos == 'b':
                            fig.subplots_adjust(bottom=0.1)
                        else:
                            fig.subplots_adjust(right=0.9)
                        plt.draw()
                    elif cbarpos in ['sh', 'sv'] and cbarpos in self.cbar:
                        plt.close(fig)
                del self.cbar[cbarpos]
        return

    def _set_window_title(self, fig):
        """sets the canvas window title"""
        fig.canvas.set_window_title(
            'Figure %i: ' % fig.number + self._replace(
                'Colorbar of %(name)s: Variable %(var)s, time %(time)s, '
                'level %(level)s', delimiter=', '))

    def _moviedata(self, times, **kwargs):
        """Returns a generator for the movie with the formatoptions"""
        for i in xrange(len(times)):
            yield {key: next(value) for key, value in kwargs.items()}

    def _runmovie(self, fmt):
        """Run function for the movie suitable with generator _moviedata"""
        dims = self.meta.keys()
        self.fmt.update(**{key: value for key, value in fmt.items()
                           if key not in dims})
        self._draw_colorbar(draw=False)

    def get_cbars(self, positions='all'):
        """returns a list of all colorbars belonging to position

        Input:
          - position: Iterable containing 'b', 'r', 'sh', 'sv'

        Returns [list of colorbars specified by positions]
        """
        if positions == 'all':
            return list(chain(*(
                val.values() for key, val in self.cbar.items())))
        else:
            return list(chain(*(
                val.values() for key, val in self.cbar.items()
                if key in positions)))

    def _replace(self, txt, delimiter=None):
        """Replaces the text from self.meta and self.data.time"""
        for mapo in self.maps:
            if mapo.data.time is not None:
                # use strftime and decode with utf-8 (in case of accented
                # string)
                try:
                    txt = mapo.data.time.tolist().strftime(txt).decode('utf-8')
                    self.logger.debug("Using time information of %s", mapo)
                    break
                except AttributeError:
                    pass
        txt = super(CbarManager, self)._replace(txt, delimiter=delimiter,
                                                *self.maps)
        return txt

    def set_global_bounds(self, maps=None, time=slice(None)):
        """Calculates the colorbar bounds of the MapBase instances specified
        in maps by considering all times specified by time.
        Input:
          - maps: List of MapBase instances. If None, self.maps is used
          - time: Alternate time slice to use in the calculation. May be a
              slice or any iterable.
        """
        if not self.wind:
            if maps is None:
                maps = self.maps
            boundsnames = ['rounded', 'sym', 'minmax', 'roundedsym']
            if self.fmt.bounds[0] in boundsnames:
                boundsdata = map(
                    lambda x: (np.ma.min(x), np.ma.max(x)),
                    chain(
                        *((data[:] for data in mapo.gen_data(
                            time=time))
                            for mapo in maps)))
                boundsdata = np.ma.array(
                    boundsdata, mask=np.isnan(boundsdata))
                self.fmt.bounds = returnbounds(
                    boundsdata[:], self.fmt.bounds)

    def close(self, num=0):
        """Closes the CbarManager instance"""
        if not num % 2:
            self._removecbar()
        if not num % 3:
            for cbarpos in self.cbar:
                for fig in self.cbar[cbarpos]:
                    plt.close(fig)
