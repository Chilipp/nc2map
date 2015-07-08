# -*- coding: utf-8 -*-
"""Module containing the MapBase and Share class

This module defines the MapBase class which is used by FieldPlot and
WindPlot and mainly is responsible for the interface to the reader (i.e. for
data management), for colorbar management and for the
mpl_toolkits.bm.Basemap management.
Furthermore it contains the Share class which is used to spread extensive
formatoptions (like lsmask (i.e. ocean_color) and lineshapes)."""
import logging
import glob
from collections import OrderedDict
from itertools import chain
from .._axes_wrapper import wrap_subplot
from itertools import izip
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .._basemap import Basemap
from _mapproperties import MapProperties
from _base_plot import BasePlot
from ..warning import warn, critical
from ..defaults import texts
# container containing methods for property definition
_props = MapProperties()


def returnbounds(data, bounds):
    """Returns automatically generated bounds.
    Input:
    - data: Array. Data used for generation of the bounds
    - 'bounds' 1D-array or tuple (Default:('rounded', 11, True):
      Defines the bounds used for the colormap. Possible types are
      + 1D-array: Defines the bounds directly by giving the values
      + tuple (string, N): Compute the bounds automatically. N gives
        the number of increments whereas <string> can be one of the
        following strings
        ++ 'rounded': Rounds min and maxvalue of the data to the next
            lower (in case of minimum) or higher (in case of maximum)
            0.5-value with respect to the exponent with base 10 of the
            maximal range (i.e. if minimum = -1.2e-4, maximum = 1.8e-4,
            min will be -1.5e-4, max 2.0e-4) using round_to_05 function.
        ++ 'roundedsym': Same as 'rounded' but symmetric around zero
            using the maximum of the data maximum and (absolute value of)
            data minimum.
        ++ 'minmax': Uses minimum and maximum of the data (without roun-
            ding)
        ++ 'sym': Same as 'minmax' but symmetric around 0 (see 'rounded'
            and 'roundedsym').
      + tuple (string, N, percentile): Same as (string, N) but uses the
        percentiles defined in the 1D-list percentile as maximum. percen-
        tile must have length 2 with [minperc, maxperc]
      + string: same as tuple with N automatically set to 11
    """
    exp = np.floor(np.log10(abs(np.max(data) - np.min(data))))
    if isinstance(bounds, str):
        bounds = (bounds, 11)
    if isinstance(bounds[0], str):
        N = bounds[1]
        # take percentiles as boundary definitions
        if len(bounds) == 3:
            perc = deepcopy(bounds[2])
            if perc[0] == 0:
                perc[0] = np.ma.min(data)
            else:
                perc[0] = np.percentile(np.ma.compressed(data), perc[0])
            if perc[1] == 100:
                perc[1] = np.ma.max(data)
            else:
                perc[1] = np.percentile(np.ma.compressed(data), perc[1])
            if perc[1] == perc[0]:
                print((
                    'Attention!: Maximum and Minimum bounds are the same!\n'
                    'I use max value for maximal bound.'))
                perc[1] = np.max(data)
            data = deepcopy(np.ma.masked_outside(data, perc[0], perc[1],
                                                 copy=True))
        if bounds[0] == 'rounded':
            cmax = np.max((
                round_to_05(np.ma.max(data), exp, np.ceil),
                round_to_05(np.ma.max(data), exp, np.floor)))
            cmin = np.min((
                round_to_05(np.ma.min(data), exp, np.floor),
                round_to_05(np.ma.min(data), exp, np.ceil)))
        elif bounds[0] == 'minmax':
            cmax = np.ma.max(data)
            cmin = np.ma.min(data)
        elif bounds[0] == 'roundedsym':
            cmax = np.max((
                np.ma.max((round_to_05(np.ma.max(data), exp, np.ceil),
                           round_to_05(np.ma.max(data), exp, np.floor))),
                np.abs(np.min((round_to_05(np.ma.min(data), exp, np.floor),
                               round_to_05(np.ma.min(data), exp, np.ceil))))))
            cmin = -cmax
        elif bounds[0] == 'sym':
            cmax = np.max((np.ma.max(data), np.abs(np.ma.min(data))))
            cmin = -cmax
        bounds = np.linspace(cmin, cmax, N, endpoint=True)
    return bounds


def round_to_05(n, exp=None, func=round):
    """Applies the round function specified in func to round n to the
    next 0.5-value with respect to its exponent with base 10 (i.e.
    1.3e-4 will be rounded to 1.5e-4) if exp is None or with respect
    to the given exponent in exp.
    Input:
        - n: Float, number to round
        - exp: Integer. Exponent for rounding
        - func: Rounding function
    Output:
        - Rounded n
    """
    from math import log10, floor
    if exp is None:
        exp = floor(log10(abs(n)))  # exponent for base 10
    ntmp = np.abs(n)/10.**exp  # mantissa for base 10
    if np.abs(func(ntmp) - ntmp) >= 0.5:
        return np.sign(n)*(func(ntmp) - 0.5)*10.**exp
    else:
        return np.sign(n)*func(ntmp)*10.**exp


class Share(object):
    """Share object to share formatoptions with extensive data requirements of
    one MapBase instance with others.
    Such formatoptions are lsmask (i.e. ocean_color and land_color) and
    lineshapes.

    Sharing methods:
      - lsmask: Share the data for the land sea mask (ocean_color, land_color)
      - lineshapes: Share the polygon shapes drawn by lineshapes fmt keyword
      - lonlatbox: Shares the longitude-latitude boundary box with other
          MapBases

    Unsharing methods:
      - unshare_lineshapes: unshares the lineshapes
      - unshare_lsmask: unshares the land sea mask
      - unshare: unshares everything
    This methods enable the upper formatoptions again in the formally shared
    MapBase instances
    """
    def __init__(self, mapo):
        """Initialization function.

        Input:
          - mapo: A MapBase instance
        """
        self.mapo = mapo
        self.set_logger()
        self._draw = 1
        self.shared = {'lsmask': set(), 'lineshapes': {}}

    def lsmask(self, maps):
        """Shares the land sea mask with all MapBase instances given in maps"""
        def calc_lat_lons(grid=5, **kwargs):
            """Parts taken from mpl_toolkits.basemap._readlsmask,
            module version 1.0.7
            kwargs are ignored"""
            if grid == 10:
                nlons = 2160
            elif grid == 5:
                nlons = 4320
            elif grid == 2.5:
                nlons = 8640
            elif grid == 1.25:
                nlons = 17280
            else:
                raise ValueError(
                    'grid for land/sea mask must be 10,5,2.5 or 1.25')
            nlats = nlons/2
            delta = 360./nlons
            lsmask_lons = np.linspace(
                -180+0.5*delta,180-0.5*delta,nlons).astype(np.float32)
            lsmask_lats = np.linspace(
                -90+0.5*delta,90-0.5*delta,nlats).astype(np.float32)
            return lsmask_lons, lsmask_lats
        try:
            maps = set(maps)
        except TypeError:
            maps = set([maps])
        ocean_color = self.mapo.fmt.ocean_color
        land_color = self.mapo.fmt.land_color
        if land_color is None and ocean_color is None:
            pass
        elif ocean_color is not None:
            try:
                options = dict(ocean_color).copy()
            except ValueError:
                options = {'ocean_color': ocean_color}
            if any('lsmask_' + key not in options for key in ['lats', 'lons']):
                lsmask_lons, lsmask_lats = calc_lat_lons(**options)
                options.setdefault('lsmask_lons', lsmask_lons)
                options.setdefault('lsmask_lats', lsmask_lats)
            options['lsmask'] = self.mapo.mapproj.lsmask
            options.setdefault('land_color', land_color or 'w')
            mapproj = self.mapo.mapproj
            for mapo in maps:
                mapo._basemapops['lsmask'] = mapproj.drawlsmask(
                    ax=mapo.ax, **options)
                mapo._shared['lsmask'] = self
        elif land_color is not None:
            # they are not so big, so don't worry
            for mapo in maps:
                mapo._draw_lsmask(ocean_color=None, land_color=land_color)
                mapo._shared['lsmask'] = self
        self.shared['lsmask'].update({mapo for mapo in maps})
        self._draw_figures(*maps)

    def lineshapes(self, maps=[], *args, **kwargs):
        """Shares the specified shapes with the MapBase instances in maps

        Input:
          - maps: List of MapBase instances.
        Further arguments may be shape keywords (see
        mapo.fmt.lineshapes.keys()) that shall be shared. If no *args and
        **kwargs are given and, all lineshapes will be shared
        Further keyword arguments may also be shapes keys and the values
        must be lists of MapBase instances. If the maps keyword is not None
        (see above) they will also be used."""
        try:
            maps = set(maps)
        except TypeError:
            maps = set([maps])
        if not maps and not kwargs:
            raise ValueError(
                "Either maps must not be None or they must be specified in "
                "kwargs!")
        lineshapes = self.mapo.fmt.lineshapes
        shapes = (list(args) or lineshapes.keys()) + kwargs.keys()
        if not shapes:
            raise ValueError(
                "No lineshapes found in formatoptions of %s!" % mapo)
        for shape in set(shapes) - set(lineshapes):
            raise KeyError(
                "Shape %s not found in formatoptions! Possible keys are %s" % (
                    shape, ', '.join(lineshapes)))
        for shape in shapes:
            kwargs[shape] = set(kwargs.setdefault(shape, set()))
            kwargs[shape].update(maps)
            self.shared['lineshapes'].setdefault(shape, set())
            self.shared['lineshapes'][shape].update(kwargs[shape])
        all_maps = set(chain(*kwargs.values()))
        for mapo in all_maps:
            options = {shape: lineshapes[shape] for shape in shapes
                       if mapo in kwargs[shape]}
            mapo_shapes = mapo._basemapops.setdefault('lineshapes', {})
            for key, val in options.items():
                mapo_shapes[key] = mapo.mapproj.readshapefile(name=key, **val)
            mapo._shared.setdefault('lineshapes', {}).update({
                shape: self for shape in options})
        self._draw_figures(*all_maps)

    def lonlatbox(self, maps):
        """Shares the longitude-latitude box with all MapBase instances given
        in maps.
        This setting has to be unshared manually!"""
        try:
            maps = set(maps)
        except TypeError:
            maps = set([maps])
        for mapo in maps:
            mapo.update(lonlatbox=self.mapo.fmt.lonlatbox)
        self._draw_figures(*maps)

    def _draw_figures(self, *maps):
        if not self._draw:
            return
        for fig in {mapo.ax.get_figure() for mapo in maps}:
            plt.figure(fig.number)
            plt.draw()

    def _get_maps(self, maps, all_maps):
        """Trys whether an instance in maps is a mapo in all_maps"""
        try:
            maps = set(maps)
        except TypeError:
            maps = set([maps])
        for i, mapo in enumerate(maps):
            if not hasattr(mapo, 'name'):
                try:
                    maps.remove(mapo)
                    maps.add([
                        _mapo for _mapo in all_maps if _mapo.name == mapo][0])
                except IndexError:
                    raise ValueError(
                        "Could not find a MapBase instance with name %s" % (
                            mapo))
        return maps

    def _from_dict(self, s, maps):
        """Load the shares from a dictionary created with
        MapBase.asdict(shared=True)"""
        maps = set(maps)
        s = dict(s)
        # draw land sea masks
        lsmask_maps = [
            mapo for mapo in maps if mapo.name in s.setdefault('lsmask',
                                                                    [])]
        if len(lsmask_maps) < len(set(s['lsmask'])):
            missing = {
                name for name in s['lsmask'] if not any(
                    mapo.name == name for mapo in lsmask_maps)}
            warn("Attention! Maps %s are missing for shared land sea "
                    "mask!" % ', '.join(missing), logger=self.logger)
        if lsmask_maps:
            self.lsmask(lsmask_maps)
        # draw shape lines
        shapes = {shape: set() for shape in s.setdefault('lineshapes', {})}
        for shape, map_names in s['lineshapes'].items():
            shapes[shape] = {mapo for mapo in maps if mapo.name in map_names}
            if not len(shapes[shape]) == len(set(map_names)):
                missing = {
                    name for name in map_names if not any(
                        mapo.name == name for mapo in set(shapes[shape]))}
                warn("Attention! Maps %s are missing for shared "
                     "lineshapes!" % ', '.join(missing), logger=self.logger)
        if shapes:
            self.lineshapes(**shapes)
        return {'lsmask': lsmask_maps, 'lineshapes': shapes}

    def unshare(self, *args):
        """Removes the shares (land sea mask and lineshapes) with the
        specified MapBase instances. *args may be MapBase instances or names of
        MapBase instances"""
        all_maps = set(chain(*self.shared['lineshapes'].values()))
        all_maps.update(self.shared['lsmask'])
        args = self._get_maps(set(args) or all_maps, all_maps)
        self._draw = 0  # disable drawing (will be done at the end)
        self.unshare_lineshapes(args)
        self.unshare_lsmask(args)
        self._draw = 1  # enable drawing
        self._draw_figures(*args)

    def unshare_lineshapes(self, maps=[], *args):
        """Removes the shares (land sea mask and lineshapes) with the
        specified MapBase instances.
        Input:
            - maps: list of MapBase instances or names of MapBase instances
        *args may be shapes keys. If no shapes are specified, all are
        removed"""
        try:
            maps = set(maps)
        except TypeError:
            maps = set([maps])
        all_maps = set(chain(*self.shared['lineshapes'].values()))
        maps = self._get_maps(maps or all_maps, all_maps)
        shapes = list(args) or self.shared['lineshapes'].keys()
        for mapo in maps:
            for shape in shapes:
                map_list = self.shared['lineshapes'].get(shape)
                if not map_list:
                    continue
                if mapo in map_list:
                    map_list.remove(mapo)
                    if not map_list:
                        del self.shared['lineshapes'][shape]
                    try:
                        mapo._shared.get('lineshapes', {}).pop(shape)
                    except KeyError:
                        pass
                    mapo._update_mapproj({'lineshapes'})
        self._draw_figures(*maps)

    def unshare_lsmask(self, maps=[]):
        """Removes the shared land sea masks with the
        specified MapBase instances. Arguments (*args) have to be MapBase
        instances that have a shared land sea mask with this instance.
        If maps evaluates to False, shares are unset"""
        try:
            maps = set(maps)
        except TypeError:
            maps = set([maps])
        all_maps = self.shared['lsmask']
        maps = self._get_maps(maps or all_maps, all_maps)
        for mapo in maps:
            if mapo in all_maps:
                all_maps.remove(mapo)
                try:
                    del mapo._shared['lsmask']
                except KeyError:
                    pass
                mapo._draw_lsmask(ocean_color=mapo.fmt.ocean_color,
                                  land_color=mapo.fmt.land_color)
        self._draw_figures(*maps)

    def asdict(self):
        """Returns a dictionary with 'lsmask' and 'lineshapes' keys. Lists are
        sets of the names of the MapBase classes which what the settings are
        shared"""
        sdict = self.shared
        fmt = {}
        fmt['lsmask'] = {mapo.name for mapo in sdict['lsmask']}
        fmt['lineshapes'] = {}
        for shape, shared in sdict['lineshapes'].items():
            fmt['lineshapes'][shape] = {mapo.name for mapo in shared}
        return fmt

    def set_logger(self, name=None, force=False):
        """This function sets the logging.Logger instance for the Share
        instance.
        Input:
          - name: name of the Logger (if None: it will be named like
             <module name>.<class name>)
          - force: True/False (Default: False). If False, do not set it if the
              instance has already a logger attribute."""
        if name is None:
            try:
                name = '%s.%s.%s' % (self.__module__, self.__class__.__name__,
                                     self.mapo.name)
            except AttributeError:
                name = '%s.%s' % (self.__module__, self.__class__.__name__)
        if not hasattr(self, 'logger') or force:
            self.logger = logging.getLogger(name)
            self.logger.debug('Initializing...')

class MapBase(BasePlot):
    """Class controlling a the plot of a single variable, timestep and level of
    a netCDF file. Properties are below. It is not recommended to use one
    single MapBase instance but to use the nc2map.Maps class. Nevertheless: for
    initialization see __init__ method.
    Use the make_plot method after initialization to make the plot.
    """

    # ------------------ define properties here -----------------------
    # General properties
    name = _props.default('name', """String. Name of the MapBase instance""")
    data = _props.default(
        'data', """
        nc2map.readers.DataField instance containing the data of the plot""")
    time = _props.dim(
        'time', """
        Integer (Default: 0). timestep in nc-file""")
    level = _props.dim(
        'level', """
        Integer (Default: 0). Level in the nc-file""")
    reader = _props.default(
        'reader', """
        ReaderBase instance containing the data to plot""")
    ax = _props.ax('ax', """axes instance the MapBase instance plots on.""")
    weights = _props.weights(
        'weights', """
        numpy array. Grid cell weights as calculated via cdos""")
    ticks = _props.ticks(
        'ticks', """
        Ticks of the colorbars. Set it with integer, 'bounds', array or None,
        receive array of ticks.""")
    ticklabels = _props.ticklabels(
        'ticklabels', """
        Ticklabels of the colorbars. Set it with an array of strings.""")

    @property
    def meta(self):
        """Dictionary. Meta information as stored in the reader"""
        meta = self.reader.get_meta(var=self.var).copy()
        meta.update(self._meta)
        meta.update({'name': self.name})
        meta.update(self.dims)
        meta.setdefault('all', '%s %s of %s' % (
            self.__class__.__name__, self.name, ', '.join(
                map(lambda item: '%s: %s' % item, self.dims.items()))))
        return meta

    def __init__(self, reader, name='mapo', ax=None, fmt={}, mapsin=None,
                 meta={}, **dims):
        """
        Input:
        - reader: nc2map.readers.ArrayReader instance containing the data
        - name: string. Name of the MapBase instance
        - time: integer. Timestep which shall be plotted
        - level: integer. level which shall be plotted
        - ax:  matplotlib.axes.AxesSubplot instance matplotlib.axes.AxesSubplot
               where the data can be plotted on
        - mapsin: The Maps instance this MapBase instance belongs to.
        - timenames: 1D-array of strings: Gives the name of the time-dimension
                     for which will be searched in the netCDF file
        - levelnames: 1D-array of strings: Gives the name of the fourt
                      dimension (e.g vertical levels) for which will be
                      searched in the netCDF file
        - lon: 1D-array of strings: Gives the name of the longitude-
               for which will be searched in the netCDF file
        - lat: 1D-array of strings: Gives the name of the latitude-
               for which will be searched in the netCDF file
        """
        self.name = name
        self.set_logger()
        super(MapBase, self).__init__(name=name, mapsin=mapsin, ax=ax,
                                      meta=meta)
        # set reader
        self.share = Share(self)
        self.reader = reader
        self.dims = dims  # save dims (necessary for get_var method)
        dims = self.reader.expand_dims(**dims)
        self.dims_orig = dims.copy()
        self.dims = dims
        self._resize = True
        self._make_plot = 1
        self._reinitialize = 0
        self._draw_mapproj = 1
        self._shared = {}


    def get_data(self, var=None, **kwargs):
        """Reads the specified variable, longitude and lattitude from netCDF
        file and shifts it.
        Input:
            - var: Variable name to read for in the netCDF file
            - time: time index in netCDF file (if None, use self.time)
            - level: level index in netCDF file (if None, use self.level)
        """
        dims = self.dims.copy()
        dims.update(kwargs)
        if var is not None:
            dims['var'] = var
            try:
                dims.pop('vlst')
            except KeyError:
                pass
        data = self.reader.get_data(**dims)
        if self.fmt.latlon:
            data.shift_data(self.mapproj).mask_outside(self.mapproj)
            return self._mask_data(data)
        else:
            return data
    
    def _calculate_opacity(self):
        self.logger.debug("Calculating opacity...")
        opacity = np.array(self.fmt.opacity, dtype=float)
        if opacity.ndim == 0:
            self.logger.debug("    Found scalar %s", opacity)
            if opacity < 0 or opacity > 1:
                raise ValueError(
                    "Float for opacity must be between 0 and 1, "
                    "not %s" % opacity)
            maxN = int(round(self._cmap.N*opacity.tolist()))
            alpha = np.linspace(0, 1., maxN)
            colors = self._cmap(range(self._cmap.N))
            colors[:maxN, -1] = alpha
            self._cmap = self._cmap.from_list(
                self._cmap.name + '_opa', colors, self._cmap.N)
        elif opacity.ndim == 1:
            self.logger.debug("    Got 1D-array of length %s", len(opacity))
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
            self.logger.debug("    Got 2D-array of shape %s", opacity.shape)
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

    def gen_data(self, var=None, datashape='2d', **furtherdims):
        """Reads the specified variable, longitude and lattitude from netCDF
        file and shifts it.
        Input:
            - var: Variable name to read for in the netCDF file
            - time: time index in netCDF file (if None, use self.time)
            - level: level index in netCDF file (if None, use self.level)
        """
        dims = self.dims.copy()
        dims.update(furtherdims)
        if var is not None:
            dims['var'] = var
            try:
                dims.pop('vlst')
            except KeyError:
                pass
        data_gen = self.reader.gen_data(datashape=datashape, **dims)
        for data in data_gen:
            data.shift_data(self.mapproj).mask_outside(self.mapproj)
            if dims.get('var') is not None:
                yield self._mask_data(data)
            else:
                for i in xrange(len(dims['vlst'])):
                    data[i, :] = self._mask_data(data[i, :])
                yield data

    def _reinit(self, reader=None, ax=None, **kwargs):
        """Reinitialize dims, ax and reader"""
        zipped_kwargs = izip(['reader', 'ax'], [reader, ax])
        not_none_kwargs = (key for key, val in zipped_kwargs
                           if val is not None)
        self.logger.debug('Reinit dimensions %s', ', '.join(kwargs.keys()))
        for key in kwargs:  # by default set dimension to 0
            self.dims_orig.setdefault(key, 0)
        self.dims.update(**kwargs)
        self.logger.debug('Reinit %s', ', '.join(not_none_kwargs))
        if ax is not None:
            self.ax = ax
        plt.sca(self.ax)
        if reader is not None:
            self.reader = reader

    def _mask_data(self, data):
        """mask data from self.fmt.mask"""
        if self.fmt.mask is not None:
            self.logger.debug('Masking data...')
            data.mask_data(self.fmt.mask.shift_data(self.mapproj)[:])
        return data

    def _replace(self, txt, fig=False, delimiter='-'):
        """Replaces the text from self.meta and self.data.time

        Input:
          - txt: string where '%(key)s' will be replaced by the value of the
              'key' information in the MapBase.meta attribute (e.g.
              '%(long_name)s' will be replaced by the long_name of the
              variable file in the NetCDF file.
              Strings suitable for the datetime.datetime.strftime function
              (e.g. %Y, %B, %H, etc) will also be replaced using the
              using the data.time information.
              The string {tinfo} will furthermore be replaced by the time
              information in the format '%B %d, %Y. %H:%M', {dinfo} will be
              replaced by '%B %d, %Y'.
          - fig: True/False. If True and this MapBase instance belongs to a
              nc2map.MapsManager instance, the _replace method of this
              instance is used to get the meta information
          - delimiter: string. Delimter that shall be used for meta
              informations if fig is True.
        Returns:
          - string with inserted meta information
        """
        txt = txt.format(**texts['labels'])
        if self.data.time is not None:
            try:
                # use strftime and decode with utf-8 (in case of accented
                # string)
                txt = self.data.time.tolist().strftime(txt).decode('utf-8')
            except AttributeError:
                pass
        txt = super(MapBase, self)._replace(
            txt=txt, fig=fig, delimiter=delimiter)
        return txt

    def _setupproj(self):
        """Set up basemap projection on plot"""
        self.logger.debug('Setting up projection...')
        if not self.fmt.enable and self._draw_mapproj and (
                not hasattr(self, 'wind') and self.wind is not None):
            self.logger.debug('    Drawing of projection not enabled, because')
            self.logger.debug('        fmt.enable: %s', self.fmt.enable)
            self.logger.debug('        self._draw_mapproj: %s',
                              self._draw_mapproj)
            if hasattr(self, 'wind'):
                self.logger.debug('        self.wind: %s', self.wind)
            return
        plt.sca(self.ax)
        self.logger.debug('    Clear axes')
        plt.cla()
        if hasattr(self, 'plot'):
            del self.plot
        self._removecbar(['b', 'r'])
        if hasattr(self, 'wind'):
            if self.wind is not None:
                self.logger.debug('    Remove wind')
                self.wind._removeplot()
                self.wind._removecbar(['b', 'r'])
        self._basemapops = {}
        basemapops = self._basemapops
        mapproj = self.mapproj
        mapproj.ax = self.ax
        basemapops['lineshapes'] = OrderedDict()
        for key, val in self.fmt.lineshapes.items():
            basemapops['lineshapes'][key] = self.mapproj.readshapefile(
                name=key, **val)
        # draw coastlines
        if self.fmt.lsm:
            self.logger.debug('    Draw land-sea-mask')
            basemapops['lsm'] = mapproj.drawcoastlines()
        # color the ocean
        self._draw_lsmask(self.fmt.ocean_color, self.fmt.land_color)
        self._draw_shares(*self._shared)
        # draw parallels
        basemapops['parallels'] = mapproj.drawparallels(
            self.fmt._paraops['parallels'],
            **{key: value for key, value in self.fmt._paraops.items()
               if key != 'parallels'})
        # draw meridians
        basemapops['meridionals'] = mapproj.drawmeridians(
            self.fmt._meriops['meridionals'],
            **{key: value for key, value in self.fmt._meriops.items()
               if key != 'meridionals'})
        # draw countries
        if self.fmt.countries:
            self.logger.debug('    Draw countries')
            basemapops['countries'] = mapproj.drawcountries()
        # save mapproj to attribute
        self._basemapops = basemapops
        self.logger.debug('    Done.')

    def _draw_lsmask(self, ocean_color, land_color):
        self.logger.debug("Draw land sea mask...")
        mapproj = self.mapproj
        basemapops = self._basemapops

        if 'lsmask' in self._shared:
            self.logger.debug(
                "Land sea mask is shared with %s --> return",
                self._shared['lsmask'].mapo.name)
            return
        # remove current land sea mask
        try:
            for poly in basemapops['lsmask']:
                poly.remove()
            del basemapops['lsmask']
        except TypeError:
            basemapops['lsmask'].remove()
            mapproj.lsmask = None
            del basemapops['lsmask']
        except KeyError:
            pass
        if ocean_color is not None:
            self.logger.info("Color land and oceans...")
            try:
                self.logger.debug("Try dict...")
                options = dict(ocean_color)
            except ValueError:
                self.logger.debug("Failed...", exc_info=True)
                options = {'ocean_color': ocean_color}
            options.setdefault('land_color', land_color or 'w')
            basemapops['lsmask'] = mapproj.drawlsmask(**options)
        elif land_color is not None:
            self.logger.info("Color land...")
            try:
                self.logger.debug("Try dict...")
                options = dict(land_color)
            except ValueError:
                self.logger.debug("Failed...", exc_info=True)
                options = {'color': land_color, 'zorder': 0}
            basemapops['lsmask'] = mapproj.fillcontinents(**options)

    def _configureaxes(self):
        """Configure axes and labels, in particular colorbar labels"""
        super(MapBase, self)._configureaxes()
        if hasattr(self, 'cbar') and self.fmt.clabel is not None:
            for cbarpos in self.cbar:
                label = self._replace(self.fmt.clabel)
                self.logger.debug("Set cbar label: %s", str(label))
                self.cbar[cbarpos].set_label(label,
                                             **self.fmt._labelops)
        elif hasattr(self, 'cbar'):
            for cbarpos in self.cbar:
                self.logger.debug('Delete label from colorbar')
                self.cbar[cbarpos].set_label('')

    def _draw_colorbar(self):
        self.logger.debug('Draw colorbars')
        if not hasattr(self, 'cbar'):
            self.cbar = {}
        for cbarpos in self.fmt.plotcbar:
            if cbarpos in self.cbar:
                self.logger.debug('Found position %s --> update', cbarpos)
                if cbarpos in ['b', 'r']:
                    self.cbar[cbarpos].update_bruteforce(self.plot)
                else:
                    plt.figure(self.cbar[cbarpos].ax.get_figure().number)
                    self.cbar[cbarpos].set_cmap(self._cmap)
                    self.cbar[cbarpos].set_norm(self._norm)
                    self.cbar[cbarpos].draw_all()
                    plt.draw()
            else:
                self.logger.debug('New position %s --> draw colorbar',
                                  cbarpos)
                orientations = ['horizontal', 'vertical',
                                'horizontal', 'vertical']
                cbarlabels = ['b', 'r', 'sh', 'sv']
                self.fmt.check_key(cbarpos, possible_keys=cbarlabels)
                orientation = orientations[cbarlabels.index(cbarpos)]
                if cbarpos in ['b', 'r']:
                    self.logger.debug('Draw colorbar in existing figure')
                    if isinstance(self.plot, mpl.streamplot.StreamplotSet):
                        self.cbar[cbarpos] = plt.colorbar(
                            self.plot.lines, orientation=orientation,
                            extend=self.fmt.extend, use_gridspec=True)
                    else:
                        self.cbar[cbarpos] = plt.colorbar(
                            self.plot, orientation=orientation,
                            extend=self.fmt.extend, use_gridspec=True)
                elif cbarpos in ['sh', 'sv']:
                    self.logger.debug('Draw colorbar in new figure')
                    if cbarpos == 'sh':
                        fig = plt.figure(figsize=(8, 1))
                        ax = fig.add_axes([0.05, 0.5, 0.9, 0.3])
                    elif cbarpos == 'sv':
                        fig = plt.figure(figsize=(1, 8))
                        ax = fig.add_axes([0.3, 0.05, 0.3, 0.9])
                    fig.canvas.set_window_title(
                        'Colorbar, var %s, time %i, level %i.' % (
                            self.var, self.time, self.level))
                    self.cbar[cbarpos] = mpl.colorbar.ColorbarBase(
                        ax, cmap=self._cmap, norm=self._norm,
                        orientation=orientation, extend=self.fmt.extend)
                    plt.sca(self.ax)

            # draw the separate colorbars
            if cbarpos in ['sh', 'sv']:
                plt.figure(self.cbar[cbarpos].ax.get_figure().number)
                self.cbar[cbarpos].draw_all()
                plt.draw()

        # set ticks
        self.ticks = self.fmt.ticks
        # set ticklabels
        self.ticklabels = self.fmt.ticklabels

    def gridweights(self, data=None, expand=True):
        """Calculates weights for the specified dimensions."""
        self.logger.debug("Calculate weights")
        if data is None:
            data = self.data
        if self.reader._udim(self.get_var()):  # return equal weights
            self.logger.debug("Found triangular grid --> return equal weights")
            tile_shape = list(data.shape)
            dims = list(data.dimensions)
            ispatial = data._DataField__spatial[0]
            if expand:
                tile_shape[ispatial] = 1
                weights = np.ma.ones(data.shape)
                weights.mask = data.mask
                for ind in data._iter_indices(ispatial):
                    weights[ind] = weights[ind]/weights[ind].sum()
                #weights /= weights.sum(ispatial)
            else:
                weights = np.ones(data.shape[ispatial])
                weights /= weights.sum()
            return weights
        try:
            self.logger.debug('    Try cdos...')
            from .._cdo import Cdo
            cdo = Cdo()
            if hasattr(self.reader, '_grid_file'):
                ifile = self.reader._grid_file
                self.logger.debug('    Use grid of %s', ifile)
                delete = False
            else:
                ifile = tempfile.NamedTemporaryFile(prefix='tmp_nc2map')
                ifile = ifile.name
                self.logger.debug('    Dump to file %s', ifile)
                self.reader.selname().dump_nc(ifile)
                delete = True
            self.logger.debug('    Calculate with cdos')
            weights = cdo.gridweights(
                input=ifile, returnData='cell_weights').shift_data(
                    self.mapproj).mask_outside(self.mapproj)
            mask = data.mask
            if mask.ndim != weights.ndim and expand:
                tile_shape = list(data.shape)
                dims = list(data.dimensions)
                ilat = dims.index(data._DataField__lat)
                ilon = dims.index(data._DataField__lon)
                tile_shape[ilat] = 1
                tile_shape[ilon] = 1
                weights = np.ma.array(np.tile(weights[:], tile_shape),
                                    mask=mask)
                for ind in data._iter_indices(ilat, ilon):
                    weights.__setitem__(
                        ind, weights.__getitem__(ind)/ \
                            weights.__getitem__(ind).sum())
            else:
                weights.mask.take = mask
                weights.data /= weights.sum()

            if delete:
                ifile.close()
            self.logger.debug("    Done")
            del cdo
            return weights[:]
        except ImportError:
            self.logger.debug(
                'Could not import cdos --> use DataField.weights() method')
            return data.gridweights
        except:
            self.logger.debug(
                'Failed to calculate weights', exc_info=True)
            critical(
                "Attention! Could not calculate weights! Ignoring "
                "weights causes errors in longitude-latitude grids!",
                logger=self.logger)
            return None

    def _removecbar(self, positions='all', resize=False):
        self.logger.debug('    Cbars to remove: %s', positions)
        if not hasattr(self, 'cbar'):
            self.logger.debug("No cbars to remove...")
            return
        if positions == 'all':
            positions = self.cbar.keys()
        positions = set(positions).intersection(self.cbar.keys())
        self.logger.debug('    Positions found: %s', ', '.join(positions))
        for cbarpos in positions:
            self.logger.debug(
                '        Remove cbar with position %s', cbarpos)
            if cbarpos in ['b', 'r']:
                # delete colorbar axes
                try:
                    self.cbar[cbarpos].remove()
                except AttributeError:
                    # do it manually for older versions
                    fig = self.cbar[cbarpos].ax.get_figure()
                    fig.delaxes(self.cbar[cbarpos].ax)
                # reset geometry
                if resize and self._resize:
                    self._reset_geometry()
            elif cbarpos in ['sh', 'sv']:
                plt.close(self.cbar[cbarpos].ax.get_figure())
            del self.cbar[cbarpos]
        return

    def asdict(self, shared=False):
        """returns formatoptions and dimensions (time, level) as dictionary"""
        fmt = self.fmt.asdict()
        for key, val in self.dims.items():
            if self.dims_orig[key] != val:
                fmt[key] = val
        if shared:
            fmt['_shared'] = self.share.asdict()
        return fmt

    def _reset_geometry(self):
        """Resets geometry of the Axes to original position"""
        self.logger.debug("Resetting geometry...")
        shape = self._get_axes_shape()
        num = self._get_axes_num()
        self.ax.change_geometry(shape[0], shape[1], num)

    def get_cbars(self, positions='all'):
        """returns a list of all colorbars belonging to position

        Input:
          - position: Iterable containing 'b', 'r', 'sh', 'sv'

        Returns [list of colorbars specified by positions]
        """
        if positions == 'all':
            return self.cbar.values()
        else:
            return [val for key, val in self.cbar.items() if key in positions]

    def _update_ticks(self, kwargs):
        """update ticks around map and at colorbar"""
        if any(key in kwargs for key in ['ticksize', 'fontsize', 'fontweight',
                                         'tickweight']):
            for label in ['parallels', 'meridionals']:
                if not self._basemapops.get(label):
                    continue
                for key, val in self._basemapops[label].items():
                    for key in ['fontsize', 'ticksize']:
                        if key in kwargs:
                            for text in val[1]:
                                text.set_size(kwargs[key])
                    for key in ['fontweight', 'tickweight']:
                        if key in kwargs:
                            for text in val[1]:
                                text.set_weight(kwargs[key])

    def _update_mapproj(self, kwargs, plot=True):
        """update map projection from dictionary kwargs"""
        self.logger.debug('Updating map projection...')
        if not self.fmt.enable or not self._draw_mapproj:
            self.logger.debug('    Drawing of projection not enabled, because')
            self.logger.debug('        fmt.enable: %s', self.fmt.enable)
            self.logger.debug('        self._draw_mapproj: %s',
                              self._draw_mapproj)
            return
        basemapops = self._basemapops
        if 'lineshapes' in kwargs:
            fshapes = self.fmt.lineshapes
            my_shapes = basemapops['lineshapes']
            shared_shapes = self._shared.get('lineshapes', {}).keys()
            shapes_to_remove = (
                item for item in my_shapes.items()
                if item[0] not in list(fshapes) + shared_shapes)
            for key, val in shapes_to_remove:
                val[-1].remove()
                del my_shapes[key]
            for key, val in fshapes.items():
                if key in my_shapes:
                    continue
                my_shapes[key] = self.mapproj.readshapefile(name=key, **val)
        if 'meridionals' in kwargs or 'merilabelpos' in kwargs:
            self.logger.debug('Found meridionals options in kwargs...')
            keys = basemapops['meridionals'].keys()
            self.logger.debug('    Delete exsting keys: %s',
                              ', '.join(map(str, keys)))
            for key in keys:
                del basemapops['meridionals'][key]
            self.fmt.meridionals = self.fmt.meridionals
            self.fmt.merilabelpos = self.fmt.merilabelpos
            if plot:
                self.logger.debug('    Draw new keys: %s', ', '.join(map(
                    str, self.fmt._meriops['meridionals'])))
                basemapops['meridionals'] = self.mapproj.drawmeridians(
                    self.fmt._meriops['meridionals'],
                    **{key: value for key, value in self.fmt._meriops.items()
                       if key != 'meridionals'})
        if 'parallels' in kwargs or 'paralabelpos' in kwargs:
            self.logger.debug('Found parallels options in kwargs...')
            keys = basemapops['parallels'].keys()
            self.logger.debug('    Delete exsting keys: %s',
                              ', '.join(map(str, keys)))
            for key in keys:
                del basemapops['parallels'][key]
            self.fmt.parallels = self.fmt.parallels
            self.fmt.paralabelpos = self.fmt.paralabelpos
            if plot:
                self.logger.debug('    Draw new keys: %s', ', '.join(map(
                    str, self.fmt._paraops['parallels'])))
                basemapops['parallels'] = self.mapproj.drawparallels(
                    self.fmt._paraops['parallels'],
                    **{key: value for key, value in self.fmt._paraops.items()
                       if key != 'parallels'})
        if 'lsm' in kwargs:
            self.logger.debug("Found 'lsm' in kwargs...")
            if 'lsm' in basemapops:
                self.logger.debug('    Delete land-sea-mask from plot')
                basemapops['lsm'].remove()
                del basemapops['lsm']
            if kwargs['lsm']:
                self.logger.debug('    Draw new land-sea-mask')
                basemapops['lsm'] = self.mapproj.drawcoastlines()
        if 'countries' in kwargs:
            self.logger.debug("Found 'countries' in kwargs...")
            if 'countries' in basemapops:
                self.logger.debug('    Delete countries from plot')
                basemapops['countries'].remove()
                del basemapops['countries']
            if kwargs['countries']:
                self.logger.debug('    Draw new countries')
                basemapops['countries'] = self.mapproj.drawcountries()

    def copy(self, reader=None, ax=None, name=None):
        """Returns a copy of the MapBase instance

        Input:
          - reader: An ArrayReader instance. If None, the one of the MapBase
              instance is used.
          - ax: Matplotlib.axes.AxesSubplot instance. If None, a new one will
              be created as soon as you perform a plot
          - name: Name of the new MapBase instance. If None, it will be set to
              the name of this MapBase instance

        Output:
          a new MapBase instance with the exact class as this one"""
        if reader is None:
            reader = self.reader
        if name is None:
            name=self.name
        return self.__class__(reader, name=name, fmt=self.asdict(), ax=ax,
                              **self.dims)

    def extract_in_reader(self, full_data=False, mask_data=True, **kwargs):
        """Returns a reader with the data corresponding to this MapBase
        instance

        Input:
          - full_data: True/False (Default: False). If True, all data, not
              only the current time step, level, etc. is used.
          - mask_data: True/False (Default: True). If True, the formatoption
              masking options (including lonlatbox, mask, maskbelow, etc.)
              is used. Otherwise the full data as stored in the corresponding
              ArrayReader instance is used. This will also change the grid to
              match the projection limits of the MapBase instance.
        """
        def to_slices(dims):
            for key, val in dims.items():
                if isinstance(val, int):
                    dims[key] = slice(val, val+1)
                elif key in self.reader._timenames:
                    val = self.reader.get_time_slice(val)
                    if isinstance(val, int):
                        dims[key] = slice(val, val+1)
                    else:
                        dims[key] = val
            return dims
        self.logger.debug("Extracting MapBase instance in reader")
        kwargs_keys = ['full_data', 'mask_data']
        max_kwarg_len = max(map(len, kwargs_keys + kwargs.keys())) + 2
        self.logger.debug("Original kwargs:")
        for key in kwargs_keys:
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              key, str(locals()[key]))
        for dim, val in kwargs.items():
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              dim, str(val))
        var = self.dims.get('var')
        vlst = self.dims.get('vlst')
        single_var = None if var is None else True
        if not single_var:
            varname = vlst[0]
        else:
            varname = var
        if not full_data:
            dims = self.dims.copy()
            dims.update(**kwargs)
            dims = to_slices(dims)
            reader = self.reader.extract(**dims)
        else:
            if single_var:
                reader = self.reader.selname(var, copy=True)
            else:
                reader = self.reader.selname(*vlst, copy=True)
        if mask_data:
            if full_data:
                data = self.reader.get_data(
                    var=var, vlst=vlst, datashape='any', **to_slices(kwargs))
            else:
                data = self.reader.get_data(datashape='any', **dims)
            data = self._mask_data(
                data.shift_data(self.mapproj).mask_outside(self.mapproj))
            if not self.reader._udim(self.get_var()):
                lon = reader.lon.data
                lat = reader.lat.data
                mapproj = Basemap(llcrnrlon=lon.min(), urcrnrlon=lon.max(),
                                    llcrnrlat=lat.min(), urcrnrlat=lat.max())
                # shift back to original settings
                data.shift_data(mapproj)
            if single_var:
                reader.variables[var].data = data[:]
            else:
                for i, var in enumerate(vlst):
                    reader.variables[var].data = data[i, :]
        return reader

    def _draw_shares(self, *args):
        """Draws shared lsmask and lineshapes
        *args may be arguments from self._shared"""
        if 'lsmask' in args:
            self._shared['lsmask'].lsmask([self])
        if 'lineshapes' in args:
            my_shapes = self._basemapops.setdefault('lineshapes', {})
            for shape, share in self._shared['lineshapes'].items():
                share.lineshapes([self], shape)

    def get_var(self):
        """Returns the variable in the reader, from which the data is
        extracted (or one of the variables if self.dims['vlst'], i.e. if
        WindPlot"""
        var = self.dims.get('var')
        var = var or self.dims['vlst'][0]
        return self.reader.variables[var]

    def close(self, num):
        if not num % 2:
            self.logger.debug("    plot")
            self._removeplot()
        if not num % 3:
            self.logger.debug("    colorbar")
            plt.close(self.ax.get_figure())
            self._removecbar(['sh', 'sv'])
        if not num % 5:
            self.logger.debug("    data")
            del self.data
        if not num % 7:
            self.logger.debug("    reader")
            num /= 7
            self.reader.close()

    def __iadd__(self, value):
        """Self agglomeration method of MapBase class"""
        for key, val in self.dims.items():
            if isinstance(val, int):
                self.dims[key] = 0
        try:  # assume MapBase instance
            self.reader = self.extract_in_reader() + value.extract_in_reader()
        except AttributeError:
            self.reader = self.extract_in_reader() + value
        self.data = self.get_data()
        return self

    def __add__(self, value):
        """Agglomeration method of MapBase class"""
        name = self.name
        try:
            name += '+%s' % value.name
        except AttributeError:
            name=None
        mapo = self.copy(name=name)
        mapo += value
        return mapo

    def __imul__(self, value):
        """Self multiplication method of MapBase class"""
        for key, val in self.dims.items():
            if isinstance(val, int):
                self.dims[key] = 0
        try:  # assume MapBase instance
            self.reader = self.extract_in_reader() * value.extract_in_reader()
        except AttributeError:
            self.reader = self.extract_in_reader() * value
        self.data = self.get_data()
        return self

    def __mul__(self, value):
        """Multiplication method of MapBase class"""
        name = self.name
        try:
            name += '*%s' % value.name
        except AttributeError:
            name=None
        mapo = self.copy(name=name)
        mapo *= value
        return mapo

    def __idiv__(self, value):
        """Self division method of MapBase class"""
        for key, val in self.dims.items():
            if isinstance(val, int):
                self.dims[key] = 0
        try:  # assume MapBase instance
            self.reader = self.extract_in_reader() / value.extract_in_reader()
        except AttributeError:
            self.reader = self.extract_in_reader() / value
        self.data = self.get_data()
        return self

    def __div__(self, value):
        """Division method of MapBase class"""
        name = self.name
        try:
            name += '/%s' % value.name
        except AttributeError:
            name = None
        mapo = self.copy(name=name)
        mapo /= value
        return mapo

    def __isub__(self, value):
        """Self subtraction method of MapBase class"""
        for key, val in self.dims.items():
            if isinstance(val, int):
                self.dims[key] = 0
        try:  # assume MapBase instance
            self.reader = self.extract_in_reader() - value.extract_in_reader()
        except AttributeError:
            self.reader -= value
        self.data = self.get_data()
        return self

    def __sub__(self, value):
        """Subtraction method of MapBase class"""
        name = self.name
        try:
            name += '-%s' % value.name
        except AttributeError:
            name=None
        mapo = self.copy(name=name)
        mapo -= value
        return mapo

    def __ipow__(self, value):
        """Self power method of MapBase class"""
        for key, val in self.dims.items():
            if isinstance(val, int):
                self.dims[key] = 0
        try:  # assume MapBase instance
            self.reader = self.extract_in_reader() ** value.extract_in_reader()
        except AttributeError:
            self.reader = self.extract_in_reader() ** value
        self.data = self.get_data()
        return self

    def __pow__(self, value):
        """Power method of MapBase class"""
        name = self.name
        try:
            name += '**%s' % value.name
        except AttributeError:
            name=None
        mapo = self.copy(name=name)
        mapo **= value
        return mapo

    def __str__(self):
        dim_str = ', '.join(
            map(lambda item: "%s: %s" % item, self.dims.items()))
        return "nc2map.mapos.%s %s of %s" % (
            self.__class__.__name__, self.name, dim_str)
