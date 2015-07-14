# -*- coding: utf-8 -*-
"""Module containing the FieldPlot class for scalar fields on the Earth

This class plots scalar variables (e.g. temperature) on the globe. Optionally a
overlayed vector field (e.g. wind) can be plotted as well."""
from itertools import izip
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib as mpl
from .._basemap import Basemap
from copy import deepcopy
from ..formatoptions import FieldFmt, FmtBase, WindFmt, _get_fmtkeys_formatted
from ..warning import warnings, Nc2MapWarning, warn
from .._cmap_ops import get_cmap
from _map_base import MapBase, _props, returnbounds
from _windplot import WindPlot


class FieldPlot(MapBase):
    """Class to visualize two-dimensional scalar fields on a map. Optionally
    you can also combine this with a nc2map.mapos.WindPlot instance (see
    __init__ methods for details."""
    # ------------------ define properties here -----------------------
    var = _props.dim('var', """String. Variable name in the reader""")

    @property
    def meta(self):
        meta = super(FieldPlot, self).meta
        if self.wind is not None:
            for key, val in self.wind.meta.items():
                meta.setdefault(key, val)
        return meta

    def __init__(self, reader, var, u=None, v=None, fmt={}, **kwargs):
        """
        Input:
            - var: string. Name of the variable which shall be plotted and red
                the netCDF file
            - u: string (Default: None). Name of the zonal wind variable if a
                 WindPlot shall be visualized (set u and v to also plot the
                 wind above the data of var)
            - v: string (Default: None). Name of the meridional wind variable
                 if a WindPlot shall be visualized
            - fmt: dictionary containing the formatoption keywords as keys and
                 settings as value.
        Keywords inherited from MapBase initialization are: %(mapbase)s
        Possible formatoption keywords are (see nc2map.show_fmtdocs() for
        details): %(fmt)s
        And the windplot specific options are (see nc2map.show_fmtdocs('wind')
        for details): %(windfmt)s
        """
        # docstring is extended below with the formatoption keywords
        self.name = kwargs.get('name', 'mapo')
        self.set_logger()

        kwargs_keys = ['var', 'u', 'v', 'fmt']
        max_kwarg_len = max(map(len, kwargs_keys)) + 2
        self.logger.debug("Original kwargs:")
        for key in kwargs_keys:
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              key, str(locals()[key]))

        super(FieldPlot, self).__init__(reader, fmt=fmt, var=var, **kwargs)
        # define formatoptions
        if isinstance(fmt, dict):
            var = fmt.get('var', var)  # update from variables
            updatewind = u is not None and v is not None
            self.fmt = FieldFmt(**dict(
                item for item in fmt.items() + [('updatewind', updatewind)]
                if item[0] not in self.dims))
            self.logger.debug(
                "Dimensions set up by formatoptions dictionary:")

        elif not fmt:
            self.fmt = FieldFmt()
        else:
            if not isinstance(fmt, FieldFmt):
                warn('Expected FieldFmt instance but found %s for '
                     'formatoptions!' % str(type(fmt)))
            self.fmt = fmt

        # create basemap
        self.logger.debug(
            "Initialize Basemap instance with %s", self.fmt._projops)
        self.mapproj = Basemap(**self.fmt._projops)

        self.data = self.get_data(var)
        self._reinitialized = 1
        if u is not None and v is not None:
            self.wind = WindPlot(
                self.reader, time=self.time, level=self.level, ax=self._ax,
                fmt=self.fmt.windplot, u=u, v=v, mapsin=self._mapsin,
                name=self.name, mapproj=self.mapproj)
            self.wind._draw_mapproj = 0
        else:
            self.wind = None

    def _reinit(self, u=None, v=None, **kwargs):
        """Function to reinitialize the data"""
        # set _reinitialized for unstructured grids in make_plot
        self._reinitialized = 1
        self._reinitialize = 0
        super(FieldPlot, self)._reinit(**kwargs)
        zipped_kwargs = izip(['u', 'v'], [u, v])
        not_none_kwargs = (key for key, val in zipped_kwargs
                           if val is not None)
        self.logger.debug('Reinit %s', ', '.join(not_none_kwargs))
        self.data = self.get_data(self.var)
        if self.wind is not None:
            self.wind.mapproj = self.mapproj
            self.wind._reinit(u=u, v=v)

    def update(self, todefault=False, plot=True, force=False, **kwargs):
        """Update the MapBase instance, formatoptions, variable, time and
        level.
        Possible keywords (kwargs) are
          - all keywords as set by formatoptions (see initialization __init__)
          - time: integer. Sets the time of the MapBase instance and the data
          - level: integer. Sets the level of the MapBase instance and the data
          - var: string. Sets the variable of the MapBase instance and the data
        Additional keys in the windplot dictionary are:
          - u: string. Sets the variable of the WindPlot instance and reloads
              the data
          - v: Sets the variable of the WindPlot instance and reloads the data
          - time, level and var as above
        """
        if not hasattr(self, '_basemapops'):
            self.logger.info("Setting up projection...")
            self._setupproj()
        # set current axis
        plt.sca(self.ax)
        self.logger.debug('Updating...')
        self.logger.debug('Plot after finishing: %s', plot)
        # patterns for warnings that shall be depreciated

        # check the keywords
        dims = self.dims.copy()
        dimsorig = self.dims_orig
        for dim in self.data.dims.dtype.fields.keys():
            dims.setdefault(dim, 0)
            dimsorig.setdefault(dim, 0)
        possible_keys = self.fmt._default.keys() + dims.keys()
        for key in kwargs:
            self.fmt.check_key(key, possible_keys=possible_keys)

        # save current formatoptions
        old_fmt = self.fmt.asdict()

        # delete formatoptions which are already at the wished state
        if not todefault and not force:

            kwargs = {key: value for key, value in kwargs.items()
                      if (not isinstance(value, dict) and
                          np.all(value != old_fmt.get(
                          key, dims.get(key, self.fmt._default.get(key)))))}
        elif not force:
            self.logger.debug('Update to default...')
            oldkwargs = kwargs.copy()
            defaultitems = self.fmt._default.items()
            kwargs = {
                key: kwargs.get(key, value) for key, value in defaultitems
                if key != 'windplot' and (
                    (key not in kwargs and np.all(
                        value != getattr(self.fmt, key)))
                    or (key in kwargs and np.all(
                        kwargs[key] != getattr(self.fmt, key))))}
            if not self.fmt._enablebounds:
                kwargs = {key: value for key, value in kwargs.items()
                          if key not in ['cmap', 'bounds']}
            # update dimensions
            kwargs.update({
                key: oldkwargs.get(key, value)
                for key, value in dimsorig.items()
                if (dims[key] != dimsorig[key]
                    or (key in oldkwargs and oldkwargs[key] != dims[key]))})
            if 'windplot' in oldkwargs:
                kwargs['windplot'] = oldkwargs['windplot']
            else:
                kwargs['windplot'] = {}

        self.logger.debug("Update to ")
        try:
            max_kwarg_len = max(map(len, kwargs.keys())) + 3
        except ValueError:
            max_kwarg_len = 0
        for key, val in kwargs.items():
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              key, str(val))

        try:  # False: Remove plot, True: make plot, None: do nothing
            enable = bool(kwargs['enable'])
            self.logger.debug('Updating enable to %s', enable)
            self.fmt.enable = enable
        except KeyError:
            enable = None  # no change

        # update plotting of cbar properties and close cbars
        if 'plotcbar' in kwargs:
            self.logger.debug(
                "Found 'plotcbar' in kwargs --> remove non-used cbars")
            if kwargs['plotcbar'] in [False, None]:
                kwargs['plotcbar'] = ''
            if kwargs['plotcbar'] is True:
                kwargs['plotcbar'] = 'b'
            cbars2close = [cbar for cbar in self.fmt.plotcbar
                          if cbar not in kwargs['plotcbar']]
            self._removecbar(cbars2close)

        # update extend and ticklabels
        if ('extend' in kwargs
                or kwargs.get('ticklabels', False) is None):
            self.logger.debug(
                "Found 'extend' or 'ticklabels' in kwargs --> remove cbars")
            self._removecbar(resize=False)

        # update masking options
        maskprops = {key: value for key, value in kwargs.items()
                     if key in self.fmt._maskprops}
        if maskprops:
            self.logger.debug(
                'Found masking properties: %s. --> reinit',
                ', '.join(maskprops.keys()))
            self.fmt.update(**maskprops)

        # update basemap properties
        bmprops = {key: value for key, value in kwargs.items()
                   if key in self.fmt._bmprops}
        if bmprops:
            self.logger.debug(
                'Found basemap properties: %s. --> reinit',
                ', '.join(bmprops.keys()))
            with warnings.catch_warnings():
                if self.wind is None and not 'windplot' in kwargs:
                    warnings.filterwarnings(
                        "ignore", 'Stereographic', Nc2MapWarning,
                        'nc2map.formatoptions._windfmt', 0)
                self.fmt.update(**bmprops)

        # update mapobject dimensions and reinitialize
        newdims = {key: value for key, value in kwargs.items()
                   if key in dims.keys()}
        if newdims:
            self.logger.debug(
            'Found new dimensions: %s. --> reinit',
            ', '.join(newdims.keys()))

        if newdims or bmprops or maskprops or enable or self._reinitialize:
            if bmprops:
                self.mapproj = Basemap(**self.fmt._projops)
            self._reinit(**newdims)
            self._make_plot = 1
        if ((bmprops or not hasattr(self, '_basemapops'))
                and plot):
            self._setupproj()

        self._make_plot = self._make_plot or set(kwargs) & set(
            ['cmap', 'bounds', 'norm', 'opacity', 'rasterized', 'grid'])

        # color oceans and land
        if any(key in kwargs for key in ['ocean_color', 'land_color']):
            self._draw_lsmask(kwargs.get('ocean_color', self.fmt.ocean_color),
                              kwargs.get('land_color', self.fmt.land_color))


        # handle plot enabling changing
        if enable is False:
            self.logger.debug("Found 'enable' in kwargs --> remove plot")
            self._removeplot()

        if ('grid' in kwargs or self._make_plot) and hasattr(self, 'grid'):
            for line in self.grid:
                line.remove()
            del self.grid

        # update rest
        self.logger.debug('Update fmtBase instance')
        fmt_update_kwargs = {key: value for key, value in kwargs.items()
                             if key not in self.fmt._bmprops + dims.keys()}
        for key in dims.keys():
            try:
                fmt_update_kwargs['windplot'].pop(key)
            except KeyError:
                pass
        self.fmt.update(**fmt_update_kwargs)

        # update ticks
        self._update_ticks(kwargs)

        # update map projection
        self._update_mapproj(kwargs=kwargs, plot=plot)

        # update wind
        if self.wind is not None or 'windplot' in kwargs:
            if 'windplot' in kwargs:
                self.logger.debug("Found 'windplot' in kwargs...")
                windops = kwargs['windplot']
                # if WindPlot is currently not set, enable it now
                if (windops.get('u', None) is not None
                        and windops.get('v', None) is not None
                        and self.fmt.windplot.enable
                        and self.wind is None):
                    self.logger.debug("No windplot set so far --> Set it now")
                    self.wind = WindPlot(
                        reader=self.reader, time=self.time, level=self.level,
                        ax=self._ax, fmt=self.fmt.windplot, u=windops['u'],
                        v=windops['v'], mapsin=self._mapsin, name=self.name,
                        mapproj=self.mapproj)
                    self.wind._draw_mapproj = 0
                for key in ['time', 'level']:
                    if key in kwargs:
                        windops.setdefault(key, kwargs[key])
            else:
                windops = {}
            if self.wind is not None:
                self.logger.debug('Updating wind...')
                windops.update({key: getattr(self.fmt, '_'+key) for key in
                                    self.fmt._bmprops})
                self.wind.update(plot=False, todefault=todefault, **windops)
                self.wind._make_plot = self._make_plot or self.wind._make_plot
        if plot:
            self.make_plot()
        else:
            self._make_plot = 0

        self.logger.debug('Update Done.')

    def _mask_data(self, data):
        data = super(FieldPlot, self)._mask_data(data)
        if self.fmt.maskless is not None:
            self.logger.debug('Masking below %s', self.fmt.maskless)
            data[:] = np.ma.masked_less(data[:], self.fmt.maskless, copy=True)
        if self.fmt.maskleq is not None:
            self.logger.debug('Masking below or equal to %s',
                              self.fmt.maskleq)
            data[:] = np.ma.masked_less_equal(data[:], self.fmt.maskleq,
                                           copy=True)
        if self.fmt.maskgreater is not None:
            self.logger.debug('Masking above %s', self.fmt.maskgreater)
            data[:] = np.ma.masked_greater(data[:], self.fmt.maskgreater,
                                           copy=True)
        if self.fmt.maskgeq is not None:
            self.logger.debug('Masking above or equal to %s',
                              self.fmt.maskgeq)
            data[:] = np.ma.masked_greater_equal(data[:], self.fmt.maskgeq,
                                              copy=True)
        if self.fmt.maskbetween is not None:
            self.logger.debug('Masking between %s', str(self.fmt.maskbetween))
            data[:] = np.ma.masked_inside(data[:], self.fmt.maskbetween[0],
                                       self.fmt.maskbetween[1], copy=True)
        return data

    def _moviedata(self, times, nowind=False, **kwargs):
        """generator to get the data for the movie"""
        try:
            windops = kwargs.pop('windplot')
        except KeyError:
            windops = {}
        kwargs = {key: iter(val) for key, val in kwargs.items()}
        windops = {key: iter(val) for key, val in windops.items()}
        for i, time in enumerate(times):
            # yield time step, data, formatoptions
            fmt = {key: next(value) for key, value in kwargs.items()}
            fmt.update({key: next(value) for key, value in windops.items()})
            yield (time, fmt)

    def _setupproj(self):
        # docstring is set equal to MapBase _setupproj method below
        super(FieldPlot, self)._setupproj()
        if self.wind is not None:
            self.wind.mapproj = self.mapproj

    def _runmovie(self, args):
        """Function to update the movie with args from self._moviedata"""
        if 'windplot' not in args[-1]:
            args[-1]['windplot'] = {}
        self.update(time=args[0], **args[-1])
        return

    def _removeplot(self):
        """Removes the plot from the axes and deletes the plot property from
        the instance"""
        self.logger.debug("    Removing plot...")
        if hasattr(self, 'plot'):
            try:
                self.plot.remove()
            except:
                pass
            try:
                for line in self.grid:
                    line.remove()
                del self.grid
            except AttributeError:
                pass
            del self.plot
        self._removecbar()

    def make_plot(self):
        """Make the plot with the current settings and the WindPlot. Use it
        after reinitialization and _setupproj. Don't use this function
        to update the plot but rather the update function!"""
        self.logger.debug('Making plot...')
        if self.fmt.enable and self._make_plot:
            if not hasattr(self, '_basemapops'):
                self.logger.info("Setting up projection...")
                self._setupproj()
            plt.sca(self.ax)
            if not self.fmt._enablebounds:
                pass
            elif self.fmt.bounds[0] in ['rounded', 'sym', 'minmax',
                                      'roundedsym']:
                self.logger.debug('    Calculate bounds with %s',
                                  str(self.fmt.bounds))
                self._bounds = returnbounds(self.data[:], self.fmt.bounds)
            else:
                self.logger.debug('   Found manually set bounds')
                self._bounds = self.fmt.bounds
            if self.fmt._enablebounds:
                if self.fmt.norm == 'bounds':
                    self._cmap = get_cmap(self.fmt.cmap, len(self._bounds)-1)
                    self._norm = mpl.colors.BoundaryNorm(self._bounds,
                                                         self._cmap.N)
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
            if hasattr(self, 'plot'):
                self.logger.debug('    Remove plot')
                self.plot.remove()
                # does not work atm
                # self.plot.set_cmap(self._cmap)
                # self.plot.set_norm(self._norm)
                # self.plot.set_array(self.data[:-1,:-1].ravel())
            # else:
                # make plot
            self.logger.debug('    Draw plot')
            if (self.reader._udim(self.get_var())
                    and self.fmt.plottype == 'quad'):
                self.logger.debug(
                    "Found unstructured variable. I set the 'plottype' "
                    "keyword to 'tri'")
                self.fmt.plottype = 'tri'
            if self.fmt.plottype == 'quad':
                if self.fmt.grid:
                    settings = {'edgecolors': self.fmt.grid}
                else:
                    settings = {'edgecolors': 'face'}
                lat2d, lon2d = self.data.grid
                self.plot = self.mapproj.pcolormesh(
                    lon2d, lat2d, self.data[:], cmap=self._cmap,
                    norm=self._norm, rasterized=self.fmt.rasterized,
                    latlon=self.fmt.latlon, **settings)
            elif self.fmt.plottype == 'tri':
                if self.data.triangles:
                    triangles = self.data.triangles
                    self.plot = self.mapproj.tripcolor(
                        triangles, self.data[:], cmap=self._cmap,
                        norm=self._norm, rasterized=self.fmt.rasterized,
                        alpha=None,
                        latlon=False if not self._reinitialized else \
                            self.fmt.latlon)
                else:
                    triangles = Triangulation(self.data.lon, self.data.lat)
                    self.plot = self.mapproj.tripcolor(
                        triangles, self.data[:], cmap=self._cmap,
                        norm=self._norm, rasterized=self.fmt.rasterized,
                        latlon=self.fmt.latlon)
                if self.fmt.grid:
                    try:
                        self.grid = self.mapproj.triplot(
                            triangles, latlon=False, **self.fmt.grid)
                    except TypeError:
                        self.grid = self.mapproj.triplot(
                            triangles, self.fmt.grid, latlon=False)

            else:
                raise ValueError("Unknown plottype %s" % self.fmt.plottype)

        if self.fmt.enable and not (
                self.fmt.plotcbar == '' or self.fmt.plotcbar == ()
                or self.fmt.plotcbar is None or self.fmt.plotcbar is False):
            self._draw_colorbar()

        if self.wind is not None:
            if self.wind.fmt.enable:
                self.logger.debug('Wind is enabled --> make windplot')
                if not hasattr(self, "_basemapops"):
                    self.wind._setupproj()
                self.wind.make_plot()

        self._configureaxes()
        self._make_plot = 0
        self._reinitialized = 0
        if self.fmt.tight:
            self.logger.debug("'tight' is True --> make tight_layout")
            plt.tight_layout()

    def close(self, num=0):
        """Close the MapBase instance.
        Arguments may be
        - 'data': To delete all data (but not close the netCDF4.MFDataset
        instance)
        - 'figure': To close the figure and the figures of the colorbars.
        Without any arguments, everything will be closed (including the
        netCDF4.MFDataset instance) and deleted.
        """
        self.logger.debug("Closing...")
        super(FieldPlot, self).close(num)
        if not num % 7:
            num /= 7
        if self.wind is not None:
            self.wind.close(num)

    def asdict(self, **kwargs):
        """Returns a dictionary containing the current formatoptions and (if
        the time, level or var changed compared to the original
        initialization) the time, level or var"""
        fmt = super(FieldPlot, self).asdict(**kwargs)
        #if self.var != self.varorig:
            #fmt.update({'var': self.var})
        if self.wind is not None:
            fmt.update({'windplot': self.wind.asdict()})
        return fmt

    def copy(self, *args, **kwargs):
        """Copy method. Docstring is set below to be equal to the MapBase.copy
        method"""
        mapo = super(FieldPlot, self).copy(*args, **kwargs)
        if self.wind is not None:
            mapo.wind = self.wind.copy(*args, **kwargs)
        return mapo

    # ------------------ modify docstrings here --------------------------
    __init__.__doc__ += __init__.__doc__ % {
        'fmt': '\n' + _get_fmtkeys_formatted(),
        'windfmt': '\n' + _get_fmtkeys_formatted('wind', 'windonly'),
        'mapbase': '\n' + MapBase.__init__.__doc__}
    _setupproj.__doc__ = MapBase._setupproj.__doc__
    copy.__doc__ = MapBase.copy.__doc__
