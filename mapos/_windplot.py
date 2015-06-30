# -*- coding: utf-8 -*-
"""Module containing the definition of the nc2map.mapos.WindPlot class

This class is used to visualize vector data within the nc2map package"""
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .._basemap import Basemap
from ..formatoptions import WindFmt, FmtBase, FieldFmt, _get_fmtkeys_formatted
from .._cmap_ops import get_cmap
from _map_base import MapBase, _props, returnbounds
from ..warning import warn, critical


class WindPlot(MapBase):
    """Class to visualize vector data.

    See __init__ method for initialization keywords."""
    # ------------------ define properties here -----------------------
    # Data properties
    uorig = _props.default(
        'uorig', """
        Original name of the zonal windfield variable as used during
        initialization""")
    vorig = _props.default(
        'vorig', """
        Original name of the meridional windfield variable as used during
        initialization""")
    speed = _props.speed(
        'speed', """numpy.ma.array. Speed as calculated from u and v""")

    @property
    def udata(self):
        """Data array of the zonal wind"""
        return self.data[0, :]

    @udata.setter
    def udata(self, value):
        self.data[0, :] = value

    @property
    def vdata(self):
        """Data array of the meridional wind"""
        return self.data[1, :]

    @vdata.setter
    def vdata(self, value):
        self.data[1, :] = value

    @property
    def vlst(self):
        """Variable list [<name of zonal wind>, <name of meridional wind>]"""
        return self.dims['vlst']

    @vlst.setter
    def vlst(self, value):
        self.dims['vlst'] = value

    @property
    def u(self):
        """Name of the zonal windfield variable."""
        return self.vlst[0]

    @property
    def v(self):
        """Name of the meridional windfield variable."""
        return self.vlst[1]

    @property
    def meta(self):
        """Dictionary. Meta information as stored in the reader"""
        meta = self.reader.get_meta(var=self.u)
        meta.update({'name': self.name, 'var': self.var, 'u': self.u,
                     'v': self.v})
        meta.setdefault('all', '%s of %s' % (
            self.__class__.__name__, ', '.join(
                map(lambda item: '%s: %s' % item, self.dims.items()))))
        meta.update(self.dims)
        return meta

    def __init__(self, reader, u=None, v=None, var='wind', fmt={},
                 mapproj=None, vlst=None, **kwargs):
        """
        Input:
          - u: string. Name of the zonal wind variable
          - v: string. Name of the meridional wind variable
          - var: string (Default: 'wind'). Name of the variable.
          - fmt: dictionary containing the formatoption keywords as keys and
                 settings as value.
          - mapproj: mpl_toolkits.bm.Basemap instance
        Keywords inherited from MapBase initialization are: %(mapbase)s
        Possible formatoption keywords are (see nc2map.show_fmtdocs('wind') for
        details): %(windfmt)s
        """
        if u is None and v is None and vlst is None:
            raise ValueError(
                "Either one of u, v or vlst must not be None!")
        self.name = kwargs.get('name', 'mapo')
        self.set_logger()
        if vlst is None:
            vlst = [u, v]
        kwargs['vlst'] = vlst
        super(WindPlot, self).__init__(reader, fmt=fmt, **kwargs)

        # define formatoptions
        if isinstance(fmt, dict):
            var = fmt.get('var', var)
            u = fmt.get('u', u)
            v = fmt.get('v', v)
            self.fmt = WindFmt(**{key: val for key, val in fmt.items()
                                  if key not in self.dims})
        elif isinstance(fmt, FmtBase):
            self.fmt = fmt
        elif fmt is None:
            self.fmt = WindFmt()
        else:
            raise ValueError(
                'Wrong type %s for formatoptions' % str(type(fmt)))

        # create basemap
        if mapproj is None:
            self.mapproj = Basemap(**self.fmt._projops)
        else:
            self.mapproj = mapproj

        self.varorig = var
        self.uorig = u
        self.vorig = v
        if vlst is None:
            vlst = [u, v]
        self.vlst = vlst
        self.dims_orig['vlst'] = vlst[:]

        self.var = var
        self.data = self.get_data()
        if ((np.array(self.fmt.density) != 1.0).any()
                and not self.fmt.streamplot):
            self._reduceuv(perc=self.fmt.density)
        if self.fmt.reduceabove and not self.fmt.streamplot:
            self._reduceuv(perc=self.fmt.reduceabove[0],
                           pctl=self.fmt.reduceabove[1])

    def _reinit(self, u=None, v=None, **kwargs):
        """Function to reinitialize the data"""
        super(WindPlot, self)._reinit(**kwargs)
        zipped_kwargs = izip(['u', 'v'], [u, v])
        not_none_kwargs = (key for key, val in zipped_kwargs
                           if val is not None)
        self.logger.debug('Reinit %s', ', '.join(not_none_kwargs))
        if hasattr(self, '_speed'):
            del self.speed
        if u is not None:
            self.vlst[0] = u
        if v is not None:
            self.vlst[1] = v
        self.data = self.get_data()
        if ((np.array(self.fmt.density) != 1.0).any()
                and not self.fmt.streamplot):
            self._reduceuv(perc=self.fmt.density)
        if self.fmt.reduceabove is not None and not self.fmt.streamplot:
            self._reduceuv(perc=self.fmt.reduceabove[0],
                           pctl=self.fmt.reduceabove[1])

    def set_bounds(self, color, **dims):
        self.logger.debug("Calculate bounds with %s", color)
        if isinstance(color, (str, unicode)):
            self.logger.debug("Found string --> Try color coding")
            if color == 'absolute':
                self.logger.debug(
                    "    Choose color coding by absolute speed")
                if not dims:
                    color = self.speed[:]
                else:
                    color = self.calc_speed(**dims)
            elif color == 'u':
                self.logger.debug(
                    "    Choose color coding by u component")
                if not dims:
                    color = self.udata[:]
                else:
                    color = self.get_data(var=self.u, vlst=None)
            elif color == 'v':
                self.logger.debug(
                    "    Choose color coding by v component")
                if not dims:
                    color = self.udata[:]
                else:
                    color = self.get_data(var=self.v, vlst=None)
            else:
                raise ValueError(
                    "Color must be in 'absolute', 'u' or 'v' to allow color "
                    "coding!")
        if not self.fmt._enablebounds:
            pass
        elif self.fmt.bounds[0] in ['rounded', 'sym', 'minmax',
                                        'roundedsym']:
            self.logger.debug('    Calculate bounds with %s',
                                str(self.fmt.bounds))
            self._bounds = returnbounds(color[:], self.fmt.bounds)
        else:
            self.logger.debug('   Found manually set bounds')
            self._bounds = self.fmt.bounds
        return color

    def calc_speed(self, **dims):
        """Calculate the speed from u and v component. If dims are not
        given, use self.data, otherwise get new data from reader with specified
        dimensions in self.dims and dims
        """
        if not dims:
            data = self.data
        else:
            data = self.get_data(**dims)
        return np.ma.power(
            np.ma.sum(data[:]*data[:], axis=0), 0.5)

    def make_plot(self):
        """Make the plot with the current settings and the WindPlot. Use it
        after reinitialization and _setupproj. Don't use this function to
        update the plot but rather the update function!"""
        if not self.fmt.enable:
            return
        plt.sca(self.ax)
        lat2d, lon2d = self.data.grid
        # configure WindPlot options
        if self._make_plot:
            self.logger.debug('Making plot...')
            plotops = self.fmt._windplotops.copy()
            # configure colormap options if possible (i.e. if plotops['color']
            # is an array)
            try:
                self.logger.debug("    Try to set up color coding")
                plotops['color'] = self.set_bounds(plotops['color'])
                if self.fmt.norm == 'bounds':
                    plotops['cmap'] = get_cmap(plotops['cmap'],
                                            len(self._bounds)-1)
                    plotops['norm'] = mpl.colors.BoundaryNorm(
                        self._bounds, plotops['cmap'].N)
                else:
                    plotops['cmap'] = get_cmap(plotops['cmap'])
                    plotops['norm'] = self.fmt.norm
                if self.fmt._enablebounds:
                    self._cmap = plotops['cmap']
                    self._norm = plotops['norm']
                    if self.fmt.opacity is not None:
                        self._calculate_opacity()
                else:
                    plotops['cmap'] = self._cmap
                    plotops['norm'] = self._norm
                if not self.fmt.streamplot:
                    args = (lon2d, lat2d, self.udata[:], self.vdata[:],
                            plotops.pop('color')[:])
                else:
                    args = (lon2d, lat2d, self.udata[:], self.vdata[:])
            except (TypeError, ValueError):
                self.logger.debug("    Failed --> no color coding.",
                                exc_info=True)
                self._bounds = None
                self._cmap = None
                self._norm = None
                args = (lon2d, lat2d, self.udata[:], self.vdata[:])
                if 'cmap' in plotops:
                    cmap = plotops.pop('cmap')

            if plotops.get('linewidth') is not None:
                self.logger.debug("    Calculate linewidth...")
                if isinstance(plotops['linewidth'], str):
                    if plotops['linewidth'] == 'absolute':
                        self.logger.debug(
                            "    Choose linewidths by absolute speed")
                        plotops['linewidth'] = self.speed.filled()/np.max(
                            self.speed)*self.fmt.scale
                elif plotops['linewidth'] == 'u':
                    self.logger.debug(
                            "    Choose linewidths by u component")
                    plotops['linewidth'] = \
                        self.udata.filled()/np.max(
                            np.abs(self.udata))*self.fmt.scale
                elif plotops['linewidth'] == 'v':
                    self.logger.debug(
                            "    Choose linewidths by v component")
                    plotops['linewidth'] = \
                        self.vdata.filled()/np.max(
                            np.abs(self.vdata))*self.fmt.scale
                if not self.fmt.streamplot:
                    plotops['linewidth'] = np.ravel(plotops['linewidth'])

            if self.fmt.streamplot:
                self._removeplot()
                try:
                    plotops.pop('scale')
                except KeyError:
                    pass
                try:
                    self.logger.debug("Plotting with Basemap.streamplot")
                    self.plot = self.mapproj.streamplot(*args, **plotops)
                except IndexError:
                    critical("Could normalize colors (probably due to an old "
                             "matplotlib version)!", logger=self.logger)
                    self.logger.debug("Failed. Removing norm key word",
                                      exc_info=1)
                    del plotops['norm']
                    self.plot = self.mapproj.streamplot(*args, **plotops)
            else:
                if hasattr(self, 'plot'):
                    self._removeplot()
                    self.logger.debug("Plotting with Basemap.quiver")
                    self.plot = self.mapproj.quiver(*args, **plotops)
                    # does currently not work with basemap
                    # self.plot.set_UVC(*args[2:])
                    # self.plot.set_linewidth(plotops['linewidth'])
                    # self.plot.set_rasterized(plotops['rasterized'])
                    # if self._cmap is not None:
                        # self.plot.set_cmap(self._cmap)
                    # if self._norm is not None:
                        # self.plot.set_norm(self._norm)
                    # if 'color' in plotops:
                        # self.plot.set_color(plotops['color'])
                else:
                    self.logger.debug("Plotting with Basemap.quiver")
                    self.plot = self.mapproj.quiver(*args, **plotops)
        if (self.fmt.plotcbar != ''
                or self.fmt.plotcbar != ()
                or self.fmt.plotcbar is not None
                or self.fmt.plotcbar is not False):
            if self._bounds is not None:
                self._draw_colorbar()

        self._configureaxes()
        self._make_plot = 0
        if self.fmt.tight:
            self.logger.debug("'tight' is True --> make tight_layout")
            plt.tight_layout()

    def update(self, todefault=False, plot=True, force=False, **kwargs):
        """Update the MapBase instance, formatoptions, variable, time and
        level.
        Possible key words (kwargs) are
          - all keywords as set by formatoptions (see initialization method
              __init__)
          - time: integer. Sets the time of the MapBase instance and reloads
              the data
          - level: integer. Sets the level of the MapBase instance and reloads
              the data
          - var: string. Sets the variable name of the MapBase instance
          - u: string. Sets the variable of the WindPlot instance and reloads
              the data
          - v: Sets the variable of the WindPlot instance and reloads the data
          - time, level and var as above
        """
        # set current axis
        plt.sca(self.ax)

        # check the keywords
        dims = self.dims.copy()
        dimsorig = self.dims_orig
        for dim in self.data.dims.dtype.fields.keys():
            if dim == self.u + '-' + self.v:
                # don't do anything for the first dimension u-v in DataField
                continue
            dims.setdefault(dim, 0)
            dimsorig.setdefault(dim, 0)
            dims.setdefault('u', self.u)
            dims.setdefault('v', self.v)
        possible_keys = self.fmt._default.keys() + dims.keys()
        for key in kwargs:
            self.fmt.check_key(key, possible_keys=possible_keys)

        # save current formatoptions
        old_fmt = self.fmt.asdict()

        # delete formatoptions which are already at the wished state
        if not todefault and not force:
            kwargs = {key: value for key, value in kwargs.items()
                      if value != old_fmt.get(
                          key, self.fmt._default.get(
                              key, dims.get(key)))}
        elif not force:
            self.logger.debug('Update to default...')
            oldkwargs = kwargs.copy()
            defaultitems = self.fmt._default.items()
            kwargs = {
                key: kwargs.get(key, value) for key, value in defaultitems
                if (key not in kwargs and np.all(
                    value != getattr(self.fmt, key)))
                or (key in kwargs and
                    np.all(kwargs[key] != getattr(self.fmt, key)))}
            if not self.fmt._enablebounds:
                kwargs = {key: value for key, value in kwargs.items()
                          if key not in ['cmap', 'bounds']}
            # update dimensions
            kwargs.update({
                key: oldkwargs.get(key, value)
                for key, value in dimsorig.items()
                if (dims[key] != dimsorig[key]
                    or (key in oldkwargs and oldkwargs[key] != dims[key]))})
        self.logger.debug("Update to ")
        try:
            max_kwarg_len = max(map(len, kwargs.keys())) + 3
        except ValueError:
            max_kwarg_len = 0
        for key, val in kwargs.items():
            self.logger.debug("    %s:".ljust(max_kwarg_len) + " %s", key, val)

        try:  # False: Remove plot, True: make plot, None: do nothing
            enable = bool(kwargs['enable'])
            self.logger.debug('Updating enable to %s', enable)
            self.fmt.enable = enable
        except KeyError:
            enable = None  # no change

        # handle streamplot changing
        if 'streamplot' in kwargs or enable is False:
            self.logger.debug(
                "Found 'streamplot' or 'enable' in kwargs --> remove plot")
            self._removeplot()
        reinit = self._reinitialize
        if 'streamplot' in kwargs:
            self.fmt.streamplot = kwargs['streamplot']
            if self.fmt.streamplot and self.density != 1.0:
                self.logger.debug(
                    "Set streamplot to True and changed density --> reinit")
                reinit = True

        # update plotting of cbar properties
        if 'plotcbar' in kwargs:
            self.logger.debug(
                "Found 'plotcbar' in kwargs --> remove non-used cbars")
            if kwargs['plotcbar'] in [False, None]:
                kwargs['plotcbar'] = ''
            if kwargs['plotcbar'] is True:
                kwargs['plotcbar'] = 'b'
            cbar2close = [cbar for cbar in self.fmt.plotcbar
                          if cbar not in kwargs['plotcbar']]
            self._removecbar(cbar2close)

        # update extend and ticklabels
        if ('extend' in kwargs
                or kwargs.get('ticklabels', False) is None):
            self.logger.debug(
                "Found 'extend' or 'ticklabels' in kwargs --> remove cbars")
            self._removecbar(resize=False)

        if 'scale' in kwargs:  # there is no update method for the scale
            self._removeplot()

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
            self.fmt.update(**bmprops)

        if not self.fmt.streamplot:
            reduceprops = {key: value for key, value in kwargs.items()
                           if key in ['density', 'reduceabove']}
            if reduceprops:
                self.fmt.update(**reduceprops)
        else:
            reduceprops = []


        # update mapobject dimensions and reinitialize
        newdims = {key: value for key, value in kwargs.items()
                   if key in dims.keys() and key != 'var'}
        if newdims:
            self.logger.debug(
            'Found new dimensions: %s. --> reinit',
            ', '.join(newdims.keys()))
        if newdims or bmprops or reduceprops or maskprops or enable or reinit:
            if bmprops or enable:
                self.mapproj = Basemap(**self.fmt._projops)
            self._reinit(**newdims)
            self._reinitialize = 0
            self._make_plot = 1
        if bmprops and plot:
            self._setupproj()

        self._make_plot = self._make_plot or set(kwargs) & set(
            ['cmap', 'bounds', 'rasterized'] + self.fmt.windonly_keys)

        # color oceans and land
        if any(key in kwargs for key in ['ocean_color', 'land_color']):
            self._draw_lsmask(kwargs.get('ocean_color', self.fmt.ocean_color),
                              kwargs.get('land_color', self.fmt.land_color))

        # update rest
        self.logger.debug('Update fmtBase instance')

        # set linewidth for streamplot manually to None
        if todefault and 'linewidth' in kwargs and self.fmt.streamplot:
            kwargs['linewidth'] = None

        self.fmt.update(**{key: value for key, value in kwargs.items()
                           if key not in self.fmt._bmprops + dims.keys()})

        # update ticks
        self._update_ticks(kwargs)

        # update map projection
        self._update_mapproj(kwargs=kwargs, plot=plot)

        if plot:
            self.make_plot()
        else:
            self._make_plot = 0

        self.logger.debug('Update Done.')

    def _moviedata(self, times, **kwargs):
        """generator to get the data for the movie"""
        kwargs = {key: iter(val) for key, val in kwargs.items()}
        for i, time in enumerate(times):
            # yield time, u, v, formatoptions
            yield (time, {key: next(value) for key, value in kwargs.items()})

    def _runmovie(self, args):
        """Function to update the movie with args from self._moviedata"""
        self.update(time=args[0], **args[-1])

    def _removeplot(self):
        """Removes the plot from the axes and deletes the plot property from
        the instance"""
        self.logger.debug("    Removing plot...")
        if hasattr(self, 'plot'):
            if self.fmt.streamplot:
                # remove lines
                try:
                    self.plot.lines.remove()
                except ValueError:
                    pass
                # remove arrows
                keep = lambda x: not isinstance(x, mpl.patches.FancyArrowPatch)
                self.ax.patches = [patch for patch in self.ax.patches
                                   if keep(patch)]
            else:
                try:
                    self.plot.remove()
                except:
                    pass
            del self.plot
        else:
            self.logger.debug("No plot to remove")
        self._removecbar()

    def _reduceuv(self, perc=0.5, pctl=0):
        """reduces resolution of u and v to perc of original resolution if the
        mean is larger than the value of the given percentile pctl"""
        # reset speed
        speed = self.speed
        try:
            perc = list(perc)
        except TypeError:
            perc = [perc]*2
        if pctl < 0. or pctl > 100.:
            raise ValueError("Percentiles must be between 0 and 100!")
        self.logger.debug(
            "Reduce u and v to %s of x and %s of y resolution "
            "above the %s percentile.", perc[0], perc[1], pctl)
        # compute step size in x and y direction
        step = np.ceil(np.array(np.shape(speed)) /
                       (np.array(np.shape(speed))*perc)).astype(int)
        self.logger.debug("    Stepsizes: %s", str(step))
        # compute half step size where the final data will be plotted
        halfstep0 = np.ceil(step/2.0).astype(int)
        self.logger.debug("    Half step: %s", halfstep0)
        if pctl == 0:
            pctl = np.min(speed)
        else:
            pctl = np.percentile(speed.compressed(), pctl)
        udata = self.udata
        vdata = self.vdata
        weights = self.weights[0]
        # loop through rows
        for i in xrange(0, len(speed), step[0]):
            # loop through columns
            for j in xrange(0, len(speed[i]), step[1]):
                # handle boundaries of data
                halfstep = [0, 0]
                if i+step[0] >= np.shape(speed)[0]:
                    stepx = np.shape(speed)[0]-i
                else:
                    stepx = step[0]
                    halfstep[0] = halfstep0[0]
                if j+step[1] >= np.shape(speed)[1]:
                    stepy = np.shape(speed)[1]-j
                else:
                    stepy = step[1]
                    halfstep[1] = halfstep0[1]
                if weights is not None:
                    w = weights[i:i+stepx, j:j+stepy]
                else:
                    w = None

                # compute means, but only if not all grid cells are masked
                if not all(np.ravel(speed.mask[i:i+stepx, j:j+stepy])):
                    # calculate weighted mean
                    if np.ma.average(speed[i:i+stepx, j:j+stepy],
                                  weights=w) >= pctl:
                        # reduce u
                        udata[i+halfstep[0], j+halfstep[1]] = np.ma.average(
                            udata[i:i+stepx, j:j+stepy], weights=w)
                        # mask all but the data containing the mean
                        mask = np.ma.make_mask(np.ones((stepx, stepy)))
                        mask[halfstep[0], halfstep[1]] = False
                        udata[:].mask[i:i+stepx, j:j+stepy] = mask

                        # reduce v
                        vdata[i+halfstep[0], j+halfstep[1]] = np.ma.average(
                            vdata[i:i+stepx, j:j+stepy], weights=w)
                        # mask all but the data containing the mean
                        mask = np.ma.make_mask(np.ones((stepx, stepy)))
                        mask[halfstep[0], halfstep[1]] = False
                        vdata[:].mask[i:i+stepx, j:j+stepy] = mask
        self.udata = udata
        self.vdata = vdata

    def asdict(self, **kwargs):
        """Returns a dictionary containing the current formatoptions and (if
        the time, level or name changed compared to the original
        initialization) the time, level or name"""
        fmt = super(WindPlot, self).asdict(**kwargs)
        try:
            del fmt['vlst']
        except KeyError:
            pass
        if self.u != self.dims_orig['vlst'][0]:
            fmt['u'] = self.u
        if self.v != self.dims_orig['vlst'][1]:
            fmt['v'] = self.v
        return fmt
    # ------------------ modify docstrings here --------------------------
    __init__.__doc__ += __init__.__doc__ % {
        'windfmt': '\n' + _get_fmtkeys_formatted('wind', 'windonly'),
        'mapbase': '\n' + MapBase.__init__.__doc__}
