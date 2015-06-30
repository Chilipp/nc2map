# -*- coding: utf-8 -*-
"""Module containing the LinePlot class

This class is intended to extract 1-dimensional data from a ArrayReaderBase
instance and visualize it as a line plot"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from numpy import array, linspace, unique
from numpy import all as npall
from collections import OrderedDict
from copy import copy, deepcopy
from itertools import izip, product, cycle, chain
from difflib import get_close_matches
from ..formatoptions import get_fmtdocs
from ..formatoptions import SimpleFmt
from ..warning import warn, critical
from ..defaults import texts
from .._cmap_ops import get_cmap
from _simple_plot import SimplePlot


class LinePlot(SimplePlot):
    """class to plot lineplots. See __init__ method for initialization
    keywords"""

    @property
    def colors(self):
        """Colors of the lines. There sorting correspond to the data
        in the data attribute. Set it with any iterable"""
        try:
            return self._colors
        except AttributeError:
            color_cycle = self._color_cycle
            if color_cycle is None:
                color_cycle = self.ax._get_lines.color_cycle
            else:
                try:
                    color_cycle = iter(get_cmap(color_cycle)(
                        linspace(0., 1., len(self.data), endpoint=True)))
                except (TypeError, KeyError):
                    color_cycle = iter(color_cycle)
            # store color_cycle
            self._colors = [next(color_cycle) for _ in self.data]
            self._color_cycle = color_cycle
        return self._colors

    @colors.setter
    def colors(self, color_cycle):
        self._color_cycle = color_cycle
        try:
            del self._colors
        except AttributeError:
            pass

    @colors.deleter
    def colors(self):
        del self._colors

    @property
    def meta(self):
        """Dictionary containing meta information of the LinePlot
        instance"""
        meta = self.get_label_dict(delimiter=', ')
        meta.update(self._meta)
        #for name, name_dict in self.names.items():
            #meta.update({
                #"%s_%s" % (name, key): val
                #for key, val in name_dict.items()})
        dim_meta = self.get_label_dict(
            vlst=[data.dimensions[0] for data in self.data],
            delimiter=', ')
        meta.update({'dim_'+key: val for key, val in dim_meta.items()})
        var_meta = self.get_label_dict(
            vlst=(data._DataField__var for data in self.data),
            delimiter=', ')
        meta.update(var_meta)
        meta['name'] = self.name
        return meta

    @property
    def vlst(self):
        return list(frozenset(
            val['var'] for val in self.names.itervalues()))

    def __init__(self, reader, names=None, vlst=None, name='line',
                 ax=None, fmt={}, pyplotfmt={}, sort=None, mapsin=None,
                 color_cycle=None, independent='x', meta={}, **dims):
        """Initialization method of LinePlot class

        Input:
          - reader: ArrayReader instance
          - names: string, list of strings or dictionary
              Strings may include {0} which will be replaced by a counter.
              -- if string: same as list of strings (see below)
              -- list of strings: this will be used for the name setting of
                  each line. If 'label' is not explicitly set in pyplotfmt,
                  this will also set the label of the line.
                  The final number of lines depend in this case on vlst and
                  the specified **dims (see below)
              -- dictionary: {'name-of-line1': {'dim1-for-line1': value,
                                                 'dim2-for-line1': value, ...},
                              'name-of-line2': {'dim1-for-line2': value,
                                                 'dim2-for-line2': value, ...},
                              ...}
                  Define the dimensions for each line directly.
                  Which dimensions are possible depend on the variable in
                  the reader.
                  Hint: With a standard dictionary it is not possible to
                  specify the order. Use the collections.OrderedDict class
                  for that.
          - vlst: string or list of strings. Variables to plot (does only
              have an effect if names is not a dictionary)
          - name: string. Name of the LinePlot instance
          - ax: mpl.axes.SubplotBase instance to plot on. If None, a new figure
              and axes will be created
          - fmt: Dictionary containing formatoption keywords. Possible
              formatoption keywords are given below
          - pyplotfmt: Dictionary containing additional keywords for the
              plt.plot function which is used to plot each line. The
              dictionary may also contain line names (see names above) to
              specify the options for the specific line directly.
              E.g {'line0': {'color': 'k'}} will set the color for the line
              with name 'line0' to black.
              {'color': 'b', 'line0': {'color': 'k'}} will set the color for
              all but 'line0' to blue.
          - sort: None or list of dimension strings. If names is not a
              dictionary, this defines how the lines are ordered, dependent on
              the iterable dimensions in **dims and vlst (e.g.
              sort=['time', 'var']). If sort is None, it will first sorted by
              the variable and then alphabetically by the iterable dimesion.
          - mapsin: nc2map.MapsManager instance the LinePlot instance belongs
              to
          - color_cycle: Any iterable containing color definitions or a
              registered colormap suitable for maplotlib.pyplot.get_cmap method
          - independent: 'x' or 'y'. Specifies which is the independent axis,
              i.e. on which axis to plot the dimension.
        Any other keyword argument may specify the dimensions to take from the
        reader. If the dimension values are iterables, each combination of the
        iterable dimensions defines one line.

        Example:
          Assume a 3 dimensional ('time', 'lat', 'lon') temperature field
          stored in the variable 't2m'. To plot the temperature for all
          longitudes for the first time step and 3rd latitude index, use
            >>> LinePlot(myreader, vlst='t2m', time=0, lat=3)
          which is equivalent to
            >>> LinePlot(myreader, vlst='t2m', time=0, lat=3, lon=slice(None))
          On the other hand
            >>> LinePlot(myreader, vlst='t2m', time=[0, 1], lat=3)
          will create 2 lines, one for the first and one for the second
          timestep.
          However, if you only want to visualize parts of the longitudes,
          setting
            >>> LinePlot(myreader, vlst='t2m', time=[0, 1], lat=3,
                         lon=[1, 2, 3, 4, 5])
          would result in an error because it will be iterated over
          [1, 2, 3, 4, 5]. Here is how you can fix it
            >>> from collections import cycle
            >>> LinePlot(myreader, vlst='t2m', time=[0, 1], lat=3,
                         lon=cycle([[1, 2, 3, 4, 5]]))

        Now comes a list of possible formatoption keywords for fmt
        """
        # docstring is extended below
        self.name = name
        self.set_logger()

        kwargs_keys = ['name', 'names', 'vlst', 'ax', 'fmt',
                       'pyplotfmt', 'sort', 'color_cycle', 'independent']
        max_kwarg_len = max(map(len, kwargs_keys + dims.keys())) + 2
        self.logger.debug("Original kwargs:")
        for key in kwargs_keys:
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              key, str(locals()[key]))
        for dim, val in dims.items():
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              dim, str(val))

        super(LinePlot, self).__init__(name=name, mapsin=mapsin, ax=ax,
                                       fmt=fmt, meta=meta)
        self.reader = reader
        self.names = self._setupnames(names=names, vlst=vlst, sort=sort,
                                      **dims)
        self.names_from_orig = deepcopy(self.names)
        self.plotops = self._setupfmt_pyplot(pyplotfmt)
        self.data = self.get_data()
        self.lines = {}
        self.colors = color_cycle
        self.independent_axis = str(independent).lower()
        # check dims and raise error if something is wrong
        for data in self.data:
            data.check_dims(raise_error=True)

    def make_plot(self):
        """Method that makes the plot"""
        # return if not enabled
        if not self.fmt.enable:
            return
        ax = self.ax
        plt.sca(self.ax)

        plt.grid(self.fmt.grid)

        all_dims = (data.dims[data.dimensions[0]] for data in self.data)
        times = (data.time for data in self.data)
        if self.independent_axis == 'y':
            iter_data = izip(self.names.iterkeys(), self.colors,
                             all_dims, self.data, times)
        elif self.independent_axis == 'x':
            iter_data = izip(self.names.iterkeys(), self.colors,
                             self.data, all_dims, times)
        else:
            raise ValueError("Wrong value %s for independent axis!")
        for name, color, ydata, xdata, time in iter_data:
            ydata = ydata.tolist()
            xdata = xdata.tolist()
            plotops = self.plotops[name].copy()
            plotops.setdefault('color', color)
            try:
                yfilldata = plotops.pop('fill')
                assert yfilldata is not None, "yfilldata must not be None!"
                try:
                    yfilldata = array(yfilldata, dtype=float)
                except ValueError:
                    try:
                        yfilldata = self.get_data(var=str(yfilldata),
                            names={name: self.names[name].copy()})[0]
                    except KeyError:
                        try:
                            dims = self.names[name].copy()
                            del dims['var']
                            dims['vlst'] = yfilldata
                            yfilldata = self.get_data(names={name: dims})[0][:]
                        except:
                            warn("Could not determine yfilldata! It has either"
                                 " to be an array, one (or two) variable "
                                 "strings in the reader!")
                            raise
            except (KeyError, AssertionError):
                yfilldata = None

            yfill_kwargs = {}
            for key, val in plotops.items():
                if key.startswith('fill'):
                    yfill_kwargs[key[4:]] = plotops.pop(key)
            if yfilldata is not None and yfilldata.ndim == 1:
                yfilldata = np.tile(yfilldata, [2, 1])
                yfilldata = array([ydata[:] - yfilldata[0,:],
                                   ydata[:] + yfilldata[1,:]])

            # x fill data
            try:
                xfilldata = plotops.pop('xfill')
                assert xfilldata is not None, "xfilldata must not be None!"
                try:
                    xfilldata = array(xfilldata, dtype=float)
                except ValueError:
                    try:
                        xfilldata = self.get_data(var=str(xfilldata),
                            names={name: self.names[name].copy()})[0][:]
                    except KeyError:
                        try:
                            dims = self.names[name].copy()
                            del dims['var']
                            dims['vlst'] = xfilldata
                            xfilldata = self.get_data(names={name: dims})[0][:]
                        except:
                            warn("Could not determine xfilldata! It has either"
                                 " to be an array, one (or two) variable "
                                 "strings in the reader!")
                            raise
            except (KeyError, AssertionError):
                xfilldata = None
            xfill_kwargs = {}
            for key, val in plotops.items():
                if key.startswith('xfill'):
                    xfill_kwargs[key[4:]] = plotops.pop(key)
            if xfilldata is not None and xfilldata.ndim == 1:
                xfilldata = array([xdata[:] - xfilldata[:],
                                   xdata[:] + xfilldata[:]])
            plotops.setdefault('label', name)
            if time is not None:
                try:
                    plotops['label'] = time.tolist().strftime(
                        plotops['label']).decode('utf-8')
                except AttributeError:
                    pass
            try:
                plotops['label'] = plotops['label'] % self.names[name]
            except KeyError:
                self.logger.debug(
                    "Could not change the label due.", exc_info=True)
                pass
            self.logger.debug('Plotting line %s', name)
            self.lines[name] = ax.plot(
                xdata[:], ydata[:], **plotops)
            lcolor = self.lines[name][0].get_color()
            if yfilldata is not None:
                yfill_kwargs.setdefault('color', lcolor)
                yfill_kwargs.setdefault('alpha', 0.5)
                self.lines[name].append(ax.fill_between(
                    xdata[:], yfilldata[0], yfilldata[1],
                    **yfill_kwargs))
            if xfilldata is not None:
                xfill_kwargs.setdefault('color', lcolor)
                xfill_kwargs.setdefault('alpha', 0.5)
                self.lines[name].append(ax.fill_betweenx(
                    ydata[:], xfilldata[0], xfilldata[1],
                    **xfill_kwargs))
        # set xlim
        if self.fmt.xlim is not None:
            plt.xlim(*self.fmt.xlim)
        else:
            self.ax.autoscale(axis='x')
        # set ylim
        if self.fmt.ylim is not None:
            plt.ylim(*self.fmt.ylim)
        else:
            self.ax.autoscale(axis='y')

        if self.fmt.legend is not None:
            self.logger.debug("Draw legend with %s", self.fmt.legend)
            self.legend = plt.legend(**self.fmt.legend)

        self._configureaxes()

    def update(self, names=None, color_cycle=None, fmt={}, pyplotfmt={},
               dims={}, todefault=False, **kwargs):
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
        Further keywords may be any formatoption keyword from the SimpleFmt
        class.
        Note: To add new lines, use the addline method instead.
        """
        for line in self.lines.itervalues():
            for l in line:
                try:
                    l.remove()
                except ValueError:
                    pass
        self.lines = {}
        if hasattr(self, 'legend'):
            self.legend.remove()
            del self.legend
        if names is None:
            names = self.names.copy()
        elif not isinstance(names, dict):
            names = {name: val for name, val in self.names.items()
                      if name in names}
        if color_cycle is not None:
            self.colors = color_cycle
        dims = deepcopy(dims)
        fmt = deepcopy(fmt)
        pyplotfmt = deepcopy(pyplotfmt)
        for key, val in kwargs.items():
            if key in next(names.itervalues()).keys():
                dims[key] = val
            elif key in self.fmt._default.keys():
                fmt[key] = val
            else:
                pyplotfmt[key] = val
        if todefault:
            for key, val in self.fmt._default.items():
                if getattr(self.fmt, '_'+key) != val:
                    fmt.setdefault(key, val)
        self.fmt.update(**fmt)
        for name in names:
            if todefault:
                this_dims = self.names_from_orig[name]
                this_dims.update(dims)
            else:
                this_dims = dims
            self.names[name].update(this_dims)
        for key, val in self._setupfmt_pyplot(pyplotfmt, names).items():
            if todefault:
                self.plotops[key] = val
            else:
                self.plotops[key].update(val)
        self.data = self.get_data()
        if self.fmt.enable:
            self.make_plot()

    def _setupnames(self, names, vlst, sort=None, **dims):
        """Sets up the names dictionary for the plot"""
        if isinstance(names, dict):
            return OrderedDict(names.items())
        if isinstance(names, (str, unicode)):
            names = [names]
        if isinstance(vlst, (str, unicode)):
            vlst = [vlst]
        iter_dims = OrderedDict()
        for key, val in sorted(dims.items()):
            # try if iterable
            try:
                iter(val)
                iter_dims[key] = dims.pop(key)
            except TypeError:
                pass
        if vlst is None:
            raise ValueError(
                "vlst must not be None if names is not a dictionary!")
        if sort is None:
            zipped_keys = ['var'] + iter_dims.keys()
            zipped_dims = product(vlst, *iter_dims.itervalues())
        else:
            zipped_keys = list(sort)
            if not set(zipped_keys) == set(iter_dims.keys() + ['var']):
                raise ValueError(
                    "Sorting parameter (%s) differs from iterable "
                    "dimensions (%s)!" % (
                        ', '.join(zipped_keys),
                        ', '.join(iter_dims.keys() + ['var'])))
            zipped_dims = product(*[
                vlst if key == 'var' else iter_dims[key]
                for key in sort])
        if names is None:
            names = OrderedDict([
                ('line%i' % i, {
                    dim: val for dim, val in zip(zipped_keys, dimstuple)})
                for i, dimstuple in enumerate(zipped_dims)])
        else:  # assume a list of strings
            names = OrderedDict([
                (str(name).format(i), {
                    dim: val for dim, val in zip(zipped_keys, dimstuple)})
                for i, (name, dimstuple) in enumerate(izip(cycle(names),
                                                           zipped_dims))])
        # update for non-iterable dimensions
        for settings in names.itervalues():
            for dim, val in dims.items():
                settings[dim] = val
        return names

    def _setupfmt_pyplot(self, fmt, names=None):
        if names is None:
            names = self.names
        plotops = OrderedDict()
        for name in names.iterkeys():
            plotops[name] = {}
            for key, val in fmt.get(name, {}).items():
                plotops[name].setdefault(key, val)
            for key, val in {key: val for key, val in fmt.items()
                             if key not in names}.items():
                plotops[name].setdefault(key, val)
        return plotops

    def get_data(self, var=None, datashape='1d', time=None, level=None,
                 names=None, **furtherdims):
        """Reads the specified variable, longitude and lattitude from netCDF
        file and shifts it.
        Input:
            - var: Variable name to read for in the netCDF file
            - time: time index in netCDF file (if None, use self.time)
            - level: level index in netCDF file (if None, use self.level)
        """
        if names is None:
            names = self.names
        data = [0]*len(names)  # final output data for all lines
        for i, (name, dims) in enumerate(names.items()):
            dims.update(furtherdims)
            if var is not None:
                dims['var'] = var
                try:
                    del dims['vlst']
                except KeyError:
                    pass
            data[i] = self.reader.get_data(datashape=datashape, **dims)
        return data

    def addline(self, reader, names=None, vlst=None, pyplotfmt={}, sort=None,
                color_cycle=None, **dims):
        new_names = self._setupnames(names=names, vlst=vlst, sort=sort,
                                           **dims)
        self.names.update(new_names)
        self.plotops.update(self._setupfmt_pyplot(pyplotfmt))
        self.data += self.get_data()
        if color_cycle is None:
            try:
                color_cycle = self._color_cycle
                self._colors += [next(color_cycle) for _ in new_names]
            except StopIteration:
                color_cycle = cycle(self.colors)
                self._colors += [next(color_cycle) for _ in new_names]
        else:
            self.colors = color_cycle
        self.update()

    def show(self):
        plt.show(block=False)

    def asdict(self):
        """Returns the settings of the LinePlot instance
        Output: names, plotops, fmt
          - names dictionary: Contains the dimension settings (for each line
              in the LinePlot)
          - pyplot settings: Contains the settings for the plot (for each line
              in the LinePlot)
          - fmt dictionary: Contains the SimpleFmt instance (i.e. the
              formatoptions) as dictionary
        """
        return {'names': self.names, 'pyplotfmt': self.plotops,
                'fmt': self.fmt.asdict()}

    def get_label_dict(self, vlst=None, delimiter=None):
        """Returns a dictionary for the given variables

        Input:
          - vlst: List of variables in the reader
          - delimiter: string which shall be used for separating values in a
            string, if more than one MapBase instance is found. If not given
            (or None), lists will be returned, not strings
        """
        if isinstance(vlst, (str, unicode)):
            vlst = [vlst]
        if vlst is None:
            meta_list = self.names.values()
        else:
            vlst = list(vlst)
            meta_list = [self.reader.get_meta(var=var).copy() for var in vlst]
            for meta, var in zip(meta_list, vlst):
                meta['var'] = var
        meta_keys = set(chain(*(meta for meta in meta_list)))
        meta = {}
        for key in meta_keys:
            try:
                meta[key] = frozenset(
                    meta_dict.get(key) for meta_dict in meta_list) - {None}
            except TypeError:  # in case of unhashable objects (e.g. lists)
                try:
                    meta[key] = set(unique(
                        meta_dict.get(key) for meta_dict in meta_list)) - \
                            {None}
                except TypeError:
                    pass

        if delimiter is not None:
            for key, val in meta.items():
                meta[key] = delimiter.join(map(str, val))
        delimiter = delimiter or ', '
        meta.setdefault('all', 'Lines %s of %s' % (
            self.name, delimiter.join(self.names)))
        return meta

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
        for data in self.data:
            if data.time is not None:
                # use strftime and decode with utf-8 (in case of accented
                # string)
                try:
                    txt = data.time.tolist().strftime(txt).decode('utf-8')
                    break
                except AttributeError:
                    pass
        txt = super(LinePlot, self)._replace(txt=txt, fig=fig,
                                             delimiter=delimiter)
        return txt

    def extract_in_reader(self, full_data=False):
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
        if not full_data:
            readers = [self.reader.extract(**dims)
                       for name, dims in self.names.items()]
            if len(readers) > 1:
                reader = readers[0].merge(*readers[1:])
                for reader in readers:
                    reader.close()
            else:
                reader = readers[0]
        else:
            reader = self.reader.selname(*self.vlst)
        return reader

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
        if not num % 2:
            for line in self.lines.itervalues():
                for l in line:
                    try:
                        l.remove()
                    except ValueError:
                        pass
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

    def __str__(self):
        return repr(self)[1:-1]

    def __repr__(self):
        line_str = ', '.join(self.names.keys())
        return "<nc2map.mapos.%s %s of %s>" % (
            self.__class__.__name__, self.name, line_str)

    # ------------------ modify docstrings here --------------------------
    __init__.__doc__ += '\n'.join((key+':').ljust(20) + get_fmtdocs('xy')[key]
                                  for key in sorted(get_fmtdocs('xy').keys()))
