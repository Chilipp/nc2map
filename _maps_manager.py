# -*- coding: utf-8 -*-
"""Module containing the nc2map MapsManager class

This module contains the basic class for controlling multiple MapBase istances.
This classed is used for example by the nc2map.Maps class, nc2map.CbarManager
and nc2map.EvaluatorBase (and subclasses)"""
from __future__ import division
import glob
import six
from copy import deepcopy
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import izip, chain, permutations, product, repeat, cycle
from collections import OrderedDict
from .warning import warn, critical
from mapos import (MapBase, FieldPlot, WindPlot, returnbounds, round_to_05,
                   SimplePlot, LinePlot)
from mapos._mapproperties import MapProperties
import readers
from formatoptions import _get_fmtkeys_formatted
from _axes_wrapper import multiple_subplots

_props = MapProperties()


class MapsManager(object):
    """Base class for the control of multiple maps"""
    # ------------------ define properties here -----------------------

    # mapdata dictionary property
    maps = _props.default('maps', """list containing MapBase instances""")
    lines = _props.default('lines', """list containing LinePlot instances""")
    logger = _props.default('logger', """nc2map logger instance""")

    @property
    def meta(self):
        """Meta informations stored in the MapBase instances"""
        objects = self.maps + self.lines
        meta = {}
        for key in set(chain(*(mapo.meta.keys() for mapo in objects))):
            try:
                meta[key] = frozenset(
                    mapo.meta.get(key) for mapo in objects) - {None}
            except TypeError:  # in case of unhashable objects (e.g. lists)
                try:
                    meta[key] = set(np.unique(
                        mapo.meta.get(key) for mapo in objects)) - {None}
                except:
                    pass
        return meta

    @property
    def vlst(self):
        """List of variables"""
        return list(frozenset(mapo.var for mapo in self.maps))

    @property
    def name_dicts(self):
        """List of tuples (name, var, time, level), where each tuple
        corresponds to one MapBase instance"""
        return OrderedDict([(mapo.name, mapo.dims) for mapo in self.maps])

    @property
    def names(self):
        """List of the names of the object in self.maps and self.lines"""
        return self.mapnames + self.linenames

    @property
    def linenames(self):
        """List of the names of the object in self.lines"""
        return [line.name for line in self.lines]

    @property
    def mapnames(self):
        """List of the names of the object in self.maps"""
        return [mapo.name for mapo in self.maps]

    @property
    def times(self):
        """List of time steps"""
        return list(frozenset(mapo.time for mapo in self.maps))

    @property
    def levels(self):
        """List of levels"""
        return list(frozenset(mapo.level for mapo in self.maps))

    def __init__(self):
        """init function just sets some attributes"""
        self.set_logger()
        self.windonly = False
        self.plot = True
        self.maps = []
        self.lines = []
        # old fmts (for function undo)
        self._fmt = []
        # future fmts (for function redo)
        self._newfmt = []

    def make_plot(self, *args, **kwargs):
        """makes the plot of MapBase instances.
        Don't use this function but rather the update function to make plots"""
        for mapo in self.get_maps(*args, **kwargs):
            mapo.make_plot()

    def set_reader(self, mode=None, *args, **kwargs):
        """Method to set the reader of the MapsManager instance.

        Input:
          - mode: string (one of %(readers)s) or None.
              If None, it is tried to set the reader with the one that matches
              (but not with ArrayReader).
          Other arguments and keyword arguments are passed directly to the
          reader initialization."""
        # docstring is extended below
        self.logger.debug("Set reader with mode %s", mode)
        if mode is None:
            myreaders = list(readers.readers)
            myreaders.remove('ArrayReader')
            self.reader = readers.auto_set_reader(readers=myreaders, *args,
                                                  **kwargs)
        else:
            self.reader = vars(readers)[mode](*args, **kwargs)

    def get_readers(self, *args, **kwargs):
        """Get the open readers of the maps and lines

        This method can be used to get the readers
        (:class:`nc2map.readers.ReaderBase` instances) from the
        :attr:`~nc2map.MapsManager.maps` and :attr:`~nc2map.MapsManager.lines`
        attributes by there meta information

        Parameters
        ----------
        *args
            May be 'wind' to use the :meth:`~nc2map.MapsManager.get_winds`
            method instead of the :meth:`~nc2map.MapsManager.get_maps` method
            or you can give the objects to consider directly.
            Objects may be :class:`~nc2map.mapos.MapBase` instances, or
            :class:`nc2map.readers.ReaderBase` instances.
            In that case, the other parameters (``**kwargs``) to not matter.
        **kwargs
            everything from the :meth:`~nc2map.MapsManager.get_maps`

        Returns
        -------
        dict
            keys are the :class:`nc2map.readers.ReaderBase` instances of the
            specified mapos, values are lists of those mapos that use this
            reader."""
        if 'wind' in args:
            get_func = self.get_winds
            args = tuple(arg for arg in args if arg != 'wind')
        else:
            get_func = self.get_maps
        if args == ():
            maps = get_func(**kwargs)
            out = OrderedDict()
            append = True
        elif all(isinstance(arg, MapBase) for arg in args):
            maps = args
            out = OrderedDict()
            append = True
        elif all(isinstance(arg, readers.ReaderBase) for arg in args):
            out = OrderedDict([(arg, []) for arg in args])
            maps = get_func()
            append = False
        else:
            raise TypeError(
                "Wrong type of obj! Object must either be 'maps' or 'winds'!")
        for mapo in maps:
            if mapo.reader not in out and append:
                out[mapo.reader] = []
            if mapo.reader in out:
                out[mapo.reader].append(mapo)
        return out

    def get_maps(self, mode='both', maps=[], _meta={}, **kwargs):
        """Get maps and lines

        This method can be used to get maps and lines from the
        :attr:`~nc2map.MapsManager.maps` and :attr:`~nc2map.MapsManager.lines`
        attributes by there meta information

        Parameters
        ----------
        mode: {'maps', 'lines', 'both'}
            If 'maps' or 'lines', only the self.maps or self.lines is used
        maps: list of :class:`~nc2map.mapos.MapBase` or
            :class:`~nc2map.mapos.SimplePlot` instances that shall be
            considered. If empty, the self.maps and self.lines attributes
            are used
        _meta: dict
            Keys must be keys in the :attr:`nc2map.mapos.MapBase.meta`
            instances and values lists of what you want to use for
            selection. You can use this keyword for accessing meta
            attributes that make trouble with the python syntax (e.g.
            something like 'long-name')
        **kwargs
            any key from the meta attribute which can be used to identify the
            object (e.g. 'var' for the variable, 'time' for time, 'long_name'
            for longnames, etc.) that are used for filtering

        Returns
        -------
        list of objects
            Either MapBase or LinePlot instances, or both (depending on mode).
            The order depends on their index in the specific attribute (see
            :attr:`~nc2map.MapsManager.maps` and
            :attr:`~nc2map.MapsManager.lines`). However, first come maps, then
            lines.

        Examples
        --------
        Return all maps for variable 't2m'::

            mymaps.get_maps(var='t2m')"""
        try:
            maps = list(maps)
        except TypeError:
            maps = [maps]
        if maps:
            all_maps = maps
        elif mode in ['maps', 'both']:
            all_maps = self.maps[:]
        else:
            all_maps = []

        if not maps and mode in ['lines', 'both']:
            all_maps += self.lines
        kwargs.update(_meta)
        if not kwargs:
            return all_maps
        meta = self.meta
        possible_keys = meta.keys()
        wrong_keys = [key for key in kwargs if key not in possible_keys]
        if wrong_keys:
            raise KeyError(
                "Wrong dimension identifiers %s! Possible identifiers are "
                "%s." % (', '.join(wrong_keys), ', '.join(possible_keys)))
        for key, val in kwargs.items():
            if isinstance(val, (str, unicode)):
                val = {val}
            try:
                kwargs[key] = set(val)
            except TypeError:
                kwargs[key] = {val}
        maps = [mapo for mapo in all_maps
                if all(mapo.meta.get(key) in kwargs[key]
                        for key in kwargs)]
        return maps

    def get_winds(self, *args, **kwargs):
        """Get the wind plots

        This method can be used to get winds (:class:`~nc2map.mapos.WindPlot`)
        from the :attr:`~nc2map.mapos.FieldPlot.wind` attribute.

        ``*args`` and ``**kwargs`` are determined by the
        :meth:`~nc2map.MapsManager.get_maps` method.

        Returns
        -------
        list of :class:`~nc2map.mapos.WindPlot` instances

        Notes
        -----
        An empty list is returned when self.windonly is True.

        See Also
        --------
        nc2map.MapsManager.get_maps: Basic method for mapo selection"""
        if not self.windonly:
            kwargs['mode'] = 'maps'
            return [mapo.wind for mapo in self.get_maps(*args, **kwargs)
                    if hasattr(mapo, 'wind') and mapo.wind is not None]
        else:
            return []

    def get_figs(self, *args, **kwargs):
        """Get the open figures of the maps and lines

        This method can be used to get the matplotlib figures
        from the maps and lines in the :attr:`~nc2map.MapsManager.maps` and
        :attr:`~nc2map.MapsManager.lines` attributes by there meta information

        Parameters
        ----------
        *args
            May be 'wind' to use the :meth:`~nc2map.MapsManager.get_winds`
            method instead of the :meth:`~nc2map.MapsManager.get_maps` method
            or you can give the objects to consider directly.
            Objects may be :class:`~nc2map.mapos.MapBase` instances, or
            matplotlib.figure.Figure instances.
            In that case, the other parameters (``**kwargs``) to not matter.
        **kwargs
            everything from the :meth:`~nc2map.MapsManager.get_maps`

        Returns
        -------
        dict
            keys are the matplotlib.figure.Figure instances of the
            specified mapos, values are lists of those mapos that plot on
            this figure."""
        if 'wind' in args:
            get_func = self.get_winds
            args = tuple(arg for arg in args if arg != 'wind')
        else:
            get_func = self.get_maps
        if args == ():
            maps = get_func(**kwargs)
            figs = OrderedDict()
            append = True
        elif all(isinstance(arg, (MapBase, SimplePlot)) for arg in args):
            maps = args
            figs = OrderedDict()
            append = True
        elif all(isinstance(arg, mpl.figure.Figure) for arg in args):
            figs = OrderedDict([(arg, []) for arg in args])
            maps = get_func()
            append = False
        else:
            raise TypeError(
                "Wrong type of obj! Object must either be 'maps' or 'winds'!")
        for mapo in maps:
            if mapo.ax.get_figure() not in figs and append:
                figs[mapo.ax.get_figure()] = []
            if mapo.ax.get_figure() in figs:
                figs[mapo.ax.get_figure()].append(mapo)
        return figs

    def _replace(self, txt, *args, **kwargs):
        """Function to replace strings by objects from fig

        Parameters
        ----------
        txt: str
            '%(key)s' will be replaced by the value of the 'key' information in
            the MapBase.meta attribute (e.g. '%(long_name)s' will be replaced
            by the long_name of the variable file in the NetCDF file.

        Other ``*args`` and ``**kwargs`` are determined by the
        :class:`~nc2map.MapsManager.get_label_dict` method

        Returns
        -------
        str
            txt with inserted meta information
        """
        meta = self.get_label_dict(*args, **kwargs)
        return txt % meta

    def get_names(self, *args, **kwargs):
        """Get the unique names of the maps and lines

        ``*args`` and ``**kwargs`` are determined by the
        :meth:`~nc2map.MapsManager.get_maps` method

        Returns
        -------
        list of strings
            names in the specified input."""
        return [mapo.name for mapo in self.get_maps(*args, **kwargs)]

    def get_meta(self, *args, **kwargs):
        """Get the meta information for the MapBase instances

        ``*args`` and ``**kwargs`` are determined by the
        :meth:`~nc2map.MapsManager.get_maps` method

        Returns
        -------
        list of dictionaries
            The dictionaries are the :attr:`~nc2map.mapos.MapBase.meta`
            attribute of the specified input.

        See Also
        --------
        nc2map.MapsManager.get_label_dict: Concatenates the dictionaries
        nc2map.MapsManager.meta: Contains all meta information"""
        return [mapo.meta for mapo in self.get_maps(*args, **kwargs)]

    def nextt(self, *args, **kwargs):
        """Update MapBase instances to the next timestep

        All specified :class:`~nc2map.mapos.MapBase` instances are updated
        to the next time step.

        ``*args`` and ``**kwargs`` are determined by the
        :meth:`~nc2map.MapsManager.get_maps` method

        Notes
        -----
        If the time is the last time step in the reader, the very first
        one is used.

        By default, the 'mode' keyword is set to 'maps' for the get_maps
        method.

        See Also
        --------
        nc2map.MapsManager.prevt: Method to update to the previous time step
        nc2map.mapos.MapBase.time: Time attribute of the MapBase class"""
        kwargs.setdefault('mode', 'maps')
        if 'wind' in args:
            maps = self.get_winds(*(arg for arg in args if arg != 'wind'),
                                  **kwargs)
        else:
            maps = self.get_maps(*args, **kwargs)
        for mapo in maps:
            time = mapo.reader.get_time_slice(mapo.time)
            try:
                mapo.update(time=time + 1, plot=self.plot)
            except IndexError as e:
                warn(e.message + ". I show the first timestep for %s." % (
                     mapo.name))
                self.logger.debug("Failed.", exc_info=True)
                mapo.update(time=0, plot=self.plot)
        if self.plot:
            for fig in self.get_figs(*maps):
                plt.figure(fig.number)
                plt.draw()

    def prevt(self, *args, **kwargs):
        """Update MapBase instances to the previous timestep

        All specified :class:`~nc2map.mapos.MapBase` instances are updated
        to the previous time step.

        ``*args`` and ``**kwargs`` are determined by the
        :meth:`~nc2map.MapsManager.get_maps` method

        Notes
        -----
        If the time is currently the first time step in the reader, the very
        last one is used.

        By default, the 'mode' keyword is set to 'maps' for the get_maps
        method.

        See Also
        --------
        nc2map.MapsManager.nextt: Method to update to the next time step
        nc2map.mapos.MapBase.time: Time attribute of the MapBase class"""
        kwargs.setdefault('mode', 'maps')
        if 'wind' in args:
            maps = self.get_winds(
                *(arg for arg in args if arg != 'wind'), **kwargs)
        else:
            maps = self.get_maps(*args, **kwargs)
        for mapo in maps:
            time = mapo.reader.get_time_slice(mapo.time)
            try:
                mapo.update(time=time - 1, plot=self.plot)
            except IndexError as e:
                warn(e.message + ". I show the last timestep for %s." % (
                    mapo.name))
                self.logger.debug("Failed.", exc_info=True)
                mapo.update(time=-1, plot=self.plot)
        if self.plot:
            for fig in self.get_figs(*maps):
                plt.figure(fig.number)
                plt.draw()

    def show(self):
        """shows all open figures (without blocking)"""
        plt.show(block=False)

    def close(self, num=0, remove=True, **kwargs):
        """Close the plots and data

        This method closes the instance and deletes the data out of the memory.
        Plots may be specified by ``**kwargs``, specified by the the
        :meth:`~nc2map.MapsManager.get_maps` method.

        Parameters
        ----------
        num: int
            Product of the numbers 2, 3, 5 and 7 or 0 (to consider all).

                * 2: remove the plot
                * 3: close the figure
                * 5: delete the data out of memory
                * 7: close the reader
        remove: bool
            If True, mapos are removed from the maps attribute or lines
            attribute

        Examples
        --------
        Close all figures and :class:`nc2map.mapos.MapBase` instances, but
        not the readers::

            mymaps.close(15)
        """
        if not self.maps and not self.lines:
            return
        num = num or 210
        if not num % 7:
            close_reader = True
            num /= 7
        else:
            close_reader = False

        for mapo in self.get_maps(**kwargs):
            mapo.close(num)
            if remove:
                try:
                    self.maps.remove(mapo)
                except ValueError:
                    self.lines.remove(mapo)
        if close_reader:
            for reader, maps in self.get_readers(**kwargs).items():
                if len(maps) == 0:
                    mapo.reader.close()

    def _make_subplots(self, ax, n=1, *args, **kwargs):
        """Function to create subplots.

    Different from the :func:`~nc2map.subplots` function, this method creates
    multiple figures with the given shapes.

    Parameters
    ----------
    ax: tuple (x,y[,z]), subplot or list of subplots
        Default: (1, 1)
        - if matplotlib.axes.AxesSubplot instance or list of
            matplotlib.axes.AxesSubplot instances, those will be
            ravelled and returned
        - If ax is a tuple (x,y), figures will be created with x rows and
            y columns of subplots using the :func:`~nc2map.subplots`
            function. If ax is (x,y,z), only the first z subplots of each
            figure will be used.
    n: int
        number of subplots to create
    *args
        any arguments that are passed to the :func:`~nc2map.subplots` function
    **kwargs
        any keyword arguments that are passed to the :func:`~nc2map.subplots`
        function

    Returns
    -------
    list of maplotlib.axes.SubplotBase instances

    See Also
    --------
    nc2map.multiple_subplots, nc2map.subplots"""
        if not self.plot:
            if isinstance(ax, mpl.axes.SubplotBase):
                return [ax]
            try:
                if isinstance(ax[0], mpl.axes.SubplotBase):
                    return np.ravel(ax)
            except TypeError:
                pass
            return [None]*n
        return multiple_subplots(ax, n, *args, **kwargs)

    def get_label_dict(self, *args, **kwargs):
        """Returns dictionary with meta attributes

        Parameters
        ----------
        *args
            instances of :class:`~nc2map.mapos.MapBase`, figures,
            :class:`~nc2map.mapos.LinePlot`, etc.
        delimiter: str
            string that shall be used for separating values in a string, if
            multiple meta values are found in the specified input. If not given
            (or None), sets will be returned, not strings

        Returns
        -------
        meta: dict
            concatenated dictionary with meta informations as sets or as string
            (if delimiter keyword is given)"""
        if args == ():
            return {}
        if any(key != 'delimiter' for key in kwargs):
            raise KeyError(
                "Unknown keyword arguments %s" % ', '.join(
                    key for key in kwargs if key != 'delimiter' ))
        args = list(args)
        maps = []
        figs = self.get_figs(*(
            arg for arg in args if arg == 'wind'))
        args = [arg for arg in args if arg != 'wind']
        for fig in figs:
            if fig in args:
                maps = maps + figs[fig]
                args.remove(fig)
        maps = maps + args
        meta_keys = {}
        meta_keys = set(chain(*[mapo.meta.keys() for mapo in maps]))
        meta = {}
        for key in meta_keys:
            try:
                meta[key] = frozenset(
                    mapo.meta.get(key) for mapo in maps) - {None}
            except TypeError:  # in case of unhashable objects (e.g. lists)
                try:
                    meta[key] = set(np.unique(
                        mapo.meta.get(key) for mapo in maps)) - {None}
                except TypeError:
                    pass
        delimiter = kwargs.get('delimiter')

        if delimiter is not None:
            delimiter = str(delimiter)
            for key, val in meta.items():
                meta[key] = delimiter.join(map(str, val))
        return meta

    def _setupnames(self, names, vlst=None, sort=None, force_sort=False,
                    rename=True, **dims):
        """Sets up the names dictionary for the plot

        Parameters
        ----------
        names: string, list of strings or dictionary
            Set the unique names of the MapBase instances and (optionally)
            dimensions.

                - if string: same as list of strings (see below). Strings may
                    include {0} which will be replaced by a counter.
                - list of strings: this will be used for the name setting of
                    each mapo. The final number of maps depend in this case on
                    `vlst` and the specified ``**dims`` (see below)
                - dictionary:
                    Define the dimensions for each map directly.
                    Which dimensions are possible depend on the variable in
                    the `ncfile`. If you set this, `vlst` (see next) is
                    ambiguos.
                    Hint: With a standard dictionary it is not possible to
                    specify the order. Use the collections.OrderedDict class
                    for that or the `sort` keyword.
                    The structure of the dictionary must be like::

                        {'name-of-mapo1': {'dim1-for-mapo1': value,
                                        'dim2-for-mapo1': value, ...},
                        'name-of-mapo2': {'dim1-for-mapo2': value,
                                        'dim2-for-mapo2': value, ...},
                        ...}
        vlst: string or list of strings
            Defines the variables to extract
        sort: None or list of strings
            This parameter defines how the maps are ordered. It can be
            be either None, to apply no sorting, or

                1. None, then it will first be sorted by the variable and then
                    alphabetically by the iterable dimesion (e.g.
                    sort=['time', 'var'])
                2. a list of dimension strings matching to the iterable
                    dimensions in ``dims`` plus 'var' for the variable
                3. a list of names in `names` (does only work if `names` is a
                    dictionary)

        rename: bool
            If True, names will checked with the
            :meth:`~nc2map.MapsManager.check_name` method
        **dims
            Keys must be variable names of dimensions e.g. time, level, lat or
            lon (see timenames, etc. below) and values must be integers or
            iterables(e.g. lists) of ińtegers that can be extracted. For
            example consider a three dimensional field with dimensions (time,
            lat lon). dims = {'time': 0} will result in one map for the first
            time step.
            On the other hand dims = {'time': [0, 1]} will result in
            two maps, one for the first (time == 0) and one for the second
            (time == 1) time step.

        Returns
        -------
        name_dict: dict
            Ordered dictionary with names and keys sorted according to `sort`
            and values being the dimensions to extract
        """
        if isinstance(names, dict) and sort is None:
            return OrderedDict(names)
        if sort is not None:
            sort = list(sort)
        if isinstance(names, six.string_types):
            names = [names]
        if isinstance(vlst, six.string_types):
            vlst = [vlst]
        iter_dims = OrderedDict()
        for key, val in sorted(dims.items()):
            # try if iterable
            try:
                iter(val)
                iter_dims[key] = dims.pop(key)
            except TypeError:
                pass
        # do not iterate if only one iterable dimension exists and this
        # contains of integers only
        #if len(iter_dims) == 1 and isinstance(iter_dims.values()[0][0], int):
            #dims.update(iter_dims)
            #iter_dims = {}
        if vlst is None:
            raise ValueError(
                "vlst must not be None if names is not a dictionary!")
        start = 0  # start for enumerate
        if isinstance(names, dict):
            names = OrderedDict(names)
            if sort is None:
                names_keys = names.keys()
            elif any(name in sort for name in names):  # assume list of names
                names_keys = list(sort) + [name for name in names
                                           if name not in sort]
            else:  # assume dimension list and use np.argsort
                names_keys = np.array(names.keys())
                name_dims = frozenset(
                    chain(*(val.keys() for val in names.values())))
                dtypes = [
                    np.array([name_dict.get(dim, np.nan)
                              for name_dict in names.values()]).dtype
                    for dim in name_dims]
                dtype = [('name', names_keys.dtype)] + [
                    (name, dtypes[i]) for i, name in enumerate(name_dims)]
                names_vals = np.array(
                    [tuple([name] + [name_dict.get(dim, np.nan)
                                     for dim in name_dims])
                     for name, name_dict in names.items()],
                    dtype=dtype)
                names_keys = names_keys[np.argsort(names_vals, order=sort)]
            zipped_keys = (
                sorted(names[key].keys()) for key in names_keys)
            zipped_dims = (
                (item[1] for item in sorted(names[key].items()))
                for key in names_keys)
        else:
            if sort is None:
                zipped_keys = repeat(['var'] + iter_dims.keys())
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
                zipped_keys = repeat(zipped_keys)
            if names is None:
                names_keys = repeat('mapo{0}')
                for i in xrange(100):
                    if 'mapo%i' % i not in self.names:
                        start = i
                        break
            elif isinstance(names, six.string_types):
                names_keys = repeat(names)
            else:
                names_keys = names

        names = OrderedDict([
            (str(name).format(i), {
                dim: val for dim, val in zip(next(zipped_keys),
                                                dimstuple)})
            for i, (name, dimstuple) in enumerate(
                izip(cycle(names_keys), zipped_dims), start=start)])
        # update for non-iterable dimensions
        for settings in names.itervalues():
            for dim, val in dims.items():
                settings[dim] = val
        if rename:
            for name, dims in names.items():
                if name != self.check_name(name):
                    names[self.check_name(name)] = names.pop(name)
        return names

    def check_name(self, name):
        """Checks the name

        Checks whether the name is already in self.names and if yes,
        modifies it as name.i where i is an integer between 0 and 100

        Parameters
        ----------
        name: string
            Name to check

        Returns
        -------
        modified name: string
            name that is not in the maps attribute
        """
        if not self.maps and not self.lines:
            return name
        names = self.names
        if name in names:
            for i in xrange(1,100):
                if name + '.%i' % i in names:
                    if i == 100:
                        raise ValueError(
                            "Could not rename the instance!")
                    continue
                else:
                    warn("Found multiple names %s! --> Rename new "
                         " instance to %s" % (name, name + '.%i' % i))
                    return name + '.%i' % i
        return name

    def _setupfigs(self, names, fmt, subplots, reader=None, fromscratch=True,
                   u=None, v=None, meta=None):
        """set up the figures for the Maps instance

        This method sets up the :class:`~nc2map.mapos.MapBase` instances for
        the :meth:`~nc2map.MapsManager.addmap` method.

        Parameters
        ----------
        names: dict
            dictionary containing the names for the new mapos (see
            :meth:`~nc2map.MapsManager._setupnames)
        fmt: dict
            Formatoption dictionary (see :meth:`~nc2map.MapsManager._setupfmt`)
        subplots: list of subplots
            see :func:`~nc2map._make_subplots`
        reader: :class:`~nc2map.readers.ReaderBase` instance
            Only needed if fromscratch is True
        fromscratch: bool
            If True, new MapBase instances will be created. Otherwise, the
            exisiting ones will be used
        u: str
            name of the zonal wind variable (or None if it shall not be
            plotted)
        v: str
            name of the meridional wind variable (or None if it shall not be
            plotted)
        meta: dict
            dictionary containing meta information (see
            :meth:`~nc2map.MapsManager._setupfmt`)
        """
        windonly = self.windonly

        if windonly:
            mapo = WindPlot
        else:
            mapo = FieldPlot
        if not meta:
            meta = {name: {} for name in names}
        if len(subplots) != len(names):
            raise ValueError(
                ('Number of given axes (%i) does not fit to number of MapBase '
                 'instances (%i)!') % (len(subplots), len(names))
                )

        for subplot, (name, dims) in zip(subplots, names.items()):
            if fromscratch:
                self.maps.append(mapo(
                    name=name,  ax=subplot, fmt=fmt[name],
                    reader=reader, mapsin=self, u=u, v=v, meta=meta[name],
                    **dims))
            else:
                mapo = self.get_maps(name=name)[0]
                if hasattr(mapo, 'cbar'):
                    mapo._removecbar()
                    del mapo.cbar
                if hasattr(mapo, 'wind') and mapo.wind is not None:
                    mapo.wind._removeplot()
                    mapo.wind._ax = subplot
                mapo._ax = subplot
                mapo.update(plot=False, todefault=True, **fmt[mapo.name])
        return

    def _setupfmt(self, oldfmt, names=None):
        """set up the formatoption dictionary for the MapBase instances
        if names is None: use self.names"""
        # set up the dictionary for each variable
        if names is None:
            names = self.name_dicts
        dims_identifiers = names.keys()
        for key, val in chain(*map(lambda x: x.items(), names.values())):
            if key == 'time':
                try:
                    dims_identifiers.append("t%i" % val)
                except TypeError:
                    pass
            elif key == 'level':
                try:
                    dims_identifiers.append("l%i" % val)
                except TypeError:
                    pass
            elif key == 'var':
                dims_identifiers.append(val)
        removedims = lambda fmt: {key: val for key, val in fmt.items()
                                  if key not in dims_identifiers}
        if names is None:
            names = self.name_dicts
        if oldfmt is None:
            return {name: None for name in names}
        fmt = {}
        for name, name_dict in names.items():
            fmt[name] = removedims(oldfmt)
            var = name_dict.get('var')
            level = name_dict.get('level')
            time = name_dict.get('time')
            try:
                fmt[name].update(removedims(oldfmt["l%i" % level]))
            except (KeyError, TypeError):
                pass
            try:
                fmt[name].update(removedims(oldfmt["t%i" % time]))
                try:
                    fmt[name].update(
                        removedims(oldfmt["t%i" % time]["l%i" % level]))
                except (KeyError, TypeError):
                    pass
            except (KeyError, TypeError):
                pass
            try:
                fmt[name].update(removedims(oldfmt[var]))
                try:
                    fmt[name].update(removedims(oldfmt[var]["t%i" % time]))
                    try:
                        fmt[name].update(
                            removedims(oldfmt[var]["t%i" % time][
                                "l%i" % level]))
                    except (KeyError, TypeError):
                        pass
                except (KeyError, TypeError):
                    pass
            except (KeyError, TypeError):
                pass
            fmt[name].update(removedims(oldfmt.get(name, {})))
        return fmt

    def addmap(self, ncfile, names=None, vlst=None, ax=(1, 1), sort=None,
               fmt=None, u=None, v=None, mode=None, dims={}, windonly=False,
               plot=True, meta={}, **kwargs):
        """add a map instance to the instance

        This method can be used to add an additional map plot (i.e. a
        :class:`~nc2map.mapos.MapBase` instance) to this
        :class:`~nc2map.MapsManager` instance. This method is called at the
        initialization of a :class:`~nc2map.Maps` instance.

        Parameters
        ----------
        ncfile: str, MapBase instance (or lists of both) or a reader, optional

            - If MapBase instance or list of MapBase instances, all of the
                other keywords are obsolete (besides windonly and plot).
            - If string or list of strings, it must be the path to the
                netCDF-file(s)  containing the data for all variables.
                Filenames may contain wildcards (`*`, ?, etc.) as suitable
                with the Python glob module (the netCDF4.MFDataset is used
                to open the nc-file).
            - If reader (i.e. a :class:`~nc2map.readers.ReaderBase`
                instance), this will be used to extract the data from
        names: string, list of strings or dictionary, optional
            Set the unique names of the MapBase instances and (optionally)
            dimensions.

                - if string: same as list of strings (see below). Strings may
                    include {0} which will be replaced by a counter.
                - list of strings: this will be used for the name setting of
                    each mapo. The final number of maps depend in this case on
                    `vlst` and the specified ``**dims`` (see below)
                - dictionary:
                    Define the dimensions for each map directly.
                    Which dimensions are possible depend on the variable in
                    the `ncfile`. If you set this, `vlst` (see next) is
                    ambiguos.
                    Hint: With a standard dictionary it is not possible to
                    specify the order. Use the collections.OrderedDict class
                    for that or the `sort` keyword.
                    The structure of the dictionary must be like::

                        {'name-of-mapo1': {'dim1-for-mapo1': value,
                                        'dim2-for-mapo1': value, ...},
                        'name-of-mapo2': {'dim1-for-mapo2': value,
                                        'dim2-for-mapo2': value, ...},
                        ...}
        vlst: string or list of strings, optional
            Default: None. List containing variables in the reader that shall
            be plotted (or only one variable). The given strings names must
            correspond to the names in `ncfile`. If None and windonly is
            False, all variables that have a latitude and longitude dimension
            are used. If None and windonly is True, it will be set to ['wind']
            and this name will be used for the wind variable
        u: str, optional
            Default: None. Name of the zonal wind variable in the `ncfile`, if
            a WindPlot shall be visualized
        v: str, optional
            Default: None. Name of the meridional wind variable in the
            `ncfile`, if a WindPlot shall be visualized
        ax: tuple (x,y[,z]), subplot or list of subplots, optional
            Default: (1, 1)
            - if matplotlib.axes.AxesSubplot instance or list of
                matplotlib.axes.AxesSubplot instances, the data will be plotted
                into these axes.
            - If ax is a tuple (x,y), figures will be created with x rows and
                y columns of subplots using the :func:`~nc2map.subplots`
                function. If ax is (x,y,z), only the first z subplots of each
                figure will be used.
        windonly: bool, optional
            Default: False. If True, no underlying scalar field is plotted but
                only the vector field set up by u and v using the
                :class:`~nc2map.mapos.WindPlot` class
        sort: None or list of strings, optional
            This parameter defines how the maps are ordered. It can be
            be either None, to apply no sorting, or

                1. None, then it will first be sorted by the variable and then
                    alphabetically by the iterable dimesion (e.g.
                    sort=['time', 'var'])
                2. a list of dimension strings matching to the iterable
                    dimensions in ``dims`` plus 'var' for the variable
                3. a list of names in `names`
        mode: string {%(readers)s} or None, optional
            If None, it is tried to set the reader automatically with the one
            that matches.
        dims: dict, optional
            Keys must be variable names of dimensions e.g. time, level, lat or
            lon (see timenames, etc. below) and values must be integers or
            iterables(e.g. lists) of ińtegers that can be extracted. For
            example consider a three dimensional field with dimensions (time,
            lat lon). dims = {'time': 0} will result in one map for the first
            time step.
            On the other hand dims = {'time': [0, 1]} will result in
            two maps, one for the first (time == 0) and one for the second
            (time == 1) time step.
        fmt: dict, optional
            Default: None. Dictionary with formatoption keywords controlling
            the appearance of the plots (see below for the structure).
            Possible keywords are (see :mod:`nc2map.formatoptions` and
            :func:`~nc2map.show_fmtdocs`)

%(fmt_keys)s

            and the windplot specific keywords (see
            :func:`~nc2map.show_fmtdocs('wind')`) are

%(windfmt_keys)s

        meta: dict
            Dictionary with user defined meta information for each MapBase
            instance. Syntax is the same as for `fmt` (see below)

        Other Parameters
        ----------------
        timenames: 1D-array of strings, optional
            Default: %(timenames)s. Gives the name of the time dimension for
            which will be searched in `ncfile`
        levelnames: 1D-array of strings, optional
            Default: %(levelnames)s. Gives the name of the dimension (e.g
            vertical levels) for which will be searched in `ncfile`
        lonnames: 1D-array of strings, optional
            Default: %(lonnames)s. Gives the name of the longitude names for
            which will be searched in `ncfile`
        latnames: 1D-array of strings, optional
            Default: %(latnames)s. Gives the name of the latitude names for
            which will be searched in `ncfile`
        udims: 1D-array of strings, optional
            Default: %(udims)s. Dimension names in `ncfile` that indicates
            that the variable is defined on an unstructured grid

        Other ``**kwargs`` may be any dimension name - value pair from the
        NetCDF file (same rules as for the `dims` keyword).

        Examples
        --------
        Open all variables in a NetCDF file::

            ncfile = "my-own-ncfile.nc"
            mymaps.addmap(ncfile)

        Open specific variables by their name for the first and second time
        step::

            mymaps.addmap(ncfile, vlst=['t2m', 'u'], time=[0, 1])

        Open the 1st of April 2015::

            mymaps.addmap(ncfile, time=["2015-04-01"])

        Notes
        -----
        Syntax of fmt is as follows::

                fmt = {['<<<var>>>':{
                        ['t<<<time>>>': {
                            ['l<<<level>>>': {'keyword': ..., ...}]
                            [, 'keyword': ...,...]
                            }]
                        [, 'l<<<level>>>: {'keyword': ..., ...}]
                        [, 'keyword': ..., ...]
                        }]
                    [, 't<<<time>>>': {
                        ['l<<<level>>>': {'keyword': ..., ...}]
                        [, 'keyword': ...,...]
                        }]
                    [, 'l<<<level>>>: {'keyword':..., ...}]
                    [, <<<name>>>: {'keyword':..., ...}]
                    [, 'keyword':..., ...]
                    }.
        Seems complicated, but in fact rather simple considering
        following rules:
            - Formatoptions are set via 'keyword': value
            - Time and level specific keywords are put into a dictionary
                indicated by the key
                't<<<time>>>' or 'l<<<level>>>' respectively (where <<<time>>>
                and <<<level>>> is the number of the time, and or level).
            - To set default formatoptions for each map: set the keyword
                the upper most hierarchical level of formatoptions (e.g.
                fmt={'plotcbar':'r'}).
            - To set default formatoptions for each variable, times or
                separately set the keyword in the second hierarchical level
                of formatoptions (e.g.::

                    fmt = {'t4':{'plotcbar:'r'}}

                will only change the formatoptions of maps with time equal to
                4,::

                    fmt = {'l4':{'plotcbar:'r'}}

                will only change formatoptions of maps with level equal to 4).
            - To set default options for a specific variable and time, but
                all levels: put them in the 3rd hierarchical level of
                formatoptions (e.g.::

                    fmt = {<<<var>>>:{'t4':{'plotcbar':'r'}}}

                will only change the formatoptions of each level
                corresponding to variable <<<var>>> and time 4). Works the
                same for setting default options for specific variable and
                level, but all times.
            - To set a specific key for one map,
                just set::

                    fmt = {<<<var>>>: {'t<<<time>>>': {'l<<<level>>>':
                            {'plotcbar: 'r', ...}}}}

                or directly with the name of the MapBase instance (see names
                keyword)::

                    fmt = {<<<name>>>: {'plotcbar': 'r', ...}}.
        """
        # if single MapBase instance, just add it
        self.logger.debug("Got key word arguments:")
        dims = deepcopy(dims)
        dims.update(kwargs)
        dim_names = ['timenames', 'levelnames', 'lonnames', 'latnames',
                     'udims']
        dim_names = dict(item for item in dims.items() if item[0] in dim_names)
        for key in dim_names.keys():
            del dims[key]
        kwargs_keys = ['ncfile', 'names', 'vlst', 'u', 'v',  'ax', 'fmt',
                       'plot', 'sort', 'windonly']
        max_kwarg_len = max(map(len, kwargs_keys)) + 2
        for key in kwargs_keys:
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              key, str(locals()[key]))
        if dims:
            self.logger.debug('Other dims:')
            max_kwarg_len = max(map(len, dims.keys())) + 2
            for key, val in dims.items():
                self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                                  key, val)
        if dim_names:
            self.logger.debug('Other dims:')
            max_kwarg_len = max(map(len, dim_names.keys())) + 2
            for key, val in dim_names.items():
                self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                                  key, val)

        self.windonly = windonly
        self.plot = plot
        if isinstance(ncfile, MapBase):
            ncfile = [ncfile]
        # if many MapBase instances, just add them
        try:
            self.logger.debug("Try list of MapBase instances")
            if isinstance(ncfile[0], MapBase):
                if any(not isinstance(mapo, MapBase) for mapo in ncfile):
                    raise ValueError((
                        "Found mixture of objects in Input. "
                        "Please use only MapBase instances or strings!"))
                newnames = []
                for mapo in ncfile:
                    names = self.names
                    if mapo.name in names:
                        for i in xrange(1,100):
                            if mapo.name + '.%i' % i in names:
                                if i == 100:
                                    raise ValueError(
                                        "Could not rename the MapBase "
                                        "instance!")
                                continue
                            else:
                                warn("Found multiple names %s! --> Rename new "
                                     "MapBase instance to %s" % (
                                         mapo.name, mapo.name + '.%i' % i))
                                mapo.name = mapo.name + '.%i' % i
                                break
                newnames.append(mapo.name)
                self.maps.append(mapo)
                if plot:
                    self.logger.info("Setting up projections...")
                    for mapo in self.get_maps(name=newnames):
                        mapo._setupproj()
                    self.logger.info("Making plots...")
                    self.make_plot(name=newnames)
                return
        except (TypeError, KeyError, AttributeError):
            self.logger.debug("Failed.", exc_info=True)
            pass
        # else, initialize them

        if isinstance(ncfile, dict):
            self.set_reader(mode, **{
                key: val for key, val in ncfile.items() + dim_names.items()})
        else:
            self.set_reader(mode, ncfile, **dim_names)
        if vlst is None and not self.windonly:
            vlst = self.reader.lola_variables.keys()
            if len(vlst) == 0:
                raise ValueError(
                    "Found no displayable variables in the input while looking"
                    " for longitude names %s and latitude names %s!" % (
                        ', '.join(self.reader._lonnames),
                        ', '.join(self.reader._latnames)))
        elif vlst is None and self.windonly:
            vlst = ['wind']
        elif isinstance(vlst, str):
            vlst = [vlst]
        name_dicts = self._setupnames(names=names, vlst=vlst, sort=sort,
                                      **dims)
        names = name_dicts.keys()
        subplots = self._make_subplots(ax, len(names))
        self._setupfigs(name_dicts, self._setupfmt(fmt, name_dicts), subplots,
                        self.reader, u=u, v=v,
                        meta=self._setupfmt(meta, name_dicts))
        if self.plot:
            self.logger.info("Setting up projections...")
            for mapo in self.get_maps(name=names):
                mapo._setupproj()
            self.logger.info("Making plots...")
            self.make_plot(name=names)

            for fig in self.get_figs():
                self._set_window_title(fig)

    def addline(self, ncfile, names=None, vlst=None, name=None,
                 ax=None, fmt={}, pyplotfmt={}, sort=None, color_cycle=None,
                 independent='x', mode=None, dims={}, meta={},
                 plot=True, **kwargs):
        """Add a one dimensional line plot to the MapsManager instance

        Parameters
        ----------
        ncfile: str, MapBase instance (or lists of both) or a reader

            - If MapBase instance or list of MapBase instances, all of the
                other keywords are obsolete (besides windonly and plot).
            - If string or list of strings, it must be the path to the
                netCDF-file(s)  containing the data for all variables.
                Filenames may contain wildcards (`*`, ?, etc.) as suitable
                with the Python glob module (the netCDF4.MFDataset is used
                to open the nc-file).
            - If reader (i.e. a :class:`~nc2map.readers.ReaderBase`
                instance), this will be used to extract the data from
        names: string, list of strings or dictionary
            Set the unique names of the lines and (optionally) dimensions.

                - if string: same as list of strings (see below). Strings may
                    include {0} which will be replaced by a counter.
                - list of strings: this will be used for the name setting of
                    each line. If 'label' is not explicitly set in `pyplotfmt`,
                    this will also set the label of the line. The final number
                    of maps depend in this case on `vlst` and the specified
                    ``**dims`` (see below)
                - dictionary:
                    Define the dimensions for each map directly.
                    Which dimensions are possible depend on the variable in
                    the `ncfile`. If you set this, `vlst` (see next) is
                    ambiguos.
                    Hint: With a standard dictionary it is not possible to
                    specify the order. Use the collections.OrderedDict class
                    for that.
                    The structure of the dictionary must be like::

                        {'name-of-line1': {'dim1-for-line1': value,
                                        'dim2-for-line1': value, ...},
                        'name-of-line2': {'dim1-for-line2': value,
                                        'dim2-for-line2': value, ...},
                        ...}
        vlst: string or list of strings
            Default: None. List containing variables in the reader that shall
            be plotted (or only one variable). The given strings names must
            correspond to the names in `ncfile`. If None, all variables which
            are not declared as dimensions are used.
        name: str
            Name of the LinePlot instance
        ax: mpl.axes.SubplotBase
            axes to to plot on. If None, a new figure and axes will be created
        fmt: Dictionary containing formatoption keywords. Possible
            formatoption keywords (see :mod:`~nc2map.formatoptions` or
            :func:`~nc2map.show_fmtdocs()`) are

%(fmt_keys)s

        pyplotfmt: dict
            Dictionary containing additional keywords for the plt.plot function
            which is used to plot each line. The dictionary may also contain
            line names (see names above) to specify the options for the
            specific line directly.
            E.g {'line0': {'color': 'k'}} will set the color for the line
            with name 'line0' to black.
            {'color': 'b', 'line0': {'color': 'k'}} will set the color for
            all but 'line0' to blue.
        sort: None or list of dimension strings.
            If names is not a dictionary, this defines how the lines are
            ordered, dependent on the iterable dimensions in **dims and vlst
            (e.g. sort=['time', 'var']). If sort is None, it will first sorted
            by the variable and then alphabetically by the iterable dimesion.
        mode: string {%(readers)s} or None.
            If None, it is tried to set the reader with the one that matches.
        color_cycle: iterable
            Any iterable (e.g. a list) containing color definitions or a
            colormap suitable for :func:`~nc2map.get_cmap` method
        independent: {'x', 'y'}
            Specifies which is the independent axis, i.e. on which axis to plot
            the dimension.
        meta: dict
            Dictionary with user defined meta information for each LinePlot
            instance. Syntax is the same as for `fmt` (see below)
        dims: dictionary. Keys must be variable names of dimensions
            e.g. time, level, lat or lon (see timenames, etc.
            below) and values must be integers or iterables(e.g. lists) of
            ińtegers that can be extracted. (see the dimension example below)
            Please note, that by default it is looked for %(timenames)s
            for the time dimension, %(levelnames)s for the (vertical) level
            dimension, %(lonnames)s for the longitude dimension and
            %(latnames)s for the latitude dimension. You can change this by
            giving the timenames, etc. keywords below.

        Other Parameters
        ----------------
        timenames: 1D-array of strings, optional
            Default: %(timenames)s. Gives the name of the time dimension for
            which will be searched in `ncfile`
        levelnames: 1D-array of strings, optional
            Default: %(levelnames)s. Gives the name of the dimension (e.g
            vertical levels) for which will be searched in `ncfile`
        lonnames: 1D-array of strings, optional
            Default: %(lonnames)s. Gives the name of the longitude names for
            which will be searched in `ncfile`
        latnames: 1D-array of strings, optional
            Default: %(latnames)s. Gives the name of the latitude names for
            which will be searched in `ncfile`
        udims: 1D-array of strings, optional
            Default: %(udims)s. Dimension names in `ncfile` that indicates
            that the variable is defined on an unstructured grid

        Other ``**kwargs`` may be any dimension name - value pair from the
        NetCDF file (same rules as for the `dims` keyword).

        Examples
        --------
        Assume a 3 dimensional ('time', 'lat', 'lon') temperature field
        stored in the variable 't2m'. To plot the temperature for all
        longitudes for the first time step and 3rd latitude index, use::

            mymaps.addline('test.nc', vlst='t2m', time=0, lat=3)

        which is equivalent to::

            mymaps.addline('test.nc', vlst='t2m', time=0, lat=3,
                           lon=slice(None))

        On the other hand::

            mymaps.addline('test.nc', vlst='t2m', time=[0, 1], lat=3)

        will create 2 lines, one for the first and one for the second
        timestep.
        However, if you only want to visualize parts of the longitudes,
        setting::

            mymaps.addline('test.nc', vlst='t2m', time=[0, 1], lat=3,
                           lon=[1, 2, 3, 4, 5])
        would result in an error because it will be iterated over
        [1, 2, 3, 4, 5]. Here is how you can fix it::

            from collections import cycle
            mymaps.addline('test.nc', vlst='t2m', time=[0, 1], lat=3,
                           lon=cycle([[1, 2, 3, 4, 5]]))"""
        def new_name():
            linenames = self.linenames
            for i in xrange(100):
                if 'line%i' % i in linenames:
                    continue
                return 'line%i' % i
        self.plot = plot
        self.logger.debug("Got key word arguments:")
        dims = deepcopy(dims)
        dims.update(kwargs)
        dim_names = ['timenames', 'levelnames', 'lonnames', 'latnames',
                     'udims']
        dim_names = dict(item for item in dims.items() if item[0] in dim_names)
        for key in dim_names.keys():
            del dims[key]
        kwargs_keys = ['ncfile', 'names', 'vlst', 'name', 'ax', 'fmt',
                       'pyplotfmt', 'sort', 'color_cycle', 'independent',
                       'mode', 'dims', 'meta', 'plot']
        max_kwarg_len = max(map(len, kwargs_keys)) + 2
        if name is None:
            name = new_name()
        for key in kwargs_keys:
            self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                              key, str(locals()[key]))
        if dims:
            self.logger.debug('Other dims:')
            max_kwarg_len = max(map(len, dims.keys())) + 2
            for key, val in dims.items():
                self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                                  key, val)
        if dim_names:
            self.logger.debug('Other dims:')
            max_kwarg_len = max(map(len, dim_names.keys())) + 2
            for key, val in dim_names.items():
                self.logger.debug('    %s:'.ljust(max_kwarg_len)  + '%s',
                                  key, val)
        if isinstance(ncfile, SimplePlot):
            ncfile = [ncfile]
        # if many MapBase instances, just add them
        try:
            self.logger.debug("Try for SimplePlot instances")
            if isinstance(ncfile[0], SimplePlot):
                if any(not isinstance(line, SimplePlot) for line in ncfile):
                    raise ValueError((
                        "Found mixture of objects in Input. "
                        "Please use only SimplePlot instances or strings!"))
                newnames = []
                for line in ncfile:
                    line.name = self.check_name(line.name)
                    newnames.append(line.name)
                    self.lines.append(line)
                if plot:
                    self.logger.info("Making plots...")
                    for line in self.get_maps(name=newnames, mode='lines'):
                        line.make_plot()
                    for fig in self.get_figs(name=newnames, mode='lines'):
                        self._set_window_title(fig)
                return
        except TypeError:
            self.logger.debug("Failed.", exc_info=True)
            pass
        if isinstance(ncfile, dict):
            self.set_reader(mode, **{
                key: val for key, val in ncfile.items() + dim_names.items()})
        else:
            self.set_reader(mode, ncfile, **dim_names)
        name = self.check_name(name)
        self.lines.append(LinePlot(
            self.reader, names=names, vlst=vlst, name=name,
            ax=ax, fmt=fmt, pyplotfmt=pyplotfmt, sort=sort, mapsin=self,
                 color_cycle=color_cycle, independent=independent,
                 **dims))
        if plot:
            self.logger.info("Making plots...")
            self.lines[-1].make_plot()
            self._set_window_title(self.lines[-1].ax.get_figure())

    def _set_window_title(self, fig):
        """sets the canvas window title"""
        fig.canvas.set_window_title(
            'Figure %i: %s' % (fig.number, self._replace(
                '%(all)s', fig, delimiter=', ')))

    def set_logger(self, name=None, force=False):
        """This function sets the logging.Logger instance in the MapsManager
        instance.

        Parameters
        ----------
        name: str
            name of the Logger (if None: it will be named like
            <module name>.<class name>)
        force: bool
            Default: False. If False, do not set it if the instance has
            already a logger attribute."""
        if name is None:
            name = '%s.%s' % (self.__module__, self.__class__.__name__)
        if not hasattr(self, 'logger') or force:
            self.logger = logging.getLogger(name)
            self.logger.debug('Initializing...')

    def _resort_maps(self, names):
        """resorts maps attribute according to the specified names

        Parameters
        ----------
        names: list of strings
            names of MapBase instances in the :attr:`~nc2map.Maps.maps`
            attribute"""
        maps = set(self.maps[:])
        wrong_names = [name for name in names if name not in self.mapnames]
        if wrong_names:
            warn("Unknown map names %s!" % ', '.join(wrong_names))
        names = [name for name in names if name not in wrong_names]
        # add missing names and put them at the end
        names = list(names) + [name for name in self.mapnames
                               if name not in names]
        self.maps = [[mapo for mapo in self.maps if mapo.name == name][0]
                     for name in names]

    def asdict(self, *args, **kwargs):
        """returns the current formatoptions settings

        Dependent on ``*args``, several dictionaries are returned

        Parameters
        ----------
        *args
            Arguments may be strings {'maps', 'lines'}

                - 'maps' for a dictionary of the formatoptions of the
                    MapBase instances in :attr:`~nc2map.MapsManager.maps`
                    attribute
                - 'lines' for a dictionary of the formatoptions of the
                    LinePlot instances in :attr:`~nc2map.MapsManager.lines`
                    attribute
        **kwargs
            any key-value pair of the :meth:`~nc2map.MapsManager.get_maps`
            method

        Returns
        -------
        fmt: list
            list of formatoption settings. if 'maps' in ``*args``, the first
            entry will be the maps formatoptions, and if 'lines' in ``*args``,
            the second the formatoptions of the lines.

        See Also
        --------
        nc2map.mapos.MapBase.asdict: used if 'maps' is in ``*args``
        nc2map.mapos.LinePlot.asdict: used if 'lines' is in ``*args``"""
        out = []
        if args == () or 'maps' in args:
            kwargs['mode'] = 'maps'
            out += [{mapo.name: mapo.asdict()
                    for mapo in self.get_maps(**kwargs)}]
        if args == () or 'lines' in args:
            kwargs['mode'] = 'lines'
            out += [{line.name: line.asdict()
                     for line in self.get_maps(**kwargs)}]
        return out

    def get_fmtkeys(self, *args):
        """Get formatoption keys as a list

        Parameters
        ----------
        *args: str
            any formatoptions keyword (without: return all). Further ``*args``
            may be,

            - 'wind': to plot the wind formatoption keywords
            - 'windonly': to plot the wind only
            - 'simple': to print the formatoptions of
                :class:`~nc2map.formatoptions.SimpleFmt` for
                :class:`~nc2map.mapos.SimplePlot` instances (e.g.
                :class:`~nc2map.mapos.Lineplot`, etc.)

        Returns
        -------
        fmt_keys: list
            keys in ``*args`` that really are formatoptions"""
        if 'wind' not in args:
            return self.get_maps()[0].get_fmtkeys(*args)
        else:
            return self.get_winds()[0].get_fmtkeys(*args)

    def set_meta(self, maps=None, meta={}, **kwargs):
        """Sets meta information for the given mapos

        This methods assigns meta informations to the given maps

        Parameters
        ----------
        maps: list
            MapBase instances for which to which the meta informations shall
            be assigned. If None, defaults to maps attribute.
        meta: dict
            Key value pairs must be the meta identifier and the value that
            shall be assigned (you can use this if ``**kwargs`` does not work)
        **kwargs
            Key value pairs must be the meta identifier and the value that
            shall be assigned

        Examples
        --------
        Set information about the experiment::

            mymaps.set_meta(exp='RCP8.5')
            print mymaps.meta['exp']
            # returns 'RCP8.5'

        Set meta information for specific maps::

            maps = mymaps.get_maps(name=['mapo0'])
            mymaps.set_meta(maps, exp='RCP8.5')"""
        try:
            maps = iter(maps)
        except TypeError:
            maps = self.maps
        for mapo in maps:
            mapo.set_meta(meta, **kwargs)

    def show_fmtkeys(self, *args):
        """Print formatoption keys in a readable manner

        Parameters
        ----------
        *args: str
            any formatoptions keyword (without: print all). Further ``*args``
            may be,

            - 'wind': to plot the wind formatoption keywords
            - 'windonly': to plot the wind only
            - 'simple': to print the formatoptions of
                :class:`~nc2map.formatoptions.SimpleFmt` for
                :class:`~nc2map.mapos.SimplePlot` instances (e.g.
                :class:`~nc2map.mapos.Lineplot`, etc.)"""
        if 'wind' not in args:
            self.get_maps()[0].show_fmtkeys(*args)
        else:
            self.get_winds()[0].show_fmtkeys(*args)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    # ------------------ modify docstrings here --------------------------
    addmap.__doc__ = addmap.__doc__ % dict(
        [('readers', ', '.join(map(repr, readers.readers))),
         ('fmt_keys', _get_fmtkeys_formatted(indent=12)),
         ('windfmt_keys', _get_fmtkeys_formatted(
             'wind', 'windonly', indent=12))] + readers.defaultnames.items())
    addline.__doc__ = addline.__doc__ % dict(
        [('readers', ', '.join(map(repr, readers.readers))),
         ('fmt_keys', _get_fmtkeys_formatted('xy', indent=12))] + \
             readers.defaultnames.items())
    set_reader.__doc__ = set_reader.__doc__ % (
        {'readers': ', '.join(readers.readers)})
