# -*- coding: utf-8 -*-
"""Module containing the nc2map Maps class

Basic control class for the nc2map module, governing multiple CbarManager
instances, multiple EvaluatorBase instances and multiple MapBase and
LinePlot instances."""
import glob
import logging
import pickle
from copy import copy, deepcopy
from collections import OrderedDict
from itertools import izip, izip_longest, chain, permutations, product
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from _cbar_manager import CbarManager
import readers
from .warning import warn, critical
from _maps_manager import MapsManager, _props
from mapos import MapBase, FieldPlot, WindPlot, returnbounds, round_to_05
import formatoptions
from formatoptions import FmtBase
from evaluators import ViolinEval, FldMeanEvaluator


evaluatorsdict = {  # dictionary with evaluator classes
    'violin': ViolinEval,
    'fldmean': FldMeanEvaluator
    }


class Maps(MapsManager):
    """object controlling multiple :class:`~nc2map.mapos.MapBase` instances

    This is the base class of the :mod:`nc2map` module, that controls the plot
    and interactive usage of multiple variables at the same time.

    Parameters
    ----------
    linesonly: bool
        plots lines only and uses the :meth:`addline` method


    Further ``*args`` and ``**kwargs`` are determined by the :meth:`addmap`
    (or :meth:`addline`) method.

    Examples
    --------
    Open all variables in a NetCDF file::

        ncfile = "my-own-ncfile.nc"
        mymaps = nc2map.Maps(ncfile)

    Open specific variables by their name for the first and second time step::

        mymaps = nc2map.Maps(ncfile, vlst=['t2m', 'u'], time=[0, 1])

    Open the 1st of April 2015::

        mymaps = nc2map.Maps(ncfile, time=["2015-04-01"])

    Attributes
    ----------
    maps: list
        List of :class:`~nc2map.mapos.MapBase` instances (see also
        :meth:`get_maps` method)
    lines: list
        List of :class:`~nc2map.mapos.LinePlot` instances (see also
        :meth:`get_maps` method)
    evaluators: list
        List of :class:`~nc2map.evaluators.EvaluatorBase` instances (see also
        the :meth:`evaluate` method
    meta


    Methods
    -------
    addmap(ncfile, vlst, ...)
        add another map to the Maps instance
    addline(ncfile, vlst, ...)
        add another LinePlot to the Maps instance
    update(...)
        update the plots
    output(filename, ...)
        Saves the specified figures to a file
    make_movie(filename, ...)
        makes a movie of the specified figures and saves it
    update_cbar(...)
        updates and creates shared colorbars to handle multiple
        :class:`~nc2map.mapos.MapBase` instances
    evaluate(maps, ...)
        evaluate your maps with %s evaluators
    nextt(*args, **kwargs)
        updates specified maps to the next timestep
    prevt(*args, **kwargs)
        updates all maps to the previous timestep
    reset(...)
        reinitializes the Maps instance, creates new figures and makes a new
        plot
    show()
        show all figures
    undo(num=-1)
        undo the last `num` changes made
    redo(num=1)
        redo `num` changes that were undone by the :meth:`undo`
        method
    disable_maps(**kwargs)
        disables the specified :class:`~nc2map.mapos.MapBase` from the
        updating process (see :meth:`update`)
    enable_maps(**kwargs)
        enables the specified :class:`~nc2map.mapos.MapBase` in the
        updating process (see :meth:`update`)
    dump_nc(output, ...)
        save your data into a new NetCDF file
    get_disabled(**kwargs)
        Returns disabled maps specified by ``**kwargs``
    get_maps(mode='both', maps=[], _meta={}, **kwargs)
        returns a list of all :class:`~nc2map.mapos.MapBase` instances
        contained in the :class:`Maps` instance
    get_evaluator(*args, **kwargs)
        return the class of one of the above evaluators to see their
        documentations
    get_cbars(*args, **kwargs)
        returns a list of shared colorbars (i.e. the corresponding
        :class:`~nc2map.CbarManager` instances)
    get_figs(*args, **kwargs)
        returns a dictionary with figs as keys and the corresponding
        :class:`~nc2map.mapos.MapBase` instance
    get_winds(*args, **kwargs)
        if not windonly: Returns the :class:`~nc2map.mapos.WindPlot` instances
        of the :class:`~nc2map.mapos.FieldPlot` instances
    asdict(*args)
        returns the current formatoptions of the Maps instance as dictionary
    save(filename, ...)
        creates a pickle file which can be used to reload the Maps instance
        with the :func:`nc2map.load` function
    removecbars(*args)
        removes the specified colorbars if any cbars are drawn with the
        :meth:`update_cbar` method
    close(num, **kwargs)
        closes the Maps instance and all (or some) corresponding
        :class:`~nc2map.mapos.MapBase` instances and figures


    See methods below for further details.
    """
    __doc__ %= ', '.join(evaluatorsdict)
    # ------------------ define properties here -----------------------

    # mapdata dictionary property
    evaluators = _props.default(
        'evaluators', """
        List containing the evaluator instances of the Maps instance""")

    def __init__(self, *args, **kwargs):
        """Initialization method for Maps instance

        Parameters
        ----------
        linesonly: bool
            plots lines only and uses the :meth:`addline` method

        Further ``*args`` and ``**kwargs`` are determined by the
        :meth:`~nc2map.Maps.addmap` method (or :meth:`~nc2map.Maps.addline`
        method)."""
        self.set_logger()
        super(Maps, self).__init__()
        self._cbars = []
        self.evaluators = []
        self._disabled = set()
        try:
            linesonly = kwargs.pop('linesonly')
        except KeyError:
            linesonly = False
        if not kwargs.get('_noadd'):
            if linesonly:
                self.addline(*args, **kwargs)
            else:
                self.addmap(*args, **kwargs)

    def evaluate(self, evalname, *args, **kwargs):
        """Perform an evaluation of your plots

        Makes visual evaluations of the specified
        :class:`~nc2map.mapos.MapBase`

        Parameters
        ----------
        evalname: {%s}
            Name of the evaluator

        ``*args`` and ``**kwargs`` depend on the chosen evalutor.

        See Also
        --------
        nc2map.Maps.get_evaluator: method for documentation"""
        # docstring is extended below
        try:
            evaluator = evaluatorsdict[evalname]
        except KeyError:
            raise KeyError(
                "Unknown evaluator %s! Possible evaluator names are %s!" % (
                    evalname, ', '.join(evaluatorsdict)))
        self.evaluators.append(
            evaluator(*args, mapsin=self, **kwargs))
        return self.evaluators[-1]

    def get_evaluator(self, evalname):
        """Returns the evaluator class

        Returns the evaluator class specified by `evalname` for out of the
        :mod:`nc2map.evaluators` module.

        Parameters
        ----------
        evalname: {%s}
            Name of the evaluator

        Returns
        -------
        evaluator: object
            :class:`~nc2map.evaluators.EvaluatorBase` subclass corresponding
            to `evalname`

        See Also
        --------
        nc2map.Maps.evaluate: method for documentation"""
        # docstring is extended below
        try:
            return evaluatorsdict[evalname]
        except KeyError:
            raise KeyError(
                "Unknown evaluator %s! Possible evaluator names are %s!" % (
                    evalname, ', '.join(evaluatorsdict)))

    def disable_maps(self, **kwargs):
        """Disables maps and lines.

        Disables the specified :class:`~nc2map.mapos.MapBase` and
        :class:`~nc2map.mapos.LinePlot` instances from the updating
        (:meth:~nc2map.Maps.update), output, etc..
        ``**kwargs`` are determined by the :meth:`~nc2map.Maps.get_maps`
        method.

        See Also
        --------
        nc2map.Maps.enable_mapo: Method to reenable the instances
        nc2map.Maps.get_disabled: Method to get disabled Maps"""
        maps = self.get_maps(**kwargs)
        self._disabled.update(maps)
        for mapo in maps:
            try:
                self.maps.remove(mapo)
            except ValueError:
                self.lines.remove(mapo)

    def enable_maps(self, **kwargs):
        """Enables maps and lines

        Enables the specified :class:`~nc2map.mapos.MapBase` and
        :class:`~nc2map.mapos.LinePlot` instances for the updating method
        (:meth:~nc2map.Maps.update), output, etc..
        ``**kwargs`` are determined by the :meth:`~nc2map.Maps.get_maps`
        method.

        See Also
        --------
        nc2map.Maps.disable_mapo: Method to disable the maps or lines
        nc2map.Maps.get_disabled: Method to get disabled Maps"""
        try:
            maps = kwargs.pop('maps')
        except KeyError:
            maps = self._disabled
        maps = self.get_maps(maps=maps, **kwargs)
        self._disabled -= set(maps)
        for mapo in maps:
            if isinstance(mapo, MapBase):
                self.maps.append(mapo)
            else:
                self.lines.append(mapo)

    def get_disabled(self, **kwargs):
        """Returns disabled maps and lines

        Gives the specified disabled :class:`~nc2map.mapos.MapBase` and
        :class:`~nc2map.mapos.LinePlot`.
        ``**kwargs`` are determined by the :meth:`~nc2map.Maps.get_maps`
        method.

        Returns
        -------
        maps and lines: list
            List of the disabled :class:`~nc2map.mapos.MapBase` and
            :class:`~nc2map.mapos.LinePlot` instances

        See Also
        --------
        nc2map.Maps.disable_mapo: Method to disable the maps or lines
        nc2map.Maps.enable_mapo: Method to reenable the instances"""
        try:
            maps = kwargs.pop('maps')
        except KeyError:
            maps = self._disabled
        return self.get_maps(maps=self._disabled, **kwargs)

    def output(self, output, *args, **kwargs):
        """Save the figures.

        Save the figures into a file. Just setting output = `filename.pdf`
        will save all figures of the maps object to `filename.pdf`

        Parameters
        ----------
        output: string, 1D-array of strings or object

            - An object may be an open
                matplotlib.backends.backend_pdf.PdfPages instance
            - If string: %%(key)s will be replaced by the meta informations
            contained in the MapBase instances.
        *args
            - Either figures or MapBase instances which shall be saved (in
                case of MapBase, the corresponding figure will be saved)
            - 'tight' making the bbox_inches of the plot tight, i.e. reduce
                the output to the plot margins
        **kwargs
            - returnpdf: bool, optional
                Default: False. If True and all files are plotted into one
                matplotlib.backends.backend_pdf.PdfPages instance, this will be
                returned
            - any keyword that specifies the MapBase instances to save (see
                get_maps method). Only enabled mapos are considered (see
                disable_mapo and enable_mapo method).
            - any other keyword as specified in the matplotlib.pyplot.savefig
                function.

        Returns
        -------
        pdf : matplotlib.backends.backend_pdf.PdfPages
            if the `returnpdf` keyword is True

        See Also
        --------
        nc2map.Maps.make_movie: Method to create a movie over the timesteps"""
        # the docstring is extended by the plt.savefig docstring below
        saveops = {key: value for key, value in kwargs.items()
                   if key not in self.meta.keys() + ['mode', 'maps']}
        try:
            returnpdf = kwargs.pop('returnpdf')
        except KeyError:
            returnpdf = False
        if 'tight' in args:
            saveops['bbox_inches'] = 'tight'
            args = tuple([arg for arg in args if arg != 'tight'])
        kwargs = {key: value for key, value in kwargs.items()
                  if key in self.meta.keys() + ['mode', 'maps']}
        if not args:
            figs = self.get_figs(**kwargs).keys()
        elif isinstance(args[0], MapBase):
            labels = self.get_label_dict(*args)
            figs = self.get_figs(**labels)
        else:
            figs = args
        no_new = hasattr(output, 'savefig')
        if isinstance(output, str) or no_new:
            if no_new or output[-4:] in ['.pdf', '.PDF']:
                if not no_new:
                    output = self._replace(output, *figs, delimiter='-')
                pdf = output if no_new else PdfPages(output)
                for fig in figs:
                    pdf.savefig(fig, **saveops)
                if not no_new and not returnpdf:
                    self.logger.info('Saving plot to %s', output)
                    pdf.close()
                    return None
                else:
                    return pdf
            else:
                strout = output
                output = []
            for fig in figs:
                output = self._replace(output, fig, delimiter='-')
        else:
            pass
        # test output
        try:
            if len(np.shape(output)) > 1:
                raise ValueError(
                    'Output array must be a 1D-array!')
            if len(figs) != len(output):
                raise ValueError((
                    'Length of output names (%i) does not fit to the number '
                    ' of figures (%i).') % (len(output), len(figs)))
        except TypeError:
            raise TypeError((
                'Output names must be either a string or an 1D-array of '
                'strings!'))
        for fig in figs:
            fig.savefig(output[figs.index(fig)], **saveops)
            self.logger.info('Plot saved to %s', output[figs.index(fig)])
        return output


    def make_movie(self, output, fmt={}, onecbar={}, steps=None,
                   checklen=False, calc_bounds=True, *args, **kwargs):
        """Create a movie of the maps

        Method to create a movie out of the specified
        :class:`~nc2map.mapos.MapBase` instances with the specified (or
        current) formatoptions.

        Parameters
        ----------
        output: string or 1D-array of strings.

            - If string: meta attributes (see :attr:`meta`) will be replaced
                by the attributes contained in the figures.
            - If 1D-array: The length of the array must fit to the specified
                figures
        fmt: dict, optional
            Default: {}. Formatoptions (same hierarchical order as in the
            :meth:`~nc2map.Maps.addmap` method) where the values of the
            formatoption keywords need to be (iterable) 1D-arrays
        onecbar: dict or list of dictionaries, optional
            Default: {}. Same settings as for update_cbar function but (like
            `fmt`) with values of formatoption keywords being 1D-arrays with
            same length as number of `steps`
        steps: List of integers or None, optional
            If None, all timesteps in the nc-file are used for the movie.
            Otherwise set the timesteps as list
        checklen: bool, optional
            Default: False. If False, the formatoption keywords are simply
            iterated (and possibly repeated). If True, their lenghts have to
            match the number of steps.
        calc_bounds: bool, optional
            If True and bounds are set automatically, they are computed to
            match the whole period. Otherwise the current bounds are used.


        Further ``*args`` and ``**kwargs`` may be figures, MapBase instances,
        var=[...], etc. as used in the :meth:`~nc2map.Maps.get_figs` method to
        specify the figures to make movies of.
        Furthermore any valid keyword of the matplotlib.animation.FuncAnimation
        save function can be set. Default value for writer is 'imagemagick',
        extra_args are ['-vcodec', 'libx264'].

        Notes
        -----
        if filename is in the additional ``**kwargs``, it will replace the
        output variable.

        See Also
        --------
        nc2map.Maps.output: Method to create a single picture of the plots"""
        # docstring will be extended below
        # default options for kwargs if not in self.meta attribute, etc.
        defaults = {'dpi': None, 'fps': 3, 'writer': 'imagemagick',
                    'extra_args': ['-vcodec', 'libx264']}
        dimnames = self.meta.keys() + ['maps', 'mode', '_meta']
        # options as set in kwargs
        movieops = {key: value for key, value in kwargs.items()
                    if key not in dimnames}
        for key, value in defaults.items():
            movieops.setdefault(key, value)
        # delete options from kwargs
        kwargs = {key: value for key, value in kwargs.items()
                  if key in dimnames}
        # reset output to 'filename' in movieops if given
        if 'filename' in movieops:
            output = movieops.pop('filename')

        fmt = self._setupfmt(fmt)

        figs = self.get_figs(*args, **kwargs)

        for fig in figs:
            if isinstance(output, str):
                out = self._replace(output, fig, delimiter='-')
            else:
                out = output[i]
            maps = figs[fig]
            cbars = self.get_cbars(*maps)

            if steps is None:
                steps = range(len(maps[0].reader.time))

            # check lengths
            if checklen:
                for name in [mapo.name for mapo in maps
                             if mapo.name in fmt]:
                    try:
                        valuelens = map(lambda x: len(x) != len(steps),
                                        fmt[name].values())
                    except TypeError, e:
                        print(
                            "Could not estimate lengths. To use iterables set "
                            "checklon to False.\n")
                        raise e
                    if any(valuelens):
                        wrongkeys = [key for key, value in fmt[name].items()
                                     if len(value) != len(steps)]
                        raise ValueError((
                            "Lengths of arguments for %s do not match to "
                            "number of steps (%i)! Set checklen keyword to "
                            "False to use iterables.") % (
                                ', '.join(map(
                                    lambda x: '%s (%i)' % (x[0], x[1]),
                                    [(key, len(fmt[name][key]))
                                     for key in wrongkeys])),
                                len(steps)))

            # save bound options
            bounds = [getattr(mapo.fmt, 'bounds') for mapo in maps]
            windbounds = [getattr(mapo.wind.fmt, 'bounds') for mapo in maps
                          if hasattr(mapo, 'wind') and mapo.wind is not None]

            if calc_bounds:
                # modify bounds
                self.logger.info("Calculate bounds")
                # handle the mapobject coordinated by one single cbar
                for cbar in cbars:
                    cbar.set_global_bounds(
                        maps=set(set(maps) & set(cbar.maps)),
                        time=steps)
                self.set_global_bounds(maps=set(maps) | set(chain(*(
                    cbar.maps for cbar in cbars))), time=steps)

            # izip has no __len__ method which is required by the animation
            # function. Therefore we define a subclass and use it for the data
            # generator
            class myizip(izip):
                def __len__(self):
                    return len(steps)

            # data generator
            if cbars != []:
                data_gen = myizip(
                    myizip(*(mapo._moviedata(steps, **fmt[mapo.name])
                             for mapo in maps)),
                    myizip(*(cbar._moviedata(steps, **onecbar)
                             for cbar in cbars)))
            else:
                data_gen = myizip(*(mapo._moviedata(steps, **fmt[mapo.name])
                                    for mapo in maps))
            # run function
            if cbars != []:
                def runmovie(args):
                    return [mapo._runmovie(args[0][maps.index(mapo)])
                            for mapo in maps] + \
                           [cbar._runmovie(args[1][cbars.index(cbar)])
                            for cbar in cbars]
            else:
                def runmovie(args):
                    return [mapo._runmovie(args[maps.index(mapo)])
                            for mapo in maps]

            # movie initialization function
            def init_func():
                plt.figure(fig.gcf())
                plt.draw()
                #self.update({}, add=False, delete=False)
                #if self._cbars:
                    #self.update_cbar({}, add=False, delete=False)

            self.logger.info("Make movie")
            ani = FuncAnimation(fig, runmovie, frames=data_gen, repeat=True,
                                init_func=plt.draw)
            for mapo in maps: mapo._resize=False
            if out == 'show':
                plt.show(block=False)
                return ani

            ani.save(out, **movieops)
            self.logger.info('Saved movie to %s', out)
            # restore initial settings
            self.update(self._fmt[-1][0], add=False, delete=False,
                        todefault=True)

            if not self._fmt[-1][2]:
                for cbar in self._cbars:
                    cbar._removecbar()
            else:
                self.update_cbar(*self._fmt[-1][2], add=False,
                                    delete=False, todefault=True)
            return


    def reinit(self, *args, **kwargs):
        """Reinitialize the specified maps

        This method makes the specified :class:`~nc2map.mapos.MapBase`
        instances in the :attr:`~nc2map.Maps.maps` to get the data
        from the reader (may be useful if you changed the reader)

        ``*args`` and ``**kwargs`` are determined by the nc2map.Maps.get_map
        method"""
        maps = self.get_maps(*args, **kwargs)
        for mapo in maps:
            mapo._reinitialize = 1
        self.update(maps=maps)

    def update(self, fmt={}, add=True, delete=True, todefault=False,
               force=False, windonly=False, **kwargs):
        """Update the MapBase instances.

        This method can be used for the interactive usage of the
        :class:`~nc2map.Maps` instance to update the specified
        :class:`~nc2map.mapos.MapBase` instances

        Parameters
        ----------
        fmt: dictionary
            The same shape and options like in the :meth:`~nc2map.Maps.addmap`
            method.
        add: bool
            Default: True. Adds the new formatoptions to old formatoptions
            allowing a undoing via the :meth:`~nc2map.Maps.undo` method
        delete: bool
            Default: True. Deletes the newer formatoptions for the
            :meth:`~nc2map.Maps.redo` method.
        todefault: bool
            Default: False. Sets all formatoptions which are not specified by
            `fmt` or ``**kwargs`` to default.
        force: bool
            By default the formatoption keywords whose values correspond to
            what is already set, are removed if not `force` is True.
        windonly: bool
            If True, ``**kwargs`` and `fmt` are only affecting the WindPlot
            instances of the FieldPlot instances (same as::

                fmt={'windplot': {key :val}}

            for a key, value pair in `fmt` or ``**kwargs``.

        **kwargs
            may be any valid formatoption keyword or a key from the meta
            attribute to specifically select :class:`~nc2map.mapos.MapBase`
            instances (see :meth:`~nc2map.Maps.get_maps` method).

        Notes
        -----
        You have to use the `fmt` keyword if you want to update the dimensions
        (e.g. time, etc.). Otherwise it will be regarded as a specifier to
        select the corresponding MapBase instances (see below). In other
        words::

            mymaps.update(fmt={'time': 1}, title='test')

        will update all MapBase instances to time=1 and title='test', whereas::

            mymaps.update(time=1, title='test')

        will only update the title of the MapBase with time=1 to title='test'

        See Also
        --------
        nc2map.Maps.update_lines: Method to update the LinePlots
        nc2map.Maps.update_cbar: Method to update shared colorbars"""
        # if not deepcopied, the update in the next line will use previous fmts
        # given to the update function
        fmt = deepcopy(fmt)
        fmt.update({key: value for key, value in kwargs.items()
                    if key not in self.meta.keys() + ['maps', '_meta']})
        fmt = self._setupfmt(fmt)
        if windonly:
            get_func = self.get_winds
        else:
            get_func = self.get_maps
        maps = get_func(mode='maps',
            **{key: value for key, value in kwargs.items()
               if key in self.meta.keys() + ['maps', '_meta']})
        # update maps
        for mapo in maps:
            mapo.update(todefault=todefault, force=force, **fmt[mapo.name])
        # update figure window title and draw
        for cbar in self.get_cbars(*maps):
            cbar._draw_colorbar()
        for fig in self.get_figs(*maps):
            plt.figure(fig.number)
            self._set_window_title(fig)
            # if it is part of a cbar, it has already been drawn above
            plt.draw()
        # add to old fmts
        if add:
            self._fmt.append(self.asdict('maps', 'lines', 'cbars'))
        # delete new fmts
        if delete:
            self._newfmt = []
        del fmt

    def set_global_bounds(self, maps=None, time=slice(None)):
        """Calculate colorbar bounds considering all time steps

        Sets the bounds in the specified :class:`~nc2map.mapos.MapBase`
        instances to the limits over all timesteps

        Parameters
        ----------
        maps: list
            List of MapBase instances. If None, self.maps is used
        time: iterable or slice
            Alternate time slice to use in the calculation"""
        # now handle the rest of the mapobjects
        if maps is None:
            maps = self.maps
        for mapo in maps:
            boundsnames = ['rounded', 'sym', 'minmax', 'roundedsym']
            if (isinstance(mapo.fmt.bounds, tuple)
                    and isinstance(mapo.fmt.bounds[0], str)):
                if isinstance(mapo, FieldPlot):
                    if (mapo.fmt.bounds[0] in boundsnames
                            and len(mapo.fmt.bounds) == 2):
                        boundsdata = map(
                            lambda x: (np.ma.min(x), np.ma.max(x)),
                            (data[:] for data in mapo.gen_data(
                                time=time)))
                        boundsdata = np.ma.array(
                            boundsdata, mask=np.isnan(boundsdata))
                        mapo.fmt.bounds = returnbounds(
                            boundsdata[:], mapo.fmt.bounds)
                    elif (mapo.fmt.bounds[0] in boundsnames
                            and len(mapo.fmt.bounds) == 3):
                        mapo.fmt.bounds = returnbounds(np.ma.concatenate(
                            tuple(data[:] for data in mapo.gen_data(
                                time=steps))), mapo.fmt.bounds)
                if (isinstance(mapo, WindPlot)
                    or (hasattr(mapo, 'wind')
                        and mapo.wind is not None
                        and mapo.wind._bounds is not None)):
                    if isinstance(mapo, WindPlot):
                        wind = mapo
                    else:
                        wind = mapo.wind
                    if wind.fmt.bounds[0] in boundsnames:
                        try:
                            wind.fmt.color = wind.set_bounds(wind.fmt.color,
                                                             time=steps)
                        except (TypeError, ValueError):
                            pass

    def update_lines(self, fmt={}, add=True, delete=True,
                     todefault=False, **kwargs):
        """Function to update the lines

        This method can be used for the interactive usage of the
        :class:`~nc2map.Maps` instance to update the specified
        :class:`~nc2map.mapos.LinePlot` instances

        Parameters
        ----------
        fmt: dictionary
            the same shape and options like in the addline method
        add: bool
            Default: True. Adds the new formatoptions to old formatoptions
            allowing a undoing via the :meth:`~nc2map.Maps.undo` method
        delete: bool
            Default: True. Deletes the newer formatoptions for the
            :meth:`~nc2map.Maps.redo` method.
        todefault: bool
            Default: False. Sets all formatoptions which are not specified by
            `fmt` or ``**kwargs`` to default.
        **kwargs
            may be any valid formatoption keyword or a key from the meta
            attribute to specifically select :class:`~nc2map.mapos.MapBase`
            instances (see :meth:`~nc2map.Maps.get_maps` method).

        See Also
        --------
        nc2map.Maps.update: Method to update the maps
        nc2map.Maps.update_cbar: Method to update shared colorbars"""
        from copy import deepcopy
        # if not deepcopied, the update in the next line will use previous fmts
        # given to the update function
        fmt = copy(fmt)
        dims_identifiers = dict(item for item in kwargs.items()
                                if item[0] in self.meta.keys())
        for dim in dims_identifiers:
            del kwargs[dim]
        fmt.update(kwargs)
        lines = self.get_maps(mode='lines', **dims_identifiers)
        names = [line.name for line in lines]
        default_fmt = {key: val for key, val in fmt.items()
                       if key not in names}
        final_fmt = {name: default_fmt for name in names}
        for name, name_dict in final_fmt.items():
            name_dict.update(fmt.get(name, {}))

        # update maps
        for line in lines:
            line.update(todefault=todefault, **final_fmt[line.name])
        for fig in self.get_figs(name=names, mode='lines'):
            plt.figure(fig.number)
            self._set_window_title(fig)
            # if it is part of a cbar, it has already been drawn above
            plt.draw()
        # add to old fmts
        if add:
            self._fmt.append(self.asdict('maps', 'lines', 'cbars'))
        # delete new fmts
        if delete:
            self._newfmt = []
        del fmt

    def addline(self, *args, **kwargs):
        # docstring is set below to be equal to MapsManager.addline method
        try:
            add = kwargs.pop('add')
        except KeyError:
            add = True
        try:
            delete = kwargs.pop('delete')
        except KeyError:
            delete = True
        super(Maps, self).addline(*args, **kwargs)
        # add to old fmts
        if add:
            # reset old fmts (for function undo)
            self._fmt = [self.asdict('maps', 'lines', 'cbars')]
        if delete:
            # reset future fmts (for function redo)
            self._newfmt = []

    def reset(self, num=0, fromscratch=False, ax=None, sort=None,
              sortlines=False, **kwargs):
        """Reinitializes the instance

        Reinitializes the :class:`Maps` instance, even if
        :meth:`~nc2map.Maps.undo` method fails.

        Parameters
        ----------
        num: int
            Number of formatoptions (like :meth:`~nc2map.Maps.undo` method).
            0 is current, -1 the one before (often the last one working), etc.
        fromscratch: bool
            If False, only figures will be closed and recreated (if ax is not
            None) or the axes will be reset if ax is None. If True the whole
            Maps instance will be closed and reopend by loading the data from
            the readers
        ax: tuple, subplot or list of subplots
            see :meth:`~nc2map.Maps.addmap` method. Specify the subplots to
            plot on
        sort: string or list of strings
            see :meth:`~nc2map.Maps.addmap` method. Specifies how the
            :class:`nc2map.mapos.MapBase` instances are sorted to the subplots
        **kwargs
            anything that is passed to the nc2map.subplots function to create
            the figures (e.g. figsize).

        See Also
        --------
        nc2map.Maps.undo: less hard undo function"""
        if not self._fmt:
            raise ValueError('Impossible option')
        if num > 0 and num >= len(self._fmt)-1:
            raise ValueError(
                'Too high number! Maximal number is %i' % len(self._fmt)-1)
        elif num < 0 and num < -len(self._fmt):
            raise ValueError(
                'Too small number! Minimal number is %i' % -len(self._fmt)+1)

        name_dicts = self.name_dicts
        # try to save readers
        linenames = self.linenames
        mapnames = self.mapnames
        vlst = self.vlst
        readers = [mapo.reader for mapo in self.maps]
        # reset cbars
        self.removecbars()
        self._cbars = []
        # set new subplots
        if ax is not None or fromscratch:
            maps_subplots = []
            lines_subplots = []
            if ax is None:
                subplots = []
                for fig, maps in self.get_figs().items():
                    shape = maps[0]._get_axes_shape()
                    try:
                        maps[0]._ax._AxesWrapper__init_kwargs.pop('num')
                    except KeyError:
                        pass
                    maps[0]._ax._AxesWrapper__init_kwargs.update(kwargs)

                    this_subplots = self._make_subplots(
                        shape, len(maps), *maps[0]._ax._AxesWrapper__init_args,
                        **maps[0]._ax._AxesWrapper__init_kwargs)
                    mapos = self.get_maps(name=[mapo.name for mapo in maps],
                                        mode='maps')
                    maps_subplots += [this_subplots[mapo._get_axes_num()-1]
                                    for mapo in mapos]
                    linos = self.get_maps(name=[line.name for line in maps],
                                        mode='lines')
                    lines_subplots += [this_subplots[line._get_axes_num()-1]
                                    for line in linos]
                    subplots += this_subplots
            else:
                subplots = self._make_subplots(
                    ax, len(self.get_maps()), **kwargs)
                maps_subplots = subplots[:len(self.maps)]
                lines_subplots = subplots[len(self.maps):]
        else:
            maps_subplots = [mapo.ax for mapo in self.maps]
            lines_subplots = [line.ax for line in self.lines]
            subplots = maps_subplots + lines_subplots
            for axes in maps_subplots + lines_subplots:
                axes.clear()
        if sort is not None and all(name in sort for name in self.names):
            maps_subplots = [subplots[i] for i, name in enumerate(sort)
                             if name in mapnames]
            lines_subplots = [subplots[i] for i, name in enumerate(sort)
                             if name in linenames]
            for name in linenames:
                sort.remove(name)
        # close the Maps instance
        if fromscratch:
            try:
                self.logger.debug("Try to close data and figures...")
                self.close(30, mode='maps')
                self.close(15, mode='lines')
            except (AttributeError, KeyError):
                self.logger.debug("Could not close figures", exc_info=1)
                warn("Could not close the figures but anyway will draw new "
                     "figures")
        elif ax is not None:
            try:
                self.logger.debug("Try to close figures...")
                self.close(3, remove=False)
            except (AttributeError, KeyError):
                self.logger.debug("Could not close figures", exc_info=1)
                warn("Could not close the figures but anyway will draw new "
                     "figures")
        # set new figures
        # change names sorting
        if sort is not None:
            name_dicts = self._setupnames(name_dicts, vlst, sort, rename=False)
            if not fromscratch:
                self._resort_maps(name_dicts.keys())
        # set up MapBase instances
        for subplot, name_dict, reader in izip_longest(
                maps_subplots, name_dicts.items(), readers):
            self._setupfigs(dict([name_dict]), fmt=self._fmt[num-1][0],
                            subplots=[subplot], reader=reader,
                            fromscratch=fromscratch)
        # set up lineplots
        lines_subplots = iter(lines_subplots)
        for line in self.lines:
            line.ax = next(lines_subplots)
            line.update(**self._fmt[num-1][1][line.name])

        if self.plot:
            self.logger.info("Setting up projections...")
            for mapo in self.get_maps(mode='maps'):
                mapo._make_plot = 1
                if hasattr(mapo, 'wind') and mapo.wind is not None:
                    mapo.wind._make_plot = 1
                mapo._setupproj()

            self.logger.info("Making plots...")
            self.make_plot(mode='maps')

            for fig in self.get_figs():
                plt.figure(fig.number)
                self._set_window_title(fig)
                plt.draw()

        if self._fmt[num-1][2]:
            self.update_cbar(*self._fmt[num-1][2], add=False, delete=False)
        # shift to new fmt
        if num != 0:
            self._newfmt = self._fmt[num:] + self._newfmt
        if num <= 0:
            self._fmt.__delslice__(len(self._fmt)+num, len(self._fmt))
        else:
            self._fmt.__delslice__(num, len(self._fmt))

    def update_cbar(self, *args, **kwargs):
        """Update or create a shared colorbar

        Shared colorbars may be used to control the appearance of multiple
        :class:`~nc2map.mapos.MapBase` instances at once. They are especially
        useful if you use automatically calculated bounds

        Parameters
        ----------
        *args
            dictionaries::

                onecbar = {'meta_key':..., 'fmt_key':...}

            'meta_key' may be anything suitable for the
            :meth:`~nc2map.Maps.get_maps` method). `fmt_key` may be any
            formatoption keyword controlling the colorbar and colormap
            aesthetics.
        add: bool
            Default: True. Adds the new formatoptions to old formatoptions
            allowing a undoing via the :meth:`~nc2map.Maps.undo` method
        delete: bool
            Default: True. Deletes the newer formatoptions for the
            :meth:`~nc2map.Maps.redo` method.
        todefault: bool
            Default: False. Sets all formatoptions which are not specified by
            `fmt` or ``**kwargs`` to default.
        **kwargs
            additionally to `add`, `delete` and `todefault` keywords, any
            formatoption keywords for colorbars or anything for the
            :meth:`~nc2map.Maps.get_maps` method is possible. Those will then
            be treated like a single dictionary (this is just to avoid nasty
            typing of :, {}, etc.).

        Notes
        -----
        If no colorbar with any of the specified dimensions is found, a new
        :class:`~nc2map.CbarManager` instance is created.

        See Also
        --------
        nc2map.Maps.update: Method for updating maps
        nc2map.Maps.update_lines: Method for updating lines"""
        dimnames = self.meta.keys() + ['_meta', 'maps']
        add = kwargs.get('add', True)
        delete = kwargs.get('delete', True)
        todefault = kwargs.get('todefault', False)
        plot = kwargs.get('plot', True)
        kwargs = {key: value for key, value in kwargs.items()
                  if key not in ['add', 'delete', 'todefault', 'plot']}
        if kwargs != {}:
            newops = list(args) + [kwargs]
        else:
            newops = list(args)
        if not newops:
            newops = [{}]
        # first set colorbars
        for cbarops in newops:
            if 'windplot' in cbarops:
                args = tuple(['wind'])
                cbarops.update(cbarops.pop('windplot'))
                wind = True
                get_func = self.get_winds
            else:
                args = ()
                wind = False
                get_func = self.get_maps
            dims = {
                key: cbarops[key] for key in dimnames if key in cbarops}
            for key in dims:
                del cbarops[key]
            # if no colorbars are set up to now and no specific var, time and
            # level options are set, make colorbars for each figure
            if not self._cbars and not dims:
                figs = self.get_figs(*args, mode='maps')
                for fig in figs:
                    self._cbars.append(CbarManager(
                        maps=figs[fig], fig=[fig], cbar={}, mapsin=self,
                        wind=wind))
            # now update colorbar objects or create them if they are not
            # existent
            dims['mode'] = 'maps'
            cbars = self.get_cbars(*args, **dims)
            if not cbars:
                self._cbars.append(CbarManager(
                    maps=get_func(**dims), cbar={}, mapsin=self, wind=wind,
                    fig=self.get_figs(*args, **dims).keys()))
                cbars = [self._cbars[-1]]
            # now draw and update colorbars
            for cbar in cbars:
                cbar.update(fmt=cbarops, todefault=todefault, plot=plot)
                if plot:
                    for fig in cbar.get_figs():
                        plt.figure(fig.number)
                        plt.draw()

        if add:
            self._fmt.append(self.asdict('maps', 'lines', 'cbars'))
        if delete:
            self._newfmt = []

    def get_cbars(self, *args, **kwargs):
        """Get the cbars

        Method to return the shared cbars (:class:`~nc2map.CbarManager`) of
        the specified input.

        Parameters
        ----------
        *args
            instances of :class:`~nc2map.mapos.MapBase`, figures or
            :class:`~nc2map.CbarManagers`
        **kwargs
            any keyword of the :meth:`~nc2map.Maps.get_maps` method (i.e. by
            the keys in the meta attribute)

        Returns
        -------
          colorbars: list
            list of :class:`~nc2map.CbarManagers`"""
        maps = []
        args = list(args)
        cbars = [cbar for cbar in args if isinstance(cbar, CbarManager)]
        kwargs['mode'] = 'maps'
        if not args:
            maps = self.get_maps(**kwargs)
        elif args == ['wind']:
            maps = self.get_winds(**kwargs)
        else:
            figs = self.get_figs(*(arg for arg in args if arg == 'wind'),
                                 mode='maps')
            for fig in figs:
                if fig in args:
                    maps = maps + figs[fig]
                    args.remove(fig)
        maps += list(
            arg for arg in args if not isinstance(arg, CbarManager))
        cbars = cbars + [cbar for cbar in self._cbars
                         if any(mapo in cbar.maps for mapo in maps)]
        return cbars

    def removecbars(self, *args, **kwargs):
        """Method to remove share colorbars

        This method removes the specified :class:`~nc2map.CbarManager`
        instances from the plot and reenables the manual setting of the bounds
        via the :meth:`~nc2map.Maps.update` method.

        Parameters
        ----------
        *args
            instances of :class:`~nc2map.mapos.MapBase`, figures or
            :class:`~nc2map.CbarManagers`
        **kwargs
            any keyword of the :meth:`~nc2map.Maps.get_maps` method (i.e. by
            the keys in the meta attribute)
        """
        cbars = self.get_cbars(*args, **kwargs)
        maps = chain(*(cbar.maps for cbar in cbars))
        for cbar in cbars:
            cbar._removecbar()
            self._cbars.remove(cbar)
        # draw figures
        for fig in self.get_figs(maps=list(maps)):
            plt.figure(fig.number)
            plt.draw()


    def undo(self, num=-1):
        """Undo the changes made.

        Parameters
        ----------
        num: int
            number of changes to go back.

        See Also
        --------
        nc2map.Maps.redo: Redo formatoption changes that were undone by this
            method"""
        if not self._fmt or len(self._fmt) == 1:
            raise ValueError('Impossible option')
        if num > 0 and num >= len(self._fmt)-1:
            raise ValueError(
                'Too high number! Maximal number is %i' % len(self._fmt)-1)
        elif num < 0 and num < -len(self._fmt):
            raise ValueError(
                'Too small number! Minimal number is %i' % (-len(self._fmt)+1))
        if not self._fmt[num-1][2]:
            self.removecbars()
        self.update(self._fmt[num-1][0], add=False, delete=False,
                    todefault=True)
        self.update_lines(add=False, delete=False, todefault=True,
                          **self._fmt[num-1][1])
        if self._fmt[num-1][2]:
            self.update_cbar(*self._fmt[num-1][2], add=False, delete=False,
                             todefault=True)
        # shift to new fmt
        self._newfmt = self._fmt[num:] + self._newfmt
        if num <= 0:
            self._fmt.__delslice__(len(self._fmt)+num, len(self._fmt))
        else:
            self._fmt.__delslice__(num, len(self._fmt))

    def redo(self, num=1):
        """Redo the changes made

        Parameters
        ----------
        num: int
            number of changes to use.

        See Also
        --------
        nc2map.Maps.undo: Undo formatoption changes"""
        if not self._newfmt:
            raise ValueError('Impossible option')
        if num > 0 and num > len(self._newfmt):
            raise ValueError(
                'Too high number! Maximal number is %i' % len(self._newfmt))
        elif num < 0 and num < -len(self._newfmt):
            raise ValueError(
                'Too small number! Minimal number is %i' % (
                    -len(self._newfmt)-1))
        if not self._newfmt[num-1][2]:
            self.removecbars()
        self.update(self._newfmt[num-1][0], add=False, delete=False,
                    todefault=True)
        self.update_lines(add=False, delete=False, todefault=True,
                          **self._newfmt[num-1][1])
        if self._newfmt[num-1][2]:
            self.update_cbar(*self._newfmt[num-1][2], add=False, delete=False,
                             todefault=True)
        # shift to old fmt
        self._fmt += self._newfmt[:num]
        if num > 0:
            self._newfmt.__delslice__(0, num)
        else:
            self._newfmt.__delslice__(0, len(self._newfmt)+num)

    def close(self, num=0, remove=True, **kwargs):
        # docstring is set below
        cbars = self.get_cbars(**kwargs)
        super(Maps, self).close(num=num, remove=remove, **kwargs)
        if not num % 2 or not num % 3:
            for cbar in self.get_cbars(*cbars):
                if not set(self.maps) & set(cbar.maps):
                    cbar.close(num=num)
                    self._cbars.remove(cbar)

    def asdict(self, *args, **kwargs):
        """Get the current formatoptions

        This method gives the formatoptions of the specified
        :class:`~nc2map.mapos.MapBase` instances,
        :class:`~nc2map.CbarManager` and :class:`~nc2map.mapos.LinePlot`
        instances as a list of dictionaries.

        Parameters
        ----------
        *args
            - 'maps' for MapBase instances in :attr:`nc2map.Maps.maps`
                attribute
            - 'lines' for SimplePlot instances in :attr:`nc2map.Maps.lines`
                attribute
            - 'cbars' to return only the dictionary controlling the
                CbarManager instances (see onecbar in the initialization)
        **kwargs
            Any key suitable for the :meth:`~nc2map.Maps.get_maps` method."""
        fmt = {}
        returns = []
        cbars = []
        if not args or any(x in args for x in ['maps', 'lines', 'frominit']):
            returns += super(Maps, self).asdict(*args, **kwargs)
        if 'cbars' in args:
            for cbar in self.get_cbars(**kwargs):
                cbars.append(cbar.asdict())
                cbars[-1]['name'] = cbar.names
            returns.append(cbars)
        return tuple(returns)

    def addmap(self, ncfile, names=None, vlst=None,  ax=(1, 1), sort=None,
               fmt=None, onecbar=False, u=None, v=None, mode=None, dims={},
               windonly=False, plot=True, add=True, delete=True, meta={},
               **kwargs):
        """
        onecbar: bool, dict or tuple of dictionaries
            Default: False. If True, one colorbar will be drawn for each
            figure. If dictionary: the syntax is as follows::

                onecbar = {'meta_key':..., 'fmt_key':...}

            'meta_key' may be anything suitable for the
            :meth:`~nc2map.Maps.get_maps` method). `fmt_key` may be any
            formatoption keyword controlling the colorbar and colormap
            aesthetics.
        add: bool
            Default: True. Adds the new formatoptions to old formatoptions
            allowing a undoing via the :meth:`~nc2map.Maps.undo` method
        delete: bool
            Default: True. Deletes the newer formatoptions for the
            :meth:`~nc2map.Maps.redo` method.
        """
        # docstring is set below from MapsManager
        super(Maps, self).addmap(ncfile=ncfile, names=names, vlst=vlst, ax=ax,
                                 sort=sort, fmt=fmt, u=u, v=v, mode=mode,
                                 meta=meta, dims=dims, windonly=windonly,
                                 plot=plot, **kwargs)
        if onecbar is not False:
            if onecbar is True:
                self.update_cbar(
                    *(self.get_label_dict(fig) for fig in self.get_figs()),
                    add=False, delete=False, plot=plot)
            elif isinstance(onecbar, dict):
                self.update_cbar(onecbar, add=False, delete=False)
            else:
                self.update_cbar(*onecbar, add=False, delete=False)


        if add:
            # reset old fmts (for function undo)
            self._fmt = [self.asdict('maps', 'lines', 'cbars')]
        if delete:
            # reset future fmts (for function redo)
            self._newfmt = []

    def save(self, output=None, ask=True, ask_ax=True, ncnames=[]):
        """Saves the settings of the Maps instance (not the data!)

        This method creates a pickle file in order to reinitialize the
        :class:`~nc2map.Maps` instance with the :func:`~nc2map.load` function

        Parameters
        ----------
        output: str
            Name of the pickle file (e.g. 'mymaps.pkl') or None if only the
            dictionary shall be returned and no file shall be created
        ask: bool
            Default: True. If True and if the initialization keywords of a
            reader can not be determined (like it is the case for
            ArrayReader instances), it will be asked for a filename to dump
            the reader into a NetCDF file.
        ask_ax: bool
            Default: True. If True and if the initialization keywords of a
            subplot (i.e. the subplot shape of a figure) can not be
            determined (like it is the case for user defined subplots from
            pyplot.subplots function), it will be asked for the subplot shape
            and subplot number
        ncnames: List of strings
            E.g. 'my-ncfile.nc'. NetCDF files that shall be used if the reader
            is not already dumped (like it is the case for
            :class:`~nc2map.readers.ArrayReader` instances).

        Returns
        -------
        settings_dict: dict
            Dictionary containing all informations to reload the instance with
            the :func:`~nc2map.load` function

        See Also
        --------
        nc2map.load: Function to load from the settings_dict or the pickle
        file that you just saved
        """
        # extract reader settings
        self.logger.debug('Save Maps instance ...')
        self.logger.debug('    Extract reader settings')
        readers_dict = OrderedDict(
            [(rd, 'reader%i' % i) for i, rd in enumerate(self.get_readers())])
        reader_settings = {}
        ncnames = iter(ncnames)
        for reader, name in readers_dict.items():
            reader_settings[name] = [reader.__class__.__name__, [], {}]
            try:
                reader_settings[name][1] = reader._init_args
            except AttributeError:
                pass
            try:
                reader_settings[name][2] = reader._init_kwargs
            except AttributeError:
                pass
            if not any(reader_settings[name][1:]):
                try:
                    fname = next(ncnames)
                except StopIteration:
                    if ask:
                        fname = raw_input(
                            "Could not estimate how to initialize the reader. "
                            "Enter a file name as string or nothing to "
                            "ignore it.\n")
                if fname:
                    reader.dump_nc(fname)
                    reader_settings[name][0] = 'NCReader'
                    reader_settings[name][1] = [fname]

                else:
                    reader_settings[name] = None
            elif not any(reader_settings[name][1:]) and not ask:
                warn("Could not estimate how to initialize the reader of %s.",
                     ', '.join(mapo.name for mapo in self.get_readers()[
                        reader]))
                reader_settings[name] = None
            if not reader_settings[name] is None:
                reader_settings[name][2].update({
                    'timenames': reader._timenames,
                    'levelnames': reader._levelnames,
                    'lonnames': reader._lonnames,
                    'latnames': reader._latnames})
        # extract figure settings
        self.logger.debug('    Extract figure settings')
        figures_dict = self.get_figs()
        figure_settings = OrderedDict()
        for fig, maps in figures_dict.items():
            try:
                mode = 'ask' if ask_ax else 'raise'
                figure_settings[fig.number] = [
                    maps[0]._get_axes_shape(mode=mode),
                    maps[0]._ax._AxesWrapper__init_args,
                    maps[0]._ax._AxesWrapper__init_kwargs]
            except ValueError:
                warn("Could not estimate figure options for %s" %
                     ', '.join(mapo.name for mapo in maps))
                figure_settings[fig.number] = None

        self.logger.debug('    Extract maps settings')
        mapo_settings = OrderedDict((mapo.name, {}) for mapo in self.maps)
        mode = 'ask' if ask_ax else 'ignore'
        for mapo, mdict in zip(self.maps, mapo_settings.values()):
            self.logger.debug('        mapo %s', mapo.name)
            mdict['fmt'] = mapo.asdict()
            mdict['dims'] = mapo.dims
            mdict['fig'] = mapo.ax.get_figure().number
            mdict['reader'] = readers_dict[mapo.reader]
            mdict['meta'] = mapo._meta
            try:
                mdict['num'] = mapo._get_axes_num(mode=mode)
            except ValueError:
                pass
            mdict['class'] = mapo.__class__.__name__
            mdict['name'] = mapo.name

        self.logger.debug('    Extract shared info...')
        share_settings = {mapo.name: mapo.asdict(shared=True)['_shared']
                          for mapo in self.maps}

        self.logger.debug('    Extract line settings')
        line_settings = OrderedDict((line.name, {}) for line in self.lines)
        for line, ldict in zip(self.lines, line_settings.values()):
            self.logger.debug('        line %s', line.name)
            ldict['init'] = line.asdict()
            ldict['fig'] = line.ax.get_figure().number
            ldict['reader'] = readers_dict[line.reader]
            ldict['num'] = line._get_axes_num(mode='ignore')
            ldict['class'] = line.__class__.__name__
            ldict['name'] = line.name
            try:
                ldict['meta'] = line._meta
            except AttributeError:
                pass

        self.logger.debug('    Extract cbars')
        cbar_settings = self.asdict('cbars')[0]
        self.logger.debug('Dump to %s', output)
        out_dict = {'readers': reader_settings,
                    'figures': figure_settings,
                    'maps': mapo_settings,
                    'lines': line_settings,
                    'cbars': cbar_settings,
                    'share': share_settings}
        if output:
            with open(output, 'w') as f:
                pickle.dump(out_dict, f)
        return out_dict

    def dump_nc(self, output, maps=None, full_data=False, mask_data=True,
                **kwargs):
        """Method for creating a NetCDF file out of the given maps.

        Parameters
        ----------
        maps: list of MapBase instances
            If None, defaults to the list in :attr:`~nc2map.Maps.maps` and
            :attr:`~nc2map.Maps.lines`
        full_data: bool
            Default: False. If True, the full data as stored in the readers
            are used. Otherwise only the current time and level step is used.
        mask_data: bool
            Default: True. If True, the formatoption masking options (including
            lonlatbox, mask, maskbelow, etc.) is used. Otherwise the full field
            as stored in the corresponding :class:`~nc2map.readers.ReaderBase`
            instance is used.
        **kwargs
            anything that is passed to the
            :meth:`~nc2map.readers.ReaderBase.dump_nc` method

        Returns
        -------
        nco : object
            :class:`~nc2map.readers.NCReader` instance that is created

        Warnings
        --------
        The readers of the `maps` are merged into one file, so make sure that
        they match according to the :meth:`~nc2map.readers.ReaderBase.merge`
        method

        Notes
        -----
        By default the `nco` is closed, unless you set close=False

        See Also
        --------
        nc2map.readers.ReaderBase.dump_nc: Basic method that is used"""
        if maps is None:
            maps = self.get_maps()
        self.logger.debug("Dump mapos %s to NetCDF file...",
                          ', '.join(mapo.name for mapo in maps))
        if len(maps) == 1:
            try:
                return maps[0].extract_in_reader(
                    full_data=full_data, mask_data=mask_data).dump_nc(
                        output=output, **kwargs)
            except TypeError:  # line does not take the mask_data keyword
                return maps[0].extract_in_reader(full_data=full_data).dump_nc(
                    output=output, **kwargs)
        readers = []
        for mapo in maps:
            self.logger.debug("Extracting mapo %s", mapo.name)
            try:
                self.logger.debug("Try with mask_data")
                readers.append(mapo.extract_in_reader(
                    full_data=full_data, mask_data=mask_data))
            except TypeError:  # line does not take the mask_data keyword
                self.logger.debug("Failed. --> Assume line", exc_info = True)
                readers.append(mapo.extract_in_reader(
                    full_data=full_data))
        return readers[0].merge(*readers[1:]).dump_nc(
            output=output, **kwargs)

    def get_label_dict(self, *args, **kwargs):
        """Returns dictionary with meta attributes

        Parameters
        ----------
        *args
            instances of :class:`~nc2map.mapos.MapBase`, figures,
            :class:`~nc2map.mapos.LinePlot`, :class:`~nc2map.CbarManager`, etc.
        delimiter: str
            string that shall be used for separating values in a string, if
            multiple meta values are found in the specified input. If not given
            (or None), sets will be returned, not strings

        Returns
        -------
        meta: dict
            concatenated dictionary with meta informations as sets or as string
            (if delimiter keyword is given)"""
        args = list(args)
        if not 'lines' in args:
            for cbar in self.get_cbars('wind') + self.get_cbars():
                if cbar in args:
                    args += cbar.maps
                    args.remove(cbar)
        return super(Maps, self).get_label_dict(*args, **kwargs)

    # ------------------ modify docstrings here --------------------------
    evaluate.__doc__ %= ', '.join(map(repr, evaluatorsdict.keys()))
    get_evaluator.__doc__ %= ', '.join(map(repr, evaluatorsdict.keys()))
    close.__doc__ = MapsManager.close.__doc__
    addline.__doc__ = MapsManager.addline.__doc__
    addmap.__doc__ = MapsManager.addmap.__doc__[
        :MapsManager.addmap.__doc__.find('fmt: ')] \
            + addmap.__doc__ + \
                MapsManager.addmap.__doc__[
                    MapsManager.addmap.__doc__.find('fmt: '):]
