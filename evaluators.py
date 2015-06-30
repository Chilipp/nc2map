# -*- coding: utf-8 -*-
"""Evaluators module of the nc2map package.

This module (currently) contains two evaluators of MapBase instances plus the
base class.

Classes are:
  - EvaluatorBase: (Subclass of MapsManager). Base class for all evaluators
      (this class itself does nothing but provide the general framework for
      evalutors)
  - ViolinEvaluator: Class that makes violion plots for selected regions
  - FldMeanEvaluator: Class that calculates and plots the field mean of a
      scalar variable
"""
import re
from itertools import product, cycle, repeat, imap, islice, chain, izip_longest
import logging
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from ._maps_manager import MapsManager
from .mapos import MapBase, ViolinPlot, LinePlot
from .formatoptions import _get_fmtkeys_formatted, get_fmtdocs, SimpleFmt
from .warning import warn


class EvaluatorBase(MapsManager):
    """Base class for evaluators of MapBase instances"""

    def __init__(self, maps, name='evaluator', mapsin=None):
        self.set_logger()
        self.name = name
        self.lines = []  # list pf SimplePlot instances
        try:
            self.maps = list(maps)  # list of MapBase instances
        except TypeError:
            self.maps = [maps]
        self._mapsin = mapsin  # Maps instance
        self.plot = True

    def update(self, fmt={}, **kwargs):
        # if not deepcopied, the update in the next line
        # will use previous fmts given to the update function
        fmt = deepcopy(fmt)
        fmt.update({key:value for key, value in kwargs.items() if key not in
                    ['names', 'vlst', 'times','levels']})
        fmt = self._setupfmt(fmt)
        maps = self.get_maps(**{key: value for key, value in kwargs.items() if
                                key in ['names', 'vlst', 'times', 'levels']})
        # update maps
        for mapo in maps: mapo.update(todefault=todefault,
                                      **fmt.get(mapo.name))

    def set_logger(self, name=None, force=False):
        """This function sets the logging.Logger instance in the MapsManager
        instance.
        Input:
          - name: name of the Logger (if None: it will be named like
             <module name>.<class name>)
          - force: True/False (Default: False). If False, do not set it if the
              instance has already a logger attribute."""
        if name is None:
            try:
                name = '%s.%s.%s' % (self.__module__, self.__class__.__name__,
                                    self.name)
            except AttributeError:
                name = '%s.%s' % (self.__module__, self.__class__.__name__)
        if not hasattr(self, 'logger') or force:
            self.logger = logging.getLogger(name)
            self.logger.debug('Initializing...')

    def __repr__(self):
        return "<%s evaluator of %s>" % (self.__class__.__name__,
                                         ', '.join(self.mapnames))

    def __str__(self):
        return repr(self)[1:-1]


class ViolinEval(EvaluatorBase):
    """Class for plotting a violin plot with seaborn.

    See __init__ method for initialization.
    This class creates ViolinPlot instances that show violin plots for
    the specified MapBase instances in the specified regions.
    By default, the created lines are added to the mapsin instance, but are
    disabled. It is recommended to use the ViolinEval.update method for
    these lines instead of the Maps.update method."""
    def __init__(self, maps, name='violin', figsize=None, ax=None,
                 fmt={}, mapsin=None, names=None, regions={'Global': {}},
                 **kwargs):
        """Initialization function for violin plot
        Input:
          - maps: List of MapBase instances or MapBase instance. !!The number
                  of unmasked elements in the MapBase.data instances must
                  match!!
          - name: name of the ViolinEvaluator instance
          - regions: dictionary. Keys are names for the regions (use 'Global'
              for non-masking) values are mask values as for the FmtBase.mask
              property: %(mask)s. Per region, one subplot is created.
          - names: string or array. names that shall be used for the MapBase
              instances on the x-axis. If None, the MapBase.name attributes
              will be used. If string, this will be used for each MapBase
              instance, where strings like '%%(name)s' will replaced by the
              specific value in the MapBase.meta attribute
          - fmt: dictionary with formatoption keywords. The dictionary may
              contain any formatoption keyword of the SimpleFmt class (see
              below) or for a more detailed specification the regions keys
              with values being dictionaries. In other words the syntax of fmt
              is as follows:
                 fmt = {['<<<region>>>':{
                            [, 'keyword': ..., ...]
                            },]
                        ['keyword':..., 'keyword': ...,]
                        }
                 Keywords outside one of the 'regions' dictionaries will set
                 default formatoptions for all subplots.
          - ax: Axes instance to draw on (if None, a new figure with size
                figsize will be created and all axes are plotted into a
                subplot grid with 2-columns and 1 row). Otherwise it may be a
                list of subplots matching the number of regions, or a tuple
                (x, y) or (x, y, z
          - ax: matplotlib.axes.AxesSubplot instance or list of
                matplotlib.axes.AxesSubplot instances or tuple (x,y[,z])
                (Default: None).
                    -- If None, a new figure with size figsize will be created
                        and all axes are plotted into a subplot grid with
                        2-columns and 1 row
                    -- If ax is an axes instance (e.g. from nc2map.subplots())
                        or a list of axes instances, the data will be plotted
                        into these axes.
                    -- If ax is a tuple (x,y), figures will be created with x
                        rows and y columns of subplots. If ax is (x,y,z), only
                        the first z subplots of each figure will be used.
                In any case the number of subplots should match the number of
                regions, because every region is plotted into one subplot.
          - figsize: size of the figure if ax is None or a tuple
          - mapsin: Maps instance the evaluator belongs to.

        Other keywords may be keywords of the seaborn.violinplot
        function (by default color='coolwarm_r", linewidth=2 are set).
        Keywords of fmt may be (see nc2map.show_fmtdocs('xy') for details)
        """
        # docstring is extended below
        # update seaborn.violinplot keys
        import seaborn as sns
        self.name = name
        self.set_logger()
        regions = OrderedDict(regions)
        fmt = deepcopy(fmt)
        if sns.__version__ == '0.5.1':
            snskwargs = {'color': "coolwarm_r", 'lw': 2}
        else:
            snskwargs = {'palette': "coolwarm_r", 'lw': 2, 'orient': 'v'}
        myfmt = SimpleFmt()
        for key, val in kwargs.items():
            if key not in myfmt._default and key not in ['names']:
                snskwargs.update({key: kwargs.pop(key)})
            else:
                fmt.update({key: val})
        super(ViolinEval, self).__init__(maps=maps, name=name, mapsin=mapsin)
        if ax is None and len(regions) == 1:
            subplots = self._make_subplots((1,1), 1, figsize=figsize)
        elif ax is None:
            subplots = self._make_subplots(
                (1, len(regions)), len(regions), sharey=True, figsize=figsize)
        else:
            subplots = self._make_subplots(
                ax, len(self.maps), sharey=True, figsize=figsize)

        if not len(subplots) == len(regions):
           warn("Attention! Length of subplots (%i) from axes definition %s "
                "does not match to the number of regions (%i)!" % (
                    len(subplots), ax, len(regions)))

        if names is None:
            fmt.setdefault('xticklabels', '%(name)s')
        else:
            fmt.setdefault('xticklabels', names)
        try:
            fmt['xticklabels'] = [
                mapo._replace(fmt['xticklabels']) for mapo in self.maps]
        except TypeError:
            pass
        all_meta = (self.get_label_dict(*self.maps, delimiter=', ')
                for region in regions)
        fmt.setdefault('title', '%(region)s')
        fmt = self._setupfmt(fmt, regions)
        origmasks = [mapo.fmt.mask for mapo in self.maps]
        for subplot, meta, (region, mask) in izip_longest(
                subplots, all_meta, regions.items()):
            for mapo in self.maps:
                meta.update({'region': region})
                if region == 'Global':
                    mask = origmasks[self.maps.index(mapo)]
                mapo.update(mask=mask, plot=False)
            name = region + '_' + '_'.join(self.mapnames)
            self.lines.append(ViolinPlot(np.dstack(
                [mapo.data[:].compressed() for mapo in self.maps])[0],
                fmt=fmt[region], name=name, ax=subplot, mapsin=mapsin,
                snskwargs=snskwargs, meta=meta))
        for mapo, mask in zip(self.maps, origmasks):
            mapo.update(mask=mask, plot=False)
        self._set_ylabels()
        for fig, lines in self.get_figs(mode='lines').items():
            fig.canvas.set_window_title(
                'Figure ' + str(fig.number) + ': Violin plots of %s' % (
                    ', '.join([line.meta['region'] for line in lines])))

        if self._mapsin is not None:
            self._mapsin._disabled.update(self.lines)

    def update(self, fmt={}, snskwargs={}, **kwargs):
        # update formatoptions
        names = self.linenames
        fmt = self._setupfmt(fmt, names)
        for line in self.lines:
            line.update(fmt=fmt[line.name], snskwargs=snskwargs, **kwargs)

    def _setupfmt(self, fmt, names):
        new_fmt = {name: {} for name in names}
        for name in set(fmt) & set(names):
            new_fmt[name] = fmt[name]
        for key in set(fmt) - set(names):
            for name in names:
                new_fmt[name].setdefault(key, fmt[key])
        return new_fmt

    def _set_ylabels(self, ylabel=None):
        line = self.lines[0]
        if line.fmt.ylabel is None:
            if line.meta.get('long_name'):
                label = ylabel or '%(long_name)s'
            else:
                label = ylabel or '%(var)s'
            if line.meta.get('units') and not ylabel:
                label += ' [%(units)s]'
            line.update(ylabel=label)

    def close(self, num=0):
        for line in self.lines:
            line.close(num)

    # --------- modify docstrings here -------------
    __init__.__doc__   = __init__.__doc__ % get_fmtdocs() + \
        _get_fmtkeys_formatted('xy')


class FldMeanEvaluator(EvaluatorBase):
    """Class for calculating the and plotting the field mean of a scalar
    variable"""
    def __init__(self, maps, error=None, name='fldmean', ax=None,
                 fmt={}, pyplotfmt={}, mapsin=None, labels=None, names=None,
                 regions={'Global': {}}, mode='maps', alias={},
                 full_data=True, merge=False, vlst='%(var)s', ncfiles=[],
                 color_cycle=None, **kwargs):
        """Initialization function for violin plot
        Input:
          - maps: List of FieldPlot instances (!Not WindPlot! it has to be a
              scalar field)
          - error string, float or list of 2 floats.
              -- if string, it may be a float combined with 'std' or n-times
                  'std'), (e.g. '0.75std', or 'std' or 'stdstd', etc.). The
                  (n-times or float-times) standard deviation will then be
                  used as error
              -- if float f: the error will be the 50-f percentile for the
                  lower error bound and the 50+f percentile for the upper error
                  bound
              -- if list [f1, f2], f1 will be the percentile of the lower
                  error bound and f2 the percentile for the upper error bound.
          - name: name of the FldMeanEvaluator instance
          - mapsin: Maps instance the evaluator belongs to.
          - labels: string or array. labels that shall be used for the MapBase
              instances in the legend. If None, it depends on the mode.
              -- if mode == 'both', labels will be like 'name, region' where
                  'name' is the MapBase.name attribute and 'region' the region
                  name
              -- if mode == 'regions': labels will be the MapBase.name
                  attribute
              -- if mode == 'maps': labels will be the region
          - names: string or array. names that shall be used for the
              lines in the LinePlot instance (see *names* keyword in
              LinePlot.__init__). If None, it depends on the mode.
              -- if mode == 'both', names will be like 'name_region' where
                  'name' is the MapBase.name attribute and 'region' the region
                  name
              -- if mode == 'regions': names will be the MapBase.name attribute
              -- if mode == 'maps': names will be the region
          - fmt: dictionary with keywords for the resulting axes of SimpleFmt.
          - pyplotfmt: dictionary with keywords for the resulting lines. If you
              want to set options for a specific line, use the MapBase.name
              value (if mode == 'maps' or 'both'), the region (if mode ==
              'regions' or 'both'), or 'name_region' (if mode == 'both') or
              whatever you specify in names.
          - ax: Tuple (x, y) or axes instances to draw on. If None, new
              figures with one single subplot will be created. If tuple (x, y),
              x determines the number of subplots per row and y the number of
              subplots per column.
          - mode: string ('regions', 'maps' or None). Determines for which
              to sort. If 'regions', all MapBase instances will be plottet into
              one plot with one plot per region. If 'maps', all regions will
              be plottet into one plot, with one plot per MapBase instance in
              maps. If None, no sorting will be applied, i.e. everything will
              be plotted into one plot.
          - regions: dictionary. Keys are names for the regions (use 'Global'
              for non-masking) values are mask values as for the FmtBase.mask
              property: %(mask)s
          - alias: dictionary. You may encounter problems if you use a region
              key for LinePlot.name or line names (e.g. like Köln).
              Therefore you can set aliases which will be used instead.
              For example, if names and labels are None,
              regions={'Köln': ['mymask.nc']},
              alias={'Köln': 'Koeln'} will set a label like
              'name, Köln' and the line name as 'name_Koeln' (where
              name is the name of the MapBase instance.
          - full_data: True/False (Default: True). If True, the mean for all
              dimensions (e.g. including a level dimension) is calculated and
              included in the resulting readers of the instances.
          - merge: True/False (Default: False). If True, resulting readers of
              the LinePlot instances are merged
          - ncfiles: List of strings or a dictionaries.
              By default, ArrayReader instances are created during the
              evaluation. However if ncfiles is set, the data is stored as
              NetCDF files and the ArrayReader instances are closed.
                -- if strings: They are the path to the new NetCDF files,
                -- if dictionary, the keys are determined by the
                    nc2map.reader.ArrayReaderBase.dump_nc method
              If you want to dump the calculated data later, use the dump_nc
              method.
          - color_cycle: Any iterable containing color definitions or a
              registered colormap suitable for maplotlib.pyplot.get_cmap method
          - vlst: String or list of variable names that shall be used for
              the variable name in the new reader. By default: '%%(var)s', i.e.
              the same variable name as it is used in the MapBase.var attribute

        Further keywords are passed to the subplot creation (see plt.subplots
        function). By default sharex and sharey are set to true.

        Keywords of fmt may be (see nc2map.show_fmtdocs('xy') for details)
        """
        # docstring is extended below
        def replace_criticals(string):
            """replaces critical characters in a string to allow it as
            keywords for functions.
            Critical characters (' ', '-', '\\', "'", '"') are replaced by
            '_'.
            However maybe not all critical characters are replaced"""
            for pattern in [' ', '-', '\\', "'", '"']:
                string = string.replace(pattern, '_')
            return string

        self.name = name
        self.set_logger()
        super(FldMeanEvaluator, self).__init__(
            maps=maps, name=name, mapsin=mapsin)
        mode = mode.lower()
        kwargs.setdefault('sharex', True)
        kwargs.setdefault('sharey', True)
        regions = OrderedDict(regions)
        ncfiles = iter(ncfiles)
        self.mode = mode
        self.regions = regions
        readers, vlst, errors = self._create_readers(
            full_data=full_data, merge=merge, vlst=vlst, error=error,
            ncfiles=ncfiles)
        readers = iter(readers)
        vlst = iter(vlst)
        try:
            fmt.setdefault('xticks', 2)
        except AttributeError:
            pass
        if mode == 'both':
            ax = self._make_subplots(ax, **kwargs)[0]
            if names is None:
                all_names = [
                    map(replace_criticals,
                        ['%s_%s' % (name, alias.get(region, region))
                         for region in regions.keys()])
                    for name in self.mapnames]
            else:
                all_names = [
                    list(repeat(name, len(regions))) for name in names]
            if isinstance(fmt, SimpleFmt):
                pass
            else:
                fmt = SimpleFmt(**fmt)
            pyplotfmt = self._setup_pyplotfmt(pyplotfmt,
                                              list(chain(*all_names)))
            if labels is None:
                labels = '%(name)s, {region}'
            # set up labels
            try:
                labels = [
                    mapo._replace(labels.format(region=region))
                    for mapo, region in product(self.maps, regions.keys())]
            except TypeError, AttributeError:
                pass
            labels = cycle(iter(labels))
            for names, mapo, error in zip(all_names, self.maps, errors):
                this_pyplotfmt = {}
                for name in names:
                    this_pyplotfmt[name] = pyplotfmt[name].copy()
                    this_pyplotfmt[name].setdefault('label', next(labels))
                    this_pyplotfmt[name].setdefault('fill', error)
                dims = self._get_dims(mapo)
                self.lines.append(LinePlot(
                    next(readers), fmt=fmt, pyplotfmt=this_pyplotfmt, ax=ax,
                    names=names, vlst=next(vlst), region=range(len(regions)),
                    name=mapo.name+'_fldmean', color_cycle=color_cycle,
                    **dims))
        elif mode == 'regions':
            ax = cycle(iter(self._make_subplots(ax, max([1, len(regions)]),
                                                **kwargs)))
            if names is None:
                all_names = [[name for region in regions.keys()]
                             for name in self.mapnames]
            else:
                all_names = [
                    list(repeat(name, len(regions))) for name in names]
            pyplotfmt = self._setup_pyplotfmt(pyplotfmt,
                                              list(chain(*all_names)))
            if labels is None:
                labels = '%(name)s'
            # set up labels
            try:
                labels = [
                    mapo._replace(labels.format(region=region))
                    for mapo, region in product(self.maps, regions.keys())]
            except TypeError, AttributeError:
                pass
            labels = cycle(iter(labels))
            if isinstance(fmt, SimpleFmt):
                fmts = cycle(fmt)
            else:
                fmt.setdefault('title', '%(region)s')
                fmts = cycle((SimpleFmt(**fmt) for region in regions))
            for names, mapo, error in zip(all_names, self.maps, errors):
                reader = next(readers)
                dims = self._get_dims(mapo)
                for i, (name, region) in enumerate(zip(names, regions.keys())):
                    this_pyplotfmt = pyplotfmt[name].copy()
                    this_pyplotfmt.setdefault('label', next(labels))
                    this_pyplotfmt.setdefault('fill', error)
                    self.lines.append(LinePlot(
                        reader, fmt=next(fmts), pyplotfmt=this_pyplotfmt,
                        ax=next(ax), names=[name], vlst=next(vlst), region=i,
                        color_cycle=color_cycle, name='%s_%s_fldmean' % (
                            mapo.name, alias.get(region, region)),
                        meta = {'region': region}, **dims))
        elif mode == 'maps':
            ax = iter(self._make_subplots(ax, max([1, len(self.mapnames)]),
                                          **kwargs))
            if names is None:
                all_names = [[region for region in regions.keys()]
                             for name in self.mapnames]
            else:
                all_names = [
                    list(repeat(name, len(regions))) for name in names]
            pyplotfmt = self._setup_pyplotfmt(pyplotfmt,
                                              list(chain(*all_names)))
            if labels is None:
                labels = '{region}'
            # set up labels
            try:
                labels = [
                    mapo._replace(labels.format(region=region))
                    for mapo, region in product(self.maps, regions.keys())]
            except TypeError, AttributeError:
                pass
            labels = cycle(iter(labels))
            for names, mapo, error in zip(all_names, self.maps, errors):
                fmt.setdefault('title', '%(maponame)s')
                this_pyplotfmt = {}
                dims = self._get_dims(mapo)
                for region in regions:
                    this_pyplotfmt[region] = pyplotfmt[region].copy()
                    this_pyplotfmt[region].setdefault('label', next(labels))
                    this_pyplotfmt[region].setdefault('fill', error)
                self.lines.append(LinePlot(
                    next(readers), fmt=fmt, pyplotfmt=this_pyplotfmt,
                    ax=next(ax), names=names, vlst=next(vlst),
                    region=range(len(regions)), name=mapo.name + '_fldmean',
                    color_cycle=color_cycle, meta = {'maponame': mapo.name},
                    **dims))
        else:
            for reader in readers:
                reader.close()
            raise ValueError(
                "Unknown mode %s. Possible values are 'regions', 'maps' or "
                "'both'." % mode)

        self._set_ylabels()
        if self._mapsin is not None:
            self._mapsin.addline(self.lines)

    def _get_dims(self, mapo):
        """Deletes the time and 'var' variable from the dimensions of mapo
        and returns them"""
        dims = mapo.dims.copy()
        for dim in mapo.reader._timenames.union({'var'}):
            del dims[dim]
        return dims

    def _create_readers(self, full_data=True, merge=False, vlst='%(var)s',
                        error=None, ncfiles=[]):
        # create readers instances
        readers = []
        errors = [None]*len(self.maps)
        regions = self.regions
        if isinstance(vlst, (str, unicode)):
            vlst = [mapo._replace(vlst) for mapo in self.maps]
        vlst = list(vlst)
        for i, mapo in enumerate(self.maps):
            self.logger.debug(
                "------ Start calculation for mapo %s ------", mapo.name)
            var = mapo.var
            new_var = mapo._replace(vlst[i])
            reader = mapo.reader.selname(var)
            try:
                del reader._grid_file
            except AttributeError:
                pass
            varo = reader.variables[var]
            if not regions:
                regions = {'Global': {}}
            origmask = mapo.fmt._mask
            shape = list(varo.shape)
            dims = list(varo.dimensions)
            if not full_data:
                for i, dim in enumerate(dims):
                    if not dim == reader.timenames:
                        shape[i] = 1
            if not reader._udim(varo):
                shape.pop(dims.index(reader.lonnames))
                shape.pop(dims.index(reader.latnames))
                dims.remove(reader.lonnames)
                dims.remove(reader.latnames)
            else:
                shape.pop(dims.index(reader._udim(varo)))
                dims.remove(reader._udim(varo))
            dims = ['region'] + dims
            varo.dimensions = dims
            comment = varo.meta.get('comment', '') + (
                'Region dependent fldmean. See variable region for details.\n')
            varo.meta['comment'] = comment
            shape = [len(regions)] + shape
            final_data = np.ma.zeros(shape)

            if error is not None:
                meta = varo.meta.copy()
                try:
                    err = len(re.findall('std', error))
                    try:
                        err *= float(
                            re.search('(\d+(\.\d*)?|\.\d+)', error).group())
                    except AttributeError:  # pass if NoneType
                        pass
                    meta['comment'] += "Standard deviation of variable %s" % (
                        new_var)
                    err_min = final_data.copy()
                    reader.createVariable(err_min, new_var+'_std', dims, meta)
                    errors[i] = new_var+'_std'
                except TypeError:
                    try:
                        err = list(error)
                    except TypeError:
                        err = [50-error, 50+error]
                    err_min = final_data.copy()
                    err_max = final_data.copy()
                    c1 = meta['comment'] + \
                        ("%s-th percentile of variable %s" % (err[0], var))
                    c2 = meta['comment'] + \
                        ("%s-th percentile of variable %s" % (err[1], var))
                    meta['comment'] = c1
                    reader.createVariable(err_min, new_var+'_pctl%s' % err[0],
                                          dims, meta.copy())
                    meta['comment'] = c2
                    reader.createVariable(err_max, new_var+'_pctl%s' % err[1],
                                          dims, meta.copy())
                    errors[i] = [new_var+'_pctl%s' % err[0],
                                 new_var+'_pctl%s' % err[1]]
            # calculate weights
            data = mapo.extract_in_reader(
                    full_data=full_data, mask_data=False,
                    time=slice(None)).get_data(
                        var=mapo.var, datashape='any')
            weights = mapo.gridweights(data)
            for i, (region, mask) in enumerate(regions.items()):
                self.logger.debug(
                    "------ Start calculation for region %s ------", region)
                if region == 'Global':
                    mask = origmask
                mapo.update(mask=mask)
                data = mapo.extract_in_reader(
                    full_data=full_data, mask_data=True,
                    time=slice(None)).get_data(
                        var=mapo.var, datashape='any')
                self.logger.debug("    Calculate fldmean")
                final_data[i, :] = data.fldmean(weights=weights)
                if error is not None and not np.array(err).ndim:
                    self.logger.debug(
                        "    Calculate %s times standard deviation ", err)
                    err_min[i, :] = err*data.fldstd(weights=weights)
                elif error is not None:
                    self.logger.debug("    Calculate percentiles %s", err)
                    err_min[i, :], err_max[i, :] = data.percentile(
                        err, weights=weights)
            varo.data = final_data
            # calculate error
            meta = {
                'comment': '\n'.join(map(
                    lambda item: '%i: %s' % item, zip(range(len(regions)),
                                                      regions.keys()))),
                'units': '-'}
            reader.createVariable(range(len(regions)), 'region', ['region'],
                                  meta=meta)
            readers.append(reader)
            mapo.update(mask=origmask)
            varo.var = new_var
            reader.renameVariable(var, new_var)
            reader.variables.pop(reader.lonnames)
            reader.variables.pop(reader.latnames)
        if merge and len(readers) > 1:
            reader = readers[0].merge(*readers[1:])
            try:
                ncfile = next(ncfiles)
                try:
                    reader = reader.to_NCReader(**ncfile)
                except TypeError:
                    reader = reader.to_NCReader(ncfile)
            except StopIteration:
                pass
            readers = repeat(reader)
        else:
            for i, reader in enumerate(readers):
                try:
                    ncfile = next(ncfiles)
                    try:
                        readers[i] = reader.to_NCReader(**ncfile)
                    except TypeError:
                        readers[i] = reader.to_NCReader(ncfile)
                except StopIteration:
                    pass
        return readers, vlst, errors

    def _set_ylabels(self):
        for line in self.lines:
            if line.fmt.ylabel is None:
                if line.meta.get('long_name'):
                    label = '%(long_name)s'
                else:
                    label = '%(var)s'
                if line.meta.get('units'):
                    label += ' [%(units)s]'
                line.fmt.ylabel = label

    def _setup_pyplotfmt(self, pyplotfmt, names):
        regions = self.regions
        mode = self.mode
        new_pyplotfmt = {}
        names = list(islice(cycle(names), 0, len(regions)*len(self.maps)))
        for name, (mapname, region) in zip(names, product(self.mapnames,
                                                          regions.keys())):
            new_pyplotfmt.setdefault(name, pyplotfmt.get(name, {}))
            for key, val in pyplotfmt.items():
                if key in self.regions.keys() + names + self.mapnames:
                    continue
                new_pyplotfmt[name].setdefault(key, val)
            for key, val in pyplotfmt.get(region, {}).items():
                if key in self.regions.keys() + names + self.mapnames:
                    continue
                new_pyplotfmt[name].setdefault(key, val)
            for key, val in pyplotfmt.get(mapname, {}).items():
                if key in self.regions.keys() + names + self.mapnames:
                    continue
                new_pyplotfmt[name].setdefault(key, val)
        return new_pyplotfmt

    # --------- modify docstrings here -------------
    __init__.__doc__   = __init__.__doc__ % get_fmtdocs() + \
        _get_fmtkeys_formatted('xy')
