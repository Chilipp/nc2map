# -*- coding: utf-8 -*-
"""Module to plot netCDF files (interactively)

This module is attempted to handle netCDF files with the use of
python package netCDF4 and to plot them with the use of python
package matplotlib.
Requirements (at least this package is tested with)
   - matplotlib version, 1.3.1
   - mpl_toolkits.basemap, version 1.07
   - netCDF4, version 1.1.3
   - six
   - Python 2.7
Main class for usage is the Maps object class. A helper function
for the formatoption keywords is show_fmtkeys, displaying the
possible formatoption keywords.
Please look into nc2map/demo (or nc2map.demo) for demonstration scripts.
If you find any bugs, please do not hesitate to contact the authors.
This is nc2map version 0.0beta, so there might be some bugs.
"""
import os
import logging
import datetime as dt
from _maps import Maps
from _maps_manager import MapsManager
from _cbar_manager import CbarManager
from _basemap import Basemap
from formatoptions import (
    get_fmtkeys, show_fmtkeys, get_fmtdocs, show_fmtdocs, get_fnames,
    get_unique_vals, close_shapes)
from ._cmap_ops import show_colormaps, get_cmap
from _axes_wrapper import wrap_subplot, subplots, multiple_subplots
from .warning import warn, critical, disable_warnings
try:
    from _cdo import Cdo
except ImportError:
    pass


__version__ = "0.00b"
__author__  = "Philipp Sommer (philipp.sommer@studium.uni-hamburg.de)"

def setup_logging(default_path=os.path.dirname(__file__) + '/logging.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    Input:
     - default_path: Default path of the yaml logging configuration file
         (Default: logging.yaml in nc2map source directory)
     - default_level: default level if default_path does not exist
         (Default: logging.INFO)
     - env_key: environment variable specifying a different logging file than
         default_path (Default: LOG_CFG)

    Function taken from
    http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python
    """
    import logging.config
    import yaml
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        for handler in config.get('handlers', {}).values():
            try:
                handler['filename'] = '%s/.%s' % (os.path.expanduser('~'),
                                                 handler['filename'])
            except KeyError:
                pass
        logging.config.dictConfig(config)
    else:
        path = None
        logging.basicConfig(level=default_level)
    return path


path = setup_logging()

logger = logging.getLogger(__name__)
logger.debug(
    "%s: Initializing nc2map, version %s",
    dt.datetime.now().isoformat(), __version__)
logger.debug("Logging configuration file: %s", path)


def get_default_shapefile():
    """Returns the default shape file used by lonlatbox and shapes formatoption
    keywords"""
    from .defaults import shapes
    return shapes['boundaryfile']


def load(filename, readers=None, ax=None, max_ax=None, ask=True,
         force_reader=False, force_ax=False, plot=True):
    """Create a Maps object from a pickle file or dictionary

    Input:
      - filename: String or dictionary. If string, it must be the path
          to a pickle file (e.g. created by the Maps.save method). If
          dictionary it must be a dictionary as created from Maps.save
          method.
      - readers: List of readers to use.
          It may happen, that not all readers were saved (i.e. they are None
          in the filename dictionary) because the initialization settings
          could not be determined during the saving of the Maps instance. Or
          maybe you want to use your own readers (see force_reader keyword).
          In that case give an iterable containing nc2map.reader.ArrayReader
          instances to use instead.
      - ax: List of subplots to use.
          It may happen, that not all figure settings were saved (i.e. they
          are None in the filename dictionary) because the initialization
          settings could not be determined during the saving of the Maps
          instance. Or maybe you want to use your own figure settings (see
          force_ax keyword).
          In that case give an iterable containing subplots
          instances to use instead.
      - max_ax: Integer. Determines the maximal number of subplots per
          figure. Does not have an effect if force_ax is True.
      - ask: True/False. If True and the reader could not be determined, it
          will be ask for the filename
      - force_reader: True/False. If True, use only the readers specified by
          the readers keyword
      - force_ax: True/False. If True, use only the subplots specified by the
          ax keyword.
      - plot: True/False. Make plots of all MapBase and SimplePlot instances
          at the end or not
    """
    import pickle
    from numpy import ravel
    from copy import deepcopy
    import readers as rd
    import mapos

    logger = logging.getLogger(__name__)
    logger.debug('Loading Maps settings...')

    try:
        readers = iter(readers)
    except TypeError:
        logger.debug('readers is not iterable, I assume it is None...')
        readers = iter([])
    try:
        ax = iter(ax)
    except TypeError:
        logger.debug('ax is not iterable, I assume it is None...')
        ax = iter([])
    try:
        logger.debug('    Try pickle load...')
        with open(filename) as f:
            idict = pickle.load(f)
    except TypeError:
        logger.debug('    Failed. --> Assume dictionary', exc_info=True)
        idict = filename
    rd_dict = idict['readers']
    fig_dict = idict['figures']
    maps_dict = deepcopy(idict.get('maps', {}))
    shared_dict = idict.get('share', {})
    lines_dict = deepcopy(idict.get('lines', {}))
    cbars = deepcopy(idict.get('cbars', []))
    logger.debug('    Open readers...')
    for reader, val in rd_dict.items():
        logger.debug('         Open reader %s', reader)
        if val is None or force_reader:
            try:
                rd_dict[reader] = next(readers)
            except StopIteration:
                warn("Could not open reader %s!" % reader)
                if ask:
                    default_reader = "NCReader"
                    fname = raw_input(
                        "Please insert the path to the NetCDF file(s) or "
                        "nothing to continue. Multiple files should be "
                        "separated by commas\n")
                    if not fname:
                        continue
                    fname = fname.split(',')
                    fname = fname if len(fname) > 1 else fname[0]
                    rtype = raw_input(
                        "Please specify which reader class to use (NCReader "
                        "or MFNCReader). without options: %s)\n" % (
                            default_reader))
                    if not rtype:
                        rtype = default_reader
                    rd_dict[reader] = getattr(rd, rtype)(fname)
            continue
        try:
            rd_dict[reader] = getattr(rd, val[0])(*val[1], **val[2])
        except TypeError:  # assume a reader
            pass
    logger.debug('    Open figures...')
    for fig, val in fig_dict.items():
        logger.debug('        Open figure %s', fig)
        if val is None or force_ax:
            warn("Could not open figure %s!" % fig)
            continue
        fig_axes = ravel(subplots(val[0][0], val[0][1], *val[1], **val[2])[1])
        fig_dict[fig] = fig_axes[slice(0, max_ax)]
        # delete excessive axes
        if len(fig_dict[fig]) < len(fig_axes):
            figo = fig_axes[0].get_figure()
            for axes in fig_axes[len(fig_dict[fig]):]:
                figo.delaxes(axes._AxesWrapper__ax)


    logger.debug('Open Maps instances')
    mymaps = Maps(_noadd=True)
    logger.debug('Add maps to Maps instance')
    for mapo, mdict in maps_dict.items():
        logger.debug('        Open %s...', mapo)
        for key, val in mdict.items():
            logger.debug('            %s: %s', key, val)
        obj = getattr(mapos, mdict.pop('class'))
        try:
            axes = fig_dict[mdict.pop('fig')][mdict.pop('num')-1]
            if force_ax:
                assert 1 == 2, 'None'  # produce error to use next(ax)
        except (AssertionError, TypeError, KeyError):
            try:
                axes = next(ax)
            except StopIteration:
                warn(
                    "Could not determine axes of mapo %s because no valid "
                    "axes was given! A new figure will be opened." % mapo)
                axes = None
        reader = rd_dict.get(mdict.pop('reader'))
        if reader is None:
            critical(
                "Could not open mapo %s because no valid reader was given!" % (
                    mapo))
            continue
        mymaps.addmap(obj(
            reader=reader, name=mdict.pop('name'), ax=axes,
            **{key: val for key, val in mdict.pop('dims').items() +
               mdict.items()}), plot=False, add=False)

    logger.debug('Add lines to Maps instance')
    for line, ldict in lines_dict.items():
        logger.debug('        Open %s...', line)
        for key, val in ldict.items():
            logger.debug('            %s: %s', key, val)
        obj = getattr(mapos, ldict.pop('class'))
        try:
            axes = fig_dict[ldict.pop('fig')][ldict.pop('num')-1]
            if force_ax:
                assert 1 == 2, 'None'  # produce error to use next(ax)
        except (AssertionError, TypeError, KeyError):
            try:
                axes = next(ax)
            except StopIteration:
                warn("Could not determine axes of line %s because no valid "
                     "axes was given! A new figure will be opened." % line)
                axes = None
        reader = rd_dict[ldict.pop('reader')]
        if reader is None:
            critical(
                "Could not open line object %s because no valid reader was "
                "given!" % line)
            continue
        mymaps.addline(obj(reader=reader, name=ldict.pop('name'),
                           ax=axes, **ldict['init']), plot=False, add=False)

    if cbars:
        mymaps.update_cbar(*cbars, plot=False, add=False)
    mymaps._fmt += [mymaps.asdict('maps', 'lines', 'cbars')]

    if plot:
        mymaps.plot = True
        logger.info("Setting up projections...")
        for mapo in mymaps.maps:
            mapo._setupproj()
        logger.info("Making plots...")
        mymaps.make_plot()
        for cbar in mymaps.get_cbars():
            cbar._draw_colorbar()
        for fig in mymaps.get_figs():
            mymaps._set_window_title(fig)
        logger.debug("    Set shared settings...")
    else:
        mymaps.plot = False

    for name, sdict in shared_dict.items():
        if not sdict:
            continue
        try:
            mapo = mymaps.get_maps(name=name)[0]
        except IndexError:
            warn("Could not set shared setting for mapo %s because it was "
                    "not found in the Maps instance!")
            continue
        mapo.share._draw = 0
        mapo.share.shared = mapo.share._from_dict(sdict, mymaps.maps)
        mapo.share._draw = 1

    return mymaps
