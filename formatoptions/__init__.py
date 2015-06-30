# -*- coding: utf-8 -*-
"""Formatoptions module of the nc2map package

This module contains the formatoption classes used for the interpretation and
documentation of the formatoptions.

Classes are
  - BaseFormatter: Basis of all formatoption classes containing the
      formatoptions controlling the figure appearance (comparable to the
      BasePlot class)
  - SimpleFmt: Formatoption class for SimplePlot instances, e.g. LinePlot and
      ViolinPlot (i.e. everything that is not plotted on a map)
  - FmtBase: Basis for all map plots (comparable to the MapBase class),
      containing the formatoptions affecting the mpl_toolkits.basemap.Basemap.
  - FieldFmt: Formatoption class for FieldPlot instances (i.e. the plot of
      scalar variables with (optional) overlayed vector field)
  - WindFmt: Formatoption class for WindPlot instances (i.e. the plot of vector
      fields)

Other methods are
  - show_fmtkeys: Shows the available formatoption keywords
  - get_fmtkeys: Same as show_fmtkeys but returns a list instead of printing
  - show_fmtdocs: Shows the available formatoption keywords plus their
      documentation
  - get_fmtdocs: Same as show_fmtdocs but returns a dictionary instead of
      printing
  - get_fnames: Returns the field names in the default boundary shape file for
      the lineshapes formatoption
  - get_unique_vals: Returns the unique record values in the default boundary
      shape for the lineshapes formatoption
"""
from _base_fmt import BaseFormatter
from _fmt_base import FmtBase
from _windfmt import WindFmt
from _fieldfmt import FieldFmt
from _simple_fmt import SimpleFmt
from _fmtproperties import close_shapes


def _fmt_doc(func, *args, **kwargs):
    """Bases for show_fmtkeys, show_fmtdocs, etc"""
    if 'wind' in args:
        myfmt = WindFmt()
    elif 'simple' in args:
        myfmt = SimpleFmt()
    else:
        myfmt = FieldFmt()
    return getattr(myfmt, func)(*args, **kwargs)


def show_fmtkeys(*args):
    """Print formatoption keys in a readable manner

    Parameters
    ----------
    *args: str
        any formatoptions keyword (without: print all). Further ``*args`` may
        be,

        - 'wind': to plot the wind formatoption keywords
        - 'windonly': to plot the wind only
        - 'simple': to print the formatoptions of
            :class:`~nc2map.formatoptions.SimpleFmt` for
            :class:`~nc2map.mapos.SimplePlot` instances (e.g.
            :class:`~nc2map.mapos.Lineplot`, etc.)"""
    _fmt_doc('show_fmtkeys', *args)


def show_fmtdocs(*args):
    """Print formatoption documentations in a readable manner

    Parameters
    ----------
    *args: str
        any formatoptions keyword (without: print all). Further ``*args`` may
        be,

        - 'wind': to plot the wind formatoption keywords
        - 'windonly': to plot the wind only
        - 'simple': to print the formatoptions of
            :class:`~nc2map.formatoptions.SimpleFmt` for
            :class:`~nc2map.mapos.SimplePlot` instances (e.g.
            :class:`~nc2map.mapos.Lineplot`, etc.)"""
    _fmt_doc('show_fmtdocs', *args)

def get_fmtdocs(*args):
    """Get the formatoption documentation

    Parameters
    ----------
    *args: str
        any formatoptions keyword (without: return all). Further ``*args`` may
        be,

        - 'wind': to plot the wind formatoption keywords
        - 'windonly': to plot the wind only
        - 'simple': to print the formatoptions of
            :class:`~nc2map.formatoptions.SimpleFmt` for
            :class:`~nc2map.mapos.SimplePlot` instances (e.g.
            :class:`~nc2map.mapos.Lineplot`, etc.)
    Returns
    -------
    fmt_docs: dict
        keys are the formatoption keywords in ``*args``, values are their
        documentation"""
    return _fmt_doc('get_fmtdocs', *args)


def get_fmtkeys(*args):
    """Get formatoption keys as a list

    Parameters
    ----------
    *args: str
        any formatoptions keyword (without: return all). Further ``*args`` may
        be,

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
    return _fmt_doc('get_fmtkeys', *args)


def _get_fmtdocs_formatted(*args):
    """Print formatoption keys in a readable manner

    This function is used by :func:`nc2map.formatoptions.show_fmtdocs`

    Parameters
    ----------
    *args: str
        any formatoptions keyword (without: return all). Further ``*args`` may
        be,

        - 'wind': to plot the wind formatoption keywords
        - 'windonly': to plot the wind only
        - 'simple': to print the formatoptions of
            :class:`~nc2map.formatoptions.SimpleFmt` for
            :class:`~nc2map.mapos.SimplePlot` instances (e.g.
            :class:`~nc2map.mapos.Lineplot`, etc.)

    Returns
    -------
    string
        a multiline strings with formatoptions and their documentation"""
    return _fmt_doc('_get_fmtdocs_formatted', *args)


def _get_fmtkeys_formatted(*args, **kwargs):
    """Print formatoption keys in a readable manner

    This function is used by :func:`nc2map.formatoptions.show_fmtkeys`

    Parameters
    ----------
    *args: str
        any formatoptions keyword (without: return all). Further ``*args`` may
        be,

        - 'wind': to plot the wind formatoption keywords
        - 'windonly': to plot the wind only
        - 'simple': to print the formatoptions of
            :class:`~nc2map.formatoptions.SimpleFmt` for
            :class:`~nc2map.mapos.SimplePlot` instances (e.g.
            :class:`~nc2map.mapos.Lineplot`, etc.)

    Returns
    -------
    string
        a multiline table with formatoption keywords"""
    return _fmt_doc('_get_fmtkeys_formatted', *args, **kwargs)


def get_fnames():
    """Return possible field names in the boundary shape file

    Returns
    -------
    list of strings
        field names in the shape file

    Notes
    -----
    The boundary shapefile is stored in
    nc2map.defaults.shapes['boundaryfile']

    See Also
    --------
    nc2map.formatoptions.get_unique_vals: Return the unique values
    nc2map.shp_utils.get_fnames: Basic method that is used"""
    from ..defaults import shapes
    from ..shp_utils import get_fnames
    return get_fnames(shapes['boundaryfile'])


def get_unique_vals(*args, **kwargs):
    """Get unique values in the boundary shape file

    Parameters
    ----------
    *args
        field names in the shape file
    **kwargs
        may be used to filter the input. Keys may be field names in the
        shape file and values lists of possible values to filter the shapes
        (see :meth:`nc2map.shp_utils.PolyWriter.extract_records` method).

    Returns
    -------
    list of numpy arrays (in the order of *args) containing the unique values

    Notes
    -----
    The boundary shapefile is stored in
    nc2map.defaults.shapes['boundaryfile']

    If no arguments are given, all fields are returned. If no keyword arguments
    are given, all shapes are considered.

    See Also
    --------
    nc2map.formatoptions.get_fnames: Return the unique values
    nc2map.shp_utils.get_unique_vals: Basic method that is used"""
    from ..defaults import shapes
    from ..shp_utils import get_unique_vals
    return get_unique_vals(shapes['boundaryfile'], *args, **kwargs)
