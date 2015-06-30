# -*- coding: utf-8 -*-
from difflib import get_close_matches
from _fmtproperties import FmtProperties
from _base_fmt import BaseFormatter, _replace_note
from ..defaults import FmtBase, shapes
from ..warning import warn, critical, warnings, Nc2MapWarning
default = FmtBase


props = FmtProperties()  # container containing methods for property definition


class FmtBase(BaseFormatter):
    """Base class of formatoptions.
    Documented properties (plotcbar, rasterized, etc.) are what can be set as
    formatoption keywords in any MapBase and maps instance."""

    # ------------------ define properties here -----------------------
    # General properties
    plotcbar = props.plotcbar(
        'plotcbar', """
        String or list of possible strings (see below). Default: %s.
        Determines where to plot the colorbar. Possibilities are 'b' for at the
        bottom of the plot, 'r' for at the right side of the plot, 'sh' for a
        horizontal colorbar in a separate figure, 'sv' for a vertical colorbar
        in a separate figure. For no colorbar use '', None, False, [], etc.
        A string may be a combination of multiple positions (e.g. 'bsh' will
        draw a colorbar at the bottom of the plot and a separate horizontal
        one).""" % default['plotcbar'])
    rasterized = props.default(
        'rasterized', """
        Boolean (Default: %s). Rasterize the pcolormesh (i.e. the mapplot)
        or not.""" % default['rasterized'])
    latlon = props.default(
        'latlon', """
        True/False (Default: %s). Sets latlon keyword for basemap plot function
        (or not).""" % default['latlon'])
    cmap = props.cmap(
        'cmap', """
        string or colormap (e.g.matplotlib.colors.LinearSegmentedColormap)
        (Default: %s). Defines the used colormap. If cmap is a colormap,
        nothing will happen. Otherwise if cmap is a string, a colorbar will be
        chosen. Possible strings are
        - 'red_white_blue' (e.g. for symmetric precipitation colorbars)
        - 'white_red_blue' (e.g. for asymmetric precipitation colorbars)
        - 'blue_white_red' (e.g. for symmetric temperature colorbars)
        - 'white_blue_red' (e.g. for asymmetric temperature colorbars)
        - any other name of a standard colorbar as provided by pyplot
            (e.g. 'jet','Greens','binary', etc.). Use function
            nc2map.show_colormaps to visualize them.""" % default['cmap'])
    ticks = props.cmapprop(
        'ticks', """
        1D-array or integer (Default: %s). Define the ticks of the colorbar.
        In case of an integer i, every i-th value of the default ticks will be
        used.""" % default['ticks'])
    extend = props.cmapprop(
        'extend', """
        string  ('neither', 'both', 'min' or 'max') (Default: %s). If
        not 'neither', make pointed end(s) for out-of-range values. These are
        set for a given colormap using the colormap set_under and set_over
        methods.""" % default['extend'])

    # labels
    clabel = props.default(
        'clabel', """
        string (Default: %s). Defines the label of the colorbar (if plotcbar
        is True).""" % default['clabel'] + _replace_note)
    ticklabels = props.cmapprop(
        'ticklabels', """
        Array (Default: %s). Defines the ticklabels of the colorbar""" %
        default['ticklabels'])
    cticksize = props.cticksize(
        'cticksize', """
        string or float (Default: %s). Defines the size of the colorbar ticks
        (see fontsize for possible values)""" % default['cticksize'])
    ctickweight = props.ctickweight(
        'ctickweight', """
        Fontweight of colorbar ticks (Default: Defined by fontweight property).
        See fontweight above for possible values.""")

    # basemap properties
    lonlatbox = props.lonlatbox(
        'lonlatbox', """
        1D-array [lon1,lon2,lat1,lat2], string (or pattern), or dictionary
        (Default: global, i.e. %s for proj=='cyl' and Northern Hemisphere for
        'northpole' and Southern for 'southpole'). Selects the region for the
        plot.
        - If string this will be compiled as a pattern to match any of the
            keys in nc2map.defaults.lonlatboxes (it contains
            longitude-latitude definitions for countries and continents).
            E.g. to focus on Germany, set lonlatbox='Germany'. To focus on
            Africa, set lonlatbox='Africa'. To focus on Germany, France
            and Italy, set lonlatbox='Germany|France|Italy'.
        - If dictionary possible keys are
            -- 'ifile' to give an input shapefile (if not set, use the shapes
               from the default shape file located at
               %s
               This Shapefile is based upon the bnd-political-boundary-a.shp
               shapes from the Vmap0 Dataset from GIS-Lab
               (http://gis-lab.info/qa/vmap0-eng.html), accessed May 2015.
            -- any field name in the input shape file (see nc2map.get_fnames
               and nc2map.get_unique_vals function) to select specific
               shapes""" % (default['lonlatbox'], shapes['boundaryfile']))
    proj = props.proj(
        'proj', """
        string ('cyl', 'robin', 'northpole', 'southpole') or dictionary
        (Default: %s). Defines the options for the projection used for the
        plot. If string, Basemap is set up automatically with settings from
        lonlatbox, if dictionary, these are the keyword arguments passed to
        mpl_toolkits.basemap.Basemap initialization.""" % default['proj'])
    lineshapes = props.lineshapes(
        'lineshapes', """
        string, list of strings or dictionary. (Default: %s). Draw polygons on
        the map from a shapefile.
        - If string or list of strings this will be seen as the values for the
            %s field in the default shapefile (see 'ifile' below) and all
            matching polygons in this shape file will be merged.
        - If dictionary possible keys are
            -- 'ifile' to give an input shapefile (if not set, use the shapes
                from the default shape file located at
                %s
                This Shapefile is based upon the bnd-political-boundary-a.shp
                shapes from the Vmap0 Dataset from GIS-Lab
                (http://gis-lab.info/qa/vmap0-eng.html), accessed May 2015.
            -- 'ofile' for the target shape file if specific shapes are
               selected or 'dissolve' is set to False
            -- 'dissolve'. True/False (Default: False). If True, all polygons
               will be merged into one single shape
            -- any field name in the input shape file (see nc2map.get_fnames
               and nc2map.get_unique_vals function) to select specific
               shapes
            -- any other key (but the 'name' key) which is finally passed to
               the readshapefile method (e.g. 'color' or 'linewidth')

        Each shape is uniquely defined through a key. If you use a dictionary d
        with the settings described above, you can set the key manually via
        {'my_own_key': d}. Otherwise a key like 'shape%%i' will automatically
        be assigned, where '%%i' depends on the number of already existing
        shapes.

        You can use these keys to remove a shape from the current plot by
        simply setting shapes='key_to_remove' (or whatever key you want to
        remove). Otherwise you can remove all drawn shapes with anything
        that evaluates to False (e.g. shapes=None).
        Please note that it might take a while to dissolve all polygons if
        'dissolve' is set to True and even to extract them if the shapefile
        is large. Therefore, if you use the shape on multiple plots, use
        the share.lineshapes method of the specific MapBase instance""" % (
            default['lineshapes'], shapes['default_field'],
            shapes['boundaryfile']))
    meridionals = props.meridionals(
        'meridionals', """
        1D-array or integer (Default: %s). Defines the lines where to draw
        meridionals. Possible types are
        - 1D-array: manually specify the location of the meridionals
        - integer: Gives the number of meridionals between maximal and minimal
            longitude (including max- and minimum line)""" %
        default['meridionals'])
    parallels = props.parallels(
        'parallels', """
        1D-array or integer (Default: %s). Defines the lines where to draw
        parallels. Possible types are
        - 1D-array: manually specify the location of the parallels
        - integer: Gives the number of parallels between maximal and minimal
            lattitude (including max- and minimum line)""" %
        default['parallels'])
    merilabelpos = props.merilabelpos(
        'merilabelpos', """
        List of 4 values (Default: %s) that control whether meridians are
        labelled where they intersect the left, right, top or bottom of the
        plot. For example labels=[1, 0, 0, 1] will cause meridians to be
        labelled where they intersect the left and bottom of the plot, but not
        the right and top.""" % default['merilabelpos'])
    paralabelpos = props.paralabelpos(
        'paralabelpos', """
        List of 4 values (Default: %s) that control whether parallels are
        labelled where they intersect the left, right, top or bottom of the
        plot. For example labels=[1, 0, 0, 1] will cause parallels to be
        labelled where they intersect the left and and bottom of the plot, but
        not the right and top.""" % default['paralabelpos'])
    lsm = props.default(
        'lsm', """
        Boolean (Default: %s). If True, the continents will be plottet.""" %
        default['lsm'])
    countries = props.default(
        'countries', """
        Boolean (Default: %s). If True, draw country borders.""" %
        default['countries'])
    land_color = props.default(
        'land_color', """
        color instance (Default: %s). Specify the color of the land. """ % (
            default['land_color']))
    ocean_color = props.default(
        'ocean_color', """
        color instance (Default: %s). Specify the color of the ocean.
        Attention! Might reduce the performance a lot if multiple plots are
        opened! To not kill everything, use the MapBase.share.lsmask method of
        the specific MapBase instance.""" % default['ocean_color'])

    # Colorcode properties
    bounds = props.bounds(
        'bounds', """
        1D-array, tuple or string (Default: %s). Defines the bounds used for
        the colormap. Possible types are
        - 1D-array: Defines the bounds directly by giving the values
        - tuple (string, N): Compute the bounds automatically. N gives the
            number of increments whereas string can be one of the following
            strings
            -- 'rounded': Rounds min and maxvalue of the data to the next
                0.5-value with respect to its exponent with base 10 (i.e.
                1.3e-4 will be rounded to 1.5e-4)
            -- 'roundedsym': Same as 'rounded' but symmetric around zero using
                the maximum of the data maximum and (absolute value of) data
                minimum.
            -- 'minmax': Uses minimum and maximum of the data (without
                rounding)
            -- 'sym': Same as 'minmax' but symmetric around 0 (see 'rounded'
                and 'roundedsym').
        - tuple (string, N, percentile): Same as (string, N) but uses the
            percentiles defined in the 1D-list percentile as maximum.
            percentile must have length 2 with [minperc, maxperc]
        - string: same as tuple with N automatically set to 11.""" %
        str(default['bounds']))

    norm = props.default(
        'norm', """
        mpl.colors.Normalization instance, 'bounds' or None (Default:
        %r). Defines the normalization instance to
        plot a normalized colorbar. If 'bounds', a mpl.colors.BoundaryNorm
        instance is used with the `bounds` formatoption keyword.""" % (
            default['norm']))

    opacity = props.default(
        'opacity', """
        Float 0<f<=1, 1D-array or 2D-array (Default: %s).
          - If float, the colormap is modified such that the alpha value
              increases until the percentile defined by this value.
          - If 1D-array, this will be interpolated to the whole colormap
              range.
          - If 2D-array, shape must be like (N, 2), where N can be as large
                as you want. Then, opacity[:, 0] must be data points and
                opacity[:, 1] must be values between 0. and 1. for the alpha
                value. Note that if the minimum of opacity[:, 0] is larger
                than  the minimal bound, the alpha value of points below will
                be set to 0. On the other hand, if the maximum of
                opacity[:, 1] is smaller than the maximal bound, the alpha
                value of points above those will be set to 1. Between it
                will be interpolated linearly.""" % default['opacity'])

    # masking properties
    mask = props.mask_from_reader(
        'mask', """
        array (x[, var][, num]) (Default: %s). The first entry must be a
        string for a netCDF file or a nc2map.readers.ReaderBase instance, the
        second entry might be the name of the variable in the mask file to
        read in, the number at the end defines the values where to mask (if
        not given, mask everywhere where the mask is 0)""" % default['mask'])

    def __init__(self, **kwargs):
        """initialization and setting of default values. Key word arguments may
        be any names of a property. Use show_fmtkeys for possible keywords and
        their documentation"""
        self.set_logger()
        super(FmtBase, self).__init__()
        self._ctickops = {}
        self._default.update(default)

        # Option dictionaries
        self._projops = {}     # settings for projection
        self._meriops = {}     # settings for meridionals on map
        self._paraops = {}     # settings for parallels on map
        self._calc_shapes = 1  # flag for calculating shapes
        # keys of the important properties for the basemap which force
        # reinitialization of the MapBase object (currently: lonlatbox and
        # proj)
        self._bmprops = []
        # keys of colormap properties (important for the dialog between
        # FieldPlot and WindPlot instance.
        self._cmapprops = []
        # if false, no changes can be made for cmap or bounds (see property
        # definition)
        self._enablebounds = True
        self._maskprops = []
        # self._general, the list containing the names of the baseclass
        # keywords, is defined below

        self.glob = [-180., 180., -90., 90.]
        # set default values
        for key, val in default.items():
            setattr(self, key, val)

        exclusive_keys = {'enable', 'clabel'}
        # base class properties
        self._general = sorted(list(
            set(self._default.keys()) - set(self._cmapprops) - exclusive_keys))
        # update for kwargs
        self.update(**kwargs)

    def _removeoldkeys(self, entries):
        """Method to remove wrong keys and modify them"""
        entries = super(FmtBase, self)._removeoldkeys(entries)
        return entries
