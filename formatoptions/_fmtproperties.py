# -*- coding: utf-8 -*-
import re
import numpy as np
from collections import OrderedDict
from itertools import starmap, chain
from _basefmtproperties import BaseFmtProperties
from ..readers import auto_set_reader
from ..defaults import shapes as default_shapes
from ..defaults import lonlatboxes
from ..warning import warn, critical

# list of open_files
open_shapes = []

def close_shapes():
    """Closes all open temporary shapes (which are deleted anyway
    when closing python)"""
    global open_shapes
    for f in open_shapes:
        f.close()
    open_shapes = []

def create_shpfile(ifile, ofile=None, dissolve=False, **kwargs):
    """Create a shape file from ifile with the specified shapes

    Input:
      - ifile: File name or shapefile.Reader like instance with the
          base shapes
      - ofile: output file name for the resulting shape file
      - dissolve: True/False. Determines whether the shapes shall be
          dissolved or not
      Additional keyword arguments may correspond to fields in ifile
      (see nc2map.shp_utils.PolyWriter.extract_shapeRecords method)
    Returns:
        string. name of the file containing the desired shapes (Note:
        if not dissolve and no fields are specified in kwargs, the
        output will be ifile
    """
    from ..shp_utils import get_fnames, PolyWriter, NamedTemporaryFiles
    fnames = get_fnames(ifile)
    field_kwargs = dict(item for item in kwargs.items() if item[0] in fnames
                        or item[0] == 'exact_matches')
    # return input file if no dissolving and no extraction shall
    # be pursued and
    if not dissolve and not field_kwargs:
        kwargs.update({'shapefile': ifile})
        return kwargs
    non_field_kwargs = dict(item for item in kwargs.items()
                            if item[0] not in fnames)
    if not ofile:
        global open_shapes
        files = NamedTemporaryFiles(prefix='tmp_nc2map')
        open_shapes += files.values()
        ofile = files.name
    w = PolyWriter()
    if dissolve:
        w.dissolve(ifile, **field_kwargs)
    else:
        w.copy(ifile, **field_kwargs)
    w.save(ofile)
    non_field_kwargs.update({'shapefile': ofile})
    return non_field_kwargs

class FmtProperties(BaseFmtProperties):
    """class containg property definitions of formatoption containers FmtBase
    and subclasses FieldFmt and WindFmt"""
    def default(self, x, doc):
        """default property"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def bmprop(self, x, doc):
        """default basemap property (currently not in use)"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x not in self._bmprops:
                self._bmprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def cmap(self, x, doc):
        """Property controlling the colormap. The setter is disabled if
        self._enablebounds is False"""

        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            if self._enablebounds:
                setattr(self, '_' + x, value)
            else:
                print((
                    "Setting of colormap is disabled. Use the update_cbar "
                    "function or removecbars first."))

            if x not in self._cmapprops:
                self._cmapprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def cmapprop(self, x, doc):
        """default colormap property"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x not in self._cmapprops:
                self._cmapprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def bounds(self, x, doc):
        """bound property"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, bounds):
            if self._enablebounds:
                if isinstance(bounds, str):
                    setattr(self, '_' + x, (bounds, 11))
                else:
                    setattr(self, '_' + x, bounds)

            else:
                print(
                    "Setting of bounds is disabled. Use the update_cbar "
                    "function or removecbars first.")
            if x not in self._cmapprops:
                self._cmapprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def lonlatbox(self, x, doc):
        """lonlatbox property to automatically configure projection after
        lonlatbox is changed"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            self.logger.debug("Setting longitude-latitude box...""")
            nround = lambda xmin, xmax: [np.floor(xmin), np.floor(xmax)]
            try:
                patt = re.compile(value)
                value = np.array([box for key, box in lonlatboxes.items()
                                  if patt.search(key)])
                value = [value[:, 0].min(), value[:, 1].max(),
                         value[:, 2].min(), value[:, 3].max()]

            except TypeError:
                try:
                    from ..shp_utils import get_bbox
                    value.setdefault('ifile', default_shapes['boundaryfile'])
                    value = get_bbox(**value)
                    value = list(
                        chain(*starmap(nround, [value[0::2], value[1::2]])))
                except (AttributeError, ImportError):
                   pass
            setattr(self, '_' + x, value)
            # update properties depending on lonlatbox
            setattr(self, '_box', value)
            if hasattr(self, 'proj'):
                self.proj = self.proj

            if x not in self._bmprops:
                self._bmprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def lineshapes(self, x, doc):
        """shapes property to manage shapes"""
        def getx(self):
            return self._final_lines

        def setx(self, value):
            def next_name():
                for i in xrange(1000):
                    if not "shape%i" % i in final_shapes:
                        return "shape%i" % i
            try:
                final_shapes = self._final_lines
                shapes = getattr(self, '_' + x)
            except AttributeError:
                self._final_lines = OrderedDict()
                shapes = OrderedDict()
                setattr(self, '_' + x, shapes)
                final_shapes = self._final_lines
            if not value or not self._calc_shapes:
                setattr(self, '_' + x, value)
                if not value:
                    self._final_lines = OrderedDict()
                return
            if not shapes:
                shapes = OrderedDict()
                setattr(self, '_' + x, shapes)
            if isinstance(value, dict):
                value = value.copy()
                if any(not isinstance(val, dict)
                           for val in value.values()):
                    value = {next_name(): value}
                    self.logger.info(
                        "Storing new shapes in %s", value.keys()[0])
                for key, val in value.items():
                    shapes[key] = val.copy()
                    val = val.copy()
                    try:
                        val['ifile'] = val.pop('shapefile')
                    except KeyError:
                        val.setdefault('ifile', default_shapes['boundaryfile'])
                    final_shapes[key] = create_shpfile(**val)
            elif isinstance(value, (str, unicode)):
                if value in final_shapes:
                    del final_shapes[value]
                    del shapes[value]
                else:
                    shapes[value] = {'ifile': default_shapes['boundaryfile'],
                                     default_shapes['default_field']: [value]}
                    final_shapes[value] = create_shpfile(
                        **shapes[value])
            elif isinstance(value[0], (str, unicode)):
                key = next_name()
                self.logger.info(
                        "Storing new shapes in %s", key)
                shapes[key] = {'ifile': default_shapes['boundaryfile'],
                               default_shapes['default_field']: value}
                final_shapes[key] = create_shpfile(**shapes[value])

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def proj(self, x, doc):
        """Projection property to automatically configure projops"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, projection):
            setattr(self, '_' + x, projection)
            if projection in ['northpole', 'npstere']:
                self._defaultrange = [-180., 180., 0., 90]
                self._box = self.lonlatbox
                if self._box == self.glob:
                    self._box = self._defaultrange
                self._projops = {  # projection options for basemap
                    'projection': 'npstere', 'lon_0': np.mean(self._box[:2]),
                    'boundinglat': self._box[2], 'llcrnrlon': self._box[0],
                    'urcrnrlon': self._box[1], 'llcrnrlat': self._box[2],
                    'urcrnrlat': self._box[3], 'round': True}
                self.meridionals = self.meridionals
                if self.merilabelpos is None:
                    self._meriops['labels'] = [1, 1, 1, 1]
                self.parallels = self.parallels
                if self.paralabelpos is None:
                    self._paraops['labels'] = [0, 0, 0, 0]

            elif projection in ['southpole', 'spstere']:
                self._defaultrange = [-180., 180., -90., 0.]
                self._box = self.lonlatbox
                if self._box == self.glob:
                    self._box = self._defaultrange
                self._projops = {  # projection options for basemap
                    'projection': 'spstere', 'lon_0': np.mean(self._box[:2]),
                    'boundinglat': self._box[3], 'llcrnrlon': self._box[0],
                    'urcrnrlon': self._box[1], 'llcrnrlat': self._box[2],
                    'urcrnrlat': self._box[3], 'round': True}
                # update meridionals and parallels
                self.meridionals = self.meridionals
                if self.merilabelpos is None:
                    self._meriops['labels'] = [1, 1, 1, 1]
                self.parallels = self.parallels
                if self.paralabelpos is None:
                    self._paraops['labels'] = [0, 0, 0, 0]

            elif projection == 'cyl':
                # update meridionals and parallels
                self._defaultrange = self.glob
                self._box = self.lonlatbox
                self._projops = {  # projection options for basemap
                    'projection': projection, 'llcrnrlon': self._box[0],
                    'urcrnrlon': self._box[1], 'llcrnrlat': self._box[2],
                    'urcrnrlat': self._box[3]}
                # update meridionals and parallels
                self.meridionals = self.meridionals
                if self.merilabelpos is None:
                    self._meriops['labels'] = [0, 0, 0, 1]
                self.parallels = self.parallels
                if self.paralabelpos is None:
                    self._paraops['labels'] = [1, 0, 0, 0]

            elif projection in ['robin', 'kav7', 'eck4', 'mbtfpq', 'hammer',
                                'moll']:
                # update meridionals and parallels
                self._defaultrange = self.glob
                self._box = self.lonlatbox
                self._projops = {  # projection options for basemap
                    'projection': projection, 'lon_0': np.mean(self._box[:2])}
                # update meridionals and parallels
                self.meridionals = self.meridionals
                if self.merilabelpos is None:
                    self._meriops['labels'] = [0, 0, 0,
                                               int(projection != 'hammer')]
                self.parallels = self.parallels
                if self.paralabelpos is None:
                    self._paraops['labels'] = [1, 0, 0, 0]

            elif projection in ['ortho', 'geos']:
                self._defaultrange = [0, 180, -90, 90]
                self._box = self.lonlatbox
                if self._box == self.glob:
                    self._box = self._defaultrange
                self._projops = {  # projection options for basemap
                    'projection': projection, 'lon_0': np.mean(self._box[:2]),
                    'lat_0': np.mean(self._box[2:]),
                    'round': True}
                # update meridionals and parallels
                self.meridionals = self.meridionals
                if self.merilabelpos is None:
                    self._meriops['labels'] = [0, 0, 0, 0]
                self.parallels = self.parallels
                if self.paralabelpos is None:
                    self._paraops['labels'] = [0, 0, 0, 0]

            elif projection in ['lambert', 'lcc']:
                self._defaultrange = [0, 180, 0, 90]
                self._box = self.lonlatbox
                if self._box == self.glob:
                    self._box = self._defaultrange
                self._projops = {  # projection options for basemap
                    'projection': 'lcc', 'lon_0': np.mean(self._box[:2]),
                    'lat_0': np.mean(self._box[2:]), 'llcrnrlon': self._box[0],
                    'urcrnrlon': self._box[1], 'llcrnrlat': self._box[2],
                    'urcrnrlat': self._box[3]}
                # update meridionals and parallels
                self.meridionals = self.meridionals
                if self.merilabelpos is None:
                    self._meriops['labels'] = [0, 0, 0, 1]
                self.parallels = self.parallels
                if self.paralabelpos is None:
                    self._paraops['labels'] = [1, 0, 0, 0]

            elif projection in ['cass', 'poly']:
                self._defaultrange = [0, 85, 0, 90]
                self._box = self.lonlatbox
                if self._box == self.glob:
                    self._box = self._defaultrange
                self._projops = {  # projection options for basemap
                    'projection': projection, 'lon_0': np.mean(self._box[:2]),
                    'lat_0': np.mean(self._box[2:]), 'llcrnrlon': self._box[0],
                    'urcrnrlon': self._box[1], 'llcrnrlat': self._box[2],
                    'urcrnrlat': self._box[3]}
                # update meridionals and parallels
                self.meridionals = self.meridionals
                if self.merilabelpos is None:
                    self._meriops['labels'] = [0, 0, 0, 1]
                self.parallels = self.parallels
                if self.paralabelpos is None:
                    self._paraops['labels'] = [1, 0, 0, 0]

            elif isinstance(projection, dict):
                self._projops = projection.copy()

            else:
                try:
                    raise ValueError(
                        "Unknown projection %s. Must be one of 'cyl', "
                        "'northpole' or 'southpole' or a dictionary which is "
                        "passed to Basemap class directly" % (projection))
                except:
                    raise ValueError(
                        "Unknown projection. Must be one of 'cyl', "
                        "'northpole' or 'southpole' or a dictionary which is "
                        "passed to Basemap class directly")

            if x not in self._bmprops:
                self._bmprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def meridionals(self, x, doc):
        """property to configure and initialize meridional plotting options"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, values):
            setattr(self, '_'+x, values)
            if type(values) is int:
                self._meriops.update({
                    x: np.linspace(self._box[0], self._box[1], values,
                                endpoint=True),
                    'fontsize': self.ticksize, 'latmax': 90})
            else:
                self._meriops.update({
                    x: values, 'fontsize': self.ticksize, 'latmax': 90})

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def parallels(self, x, doc):
        """property to configure and initialize parallel plotting options"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, values):
            setattr(self, '_'+x, values)
            if type(values) is int:
                self._paraops.update({
                    x: np.linspace(self._box[2], self._box[3], values,
                                endpoint=True),
                    'fontsize': self.ticksize, 'latmax': 90})
            else:
                self._paraops.update({
                    x: values, 'fontsize': self.ticksize, 'latmax': 90})

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def merilabelpos(self, x, doc):
        """property to define axes where to plot meridional labels"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_'+x, value)
            try:
                self.proj = self.proj
            except AttributeError:
                pass
            if value is not None:
                self._meriops.update({'labels': value})

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def paralabelpos(self, x, doc):
        """property to define axes where to plot parallel labels"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_'+x, value)
            try:
                self.proj = self.proj
            except AttributeError:
                pass
            if value is not None:
                self._paraops.update({'labels': value})

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def windplotops(self, x, doc):
        """wind plot property updating wind plot options"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if hasattr(self, '_streamplot'):
                self._windplotops.update({x: value})

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def streamplotops(self, x, doc):
        """wind plot property updating wind plot options"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if hasattr(self, '_streamplot') and self.streamplot:
                self._windplotops.update({x: value})

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def quiverops(self, x, doc):
        """wind plot property updating wind plot options"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if hasattr(self, '_streamplot') and not self.streamplot:
                self._windplotops.update({x: value})

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def cticksize(self, x, doc):
        """tick fontsize options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('size', '') not in self._text_props:
                self._text_props.append(x.replace('size', ''))
            self._ctickops.update({'fontsize': value})
            if not x in self._cmapprops:
                self._cmapprops.append(x)

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def ctickweight(self, x, doc):
        """tick fontweight options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('weight', '') not in self._text_props:
                self._text_props.append(x.replace('weight', ''))
            self._ctickops.update({'fontweight': value})
            self._cmapprops.append(x)

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def streamplot(self, x, doc):
        """streamplot property to configure windplotops"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            self._windplotops = {}
            if value is True:  # set streamplot options
                for attr in ['arrowsize', 'arrowstyle', 'density', 'linewidth',
                             'color', 'cmap', 'latlon']:
                    setattr(self, attr, getattr(self, attr))
                if self.linewidth is 0:
                    self.linewidth = None
            else:  # set quiver options
                for attr in ['color', 'cmap', 'rasterized', 'scale',
                             'linewidth', 'latlon']:
                    setattr(self, attr, getattr(self, attr))
                self._windplotops['units'] = 'xy'
                if self.linewidth is None:
                    self.linewidth = 0
                if np.any(self.density > 1.0):
                    warn("Reducing density to 1.0, because densities higher "
                         "than 1.0 are not allowed for quiver plots.")
                    self.density = 1.0

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def maskprop(self, x, doc):
        """property for masking options like maskabove or maskbelow"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            self._maskprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def mask_from_reader(self, x, doc):
        """property to return a True/False mask from a reader"""
        def getx(self):
            self.logger.debug("Get mask")
            mask = getattr(self, '_' + x)
            self.logger.debug("Mask arguments: %s" % mask)
            if mask is None:
                return None
            test = True
            if test:
                try:
                    self.logger.debug("Assume list and reader...")
                    if (hasattr(mask[0], 'get_data')
                            and hasattr(mask[0], 'lola_variables')):
                        reader = mask[0]
                        test = False
                except (TypeError, KeyError) as e:
                    self.logger.debug("    Failed", exc_info=True)
            if test:
                try:
                    self.logger.debug("Assume mapping...")
                    reader = auto_set_reader(**mask)
                    test = False
                except TypeError as e:
                    self.logger.debug("    Failed", exc_info=True)
            if test:
                try:
                    self.logger.debug(
                        "Assume list with first entry being a mapping...")
                    reader = auto_set_reader(**mask[0])
                    test = False
                except TypeError as e:
                    self.logger.debug("    Failed", exc_info=True)
            if test and not isinstance(mask[0], (str, unicode)):
                try:
                    self.logger.debug(
                        "Assume list with first entry being a list...")
                    reader = auto_set_reader(*mask[0])
                    test = False
                except (TypeError, IOError) as e:
                    self.logger.debug("    Failed", exc_info=True)
            if test:
                try:
                    self.logger.debug(
                        "Assume list with first entry being a satisfying "
                        "argument for the reader...")
                    reader = auto_set_reader(mask[0])
                    test = False
                except (TypeError, IOError) as e:
                    self.logger.debug("    Failed", exc_info=True)
            if test:  # assume a two-dimensional mask
                self.logger.debug("I assume a mask that matches")
                print mask
                return mask

            # choose variable
            vlst = reader.lola_variables.keys()
            if len(vlst) > 1:
                self.logger.debug(
                    "Found multiple variables in reader: %s" % ', '.join(vlst))
                try:
                    var = mask[1]
                except (TypeError, IndexError) as e:
                    raise ValueError(
                        "Found multiple variables in reader: %s! Specify "
                        "which to choose via the variable name in "
                        "[Reader kwargs[, variable name[, value]]]" % (
                            ', '.join(vlst)))
            else:
                var = vlst[0]

            # get data
            data = reader.get_data(var=var, datashape='2d')

            # set up final mask array
            try:
                value = mask[2]
                data[:] = np.invert(np.ma.make_mask(data[:] == value,
                                                    shrink=False))
            except (TypeError, IndexError, KeyError):
                data[:] = np.invert(np.ma.make_mask(data[:] != 0,
                                                    shrink=False))
            return data

        def setx(self, value):
            setattr(self, '_' + x, value)
            self._maskprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)

    def plotcbar(self, x, doc):
        """default colormap property"""
        def getx(self):
            return getattr(self, '_'+x)

        def setx(self, value):
            if value is True:
                value = ['b']
            elif not value:
                value = []
            else:
                pos = '(b)(r)(sh)(sv)'
                try:
                    for s in re.finditer('[^%s]+' % pos, value):
                        warn("Unknown colorbar position %s!" % s.group())
                    value = re.findall('[%s]' % pos, value)
                except TypeError:
                    value = list(value)
                    for s in (s for s in value
                              if not re.match('[%s]' % pos, s)):
                        warn("Unknown colorbar position %s!" % s)
                        value.remove(s)
            setattr(self, '_' + x, value)

            if x not in self._cmapprops:
                self._cmapprops.append(x)

        def delx(self):
            setattr(self, '_'+x, self._default[x])

        return property(getx, setx, delx, doc)
