# -*- coding: utf-8 -*-
"""readers module of the nc2map python module

This script contains the basic data danagement utilities in the nc2map
module.

It contains the following reader classes
   - The ReaderBase class defines the main methods for all readers,
       such as data extraction, the merging method and the arithmetics.
   - The NCReader class is a wrapper around a netCDF4.Dataset instance
       and implemented as a subclass of ReaderBase
   - The MFNCReader class is a wrapper around a netCDF4.MFDataset instance
       and implemented as a subclass of ReaderBase
   - The ArrayReader is a class mimiking the structure of the netCDF4.Dataset
       but without the storing of data in a file.

Furthermore it contains the DataField class, a wrapper around a
numpy.ma.MaskedArray with enhanced capabilities. And it contains the
Variable class which is the comparable version of the netCDF4.Variable class
but for ArrayReader instances."""
import os
import glob
import logging
from itertools import (izip, izip_longest, product, imap, chain, cycle, repeat,
                       tee)
from collections import OrderedDict
import datetime as dt
import numpy as np
import netCDF4 as nc
import gdal
from .warning import warn, critical, Nc2MapRuntimeWarning
import mpl_toolkits.basemap as bm
from matplotlib.tri import Triangulation, TriAnalyzer
from nc_utils import chunk_shape_3D
from .defaults import readers as defaults


defaultnames = defaults['dimnames']
readers = ['NCReader', 'MFNCReader', 'ArrayReader']


def auto_set_reader(*args, **kwargs):
    """Function to choose a reader automatically via try and error.
    Arguments and keyword arguments are passed directly to the reader class.
    Keyword arguments (beside the one for the reader initialization) are
      - readers: list of strings with reader names (if not the default readers
          shall be used). Otherwise the following default readers will be used:
    """
    # docstring is extended below
    logger = logging.getLogger("%s.auto_set_reader" % __name__)
    # check if input is a reader
    if len(args) == 1:
        logger.debug("Found one input argument --> Check if reader")
        data_reader = args[0]
        if (hasattr(data_reader, 'get_data')
                and hasattr(data_reader, 'lola_variables')):
            logger.debug(
                "Found get_data method and lola_variables --> Assume reader!")
            return data_reader
        else:
            logger.debug(
                        "Did not find get_data method and lola_variables in "
                        "input... Try now the different readers...")
    else:
        logger.debug("Found multiple arguments --> try different readers")

    try:
        test_readers = kwargs.pop('readers')
    except KeyError:
        test_readers = readers
    logger.debug("Set reader automatically. Order of trial is %s",
                 ', '.join(test_readers))
    success = False
    for reader in test_readers:
        try:
            logger.debug("Try %s...", reader)
            data_reader = globals()[reader](*args, **kwargs)
            logger.debug("Suceeded.")
            success = True
            break
        except Exception as e:
            logger.debug("Failed.", exc_info=True)
    if not success:
        raise IOError(
            "Could not open any reader with one of %s. Try manually!" %
            ', '.join(test_readers))
    return data_reader

class Icon_Triangles(object):
    def get_triangles(self, reader, varo=None, convert_spatial=True):
        """Get the longitude informations and triangles of an ICON-like grid

        This function extracts the triangles in an unstructered ICON-like[1]_
        grid. This grid consists of centered longitude informations, stored in
        variable *clon*, centered latitude informations, stored in variable
        *clat*, and the vortex coordinates, stored in variable *clon_vertices*
        and *clat_vertices*.

        Parameters
        ----------
        reader: :class:`~nc2map.ReaderBase` instance
            reader containing the grid informations
        varo: object
            variable object containing the data (only used for compatibility)
        convert_spatial: bool, optional
            Default: True. If this is True, and the spatial dimensions
            (latitudes, longitudes) are in radians, they are converted to
            degrees

        Returns
        -------
        lon: 1D-array of longitudes
        lat: 1D-array of latitudes
        triang: matplotlib.tri.Triangulation instance with the triangle
            definitions

        Raises
        ------
        nc2map.readers.GridError
            if `reader` does not have the above mentioned variables

        .. [1] Max-Planck-Institute for Meteorology, "ICON (Icosahedral
            non-hydrostatic) general circulation model",
            :ref:`http://www.mpimet.mpg.de/en/science/models/icon.html`,
            accessed June 23, 2015"""
        self.test(reader)
        clon = reader.variables['clon']
        clat = reader.variables['clat']
        clonv = reader.variables['clon_vertices']
        clatv = reader.variables['clat_vertices']
        triangles = np.reshape(range(len(clon)*3), (len(clon), 3))
        if convert_spatial:
            try:
                units = clon.units
            except AttributeError:
                units = None
            clon = reader.convert_spatial(clon, units, raise_error=False)
            clonv = reader.convert_spatial(clonv, units,
                                           raise_error=False).ravel()
            try:
                units = clat.units
            except AttributeError:
                units = None
            clat = reader.convert_spatial(clat, units, raise_error=False)
            clatv = reader.convert_spatial(clatv, units,
                                           raise_error=False).ravel()
        else:
            clon = clon[:]
            clat = clat[:]
            clonv = clonv[:].ravel()
            clatv = clatv[:].ravel()
        return clon, clat, Triangulation(clonv, clatv, triangles=triangles)

    def test(self, reader, varo=None):
        """Test the reader if it matches the conventions"""
        miss = {'clon', 'clat', 'clon_vertices', 'clat_vertices'} - set(
            reader.variables)
        if miss:
            raise GridError(
                "Missing grid variables: %s" % ', '.join(miss))

    def get_coords(self, reader, varo=None):
        self.test(reader)
        return {
            'clon': reader.variables['clon'],
            'clat': reader.variables['clat'],
            'clon_vertices': reader.variables['clon_vertices'],
            'clat_vertices': reader.variables['clat_vertices']}

class Ugrid_Triangles(object):
    def get_triangles(self, reader, varo, convert_spatial=True):
        """Get the longitude informations and triangles of an unstructured grid

        This function extracts the triangles in an unstructered grid that
        follows the Ugrid conventions.

        Parameters
        ----------
        reader: :class:`~nc2map.ReaderBase` instance
            reader containing the grid informations
        varo: object
            variable object containing the data (only used for compatibility)
        convert_spatial: bool, optional
            Default: True. If this is True, and the spatial dimensions
            (latitudes, longitudes) are in radians, they are converted to
            degrees

        Returns
        -------
        lon: 1D-array of longitudes
        lat: 1D-array of latitudes
        triang: matplotlib.tri.Triangulation instance with the triangle
        definitions

        Raises
        ------
        nc2map.readers.GridError
            if `reader` does not follow the Ugrid conventions"""
        (lonname, lon), (latname, lat), (triname, triangles) = self.test(
            reader, varo)
        if convert_spatial:
            lon = reader.convert_spatial(lon)
            lat = reader.convert_spatial(lat)
        return lon[:], lat[:], Triangulation(lon[:], lat[:],
                                            triangles=triangles[:])

    def test(self, reader, varo):
        """Tests the reader and returns grid informations"""
        try:
            mesh = varo.mesh
        except AttributeError:
            raise GridError("Variable does not have a mesh defined!")
        try:
            mesh = reader.variables[mesh]
        except KeyError:
            raise GridError("Mesh %s was not found in the reader!" % mesh)
        try:
            nodes = mesh.node_coordinates.split()[:2]
            if not len(nodes) == 2:
                raise GridError(
                    "Need two node_coordinates variables, but found only "
                    "one ({0})".format(nodes[0]))
        except AttributeError:
            raise GridError(
                "Topology variable does not have a valid (space-separated) "
                "node_coordinates attribute")
        try:
            lon = reader.variables[nodes[0]]
        except KeyError:
            raise GridError("Did not found variable %s in reader!" % nodes[0])
        try:
            lat = reader.variables[nodes[1]]
        except KeyError:
            raise GridError("Did not found variable %s in reader!" % nodes[1])
        try:
            triangles = mesh.face_node_connectivity
        except AttributeError:
            raise GridError(
                "Topology variable does not have a face_node_connectivity "
                "attribute!")
        try:
            triangles = reader.variables[triangles]
        except KeyError:
            raise GridError(
                "face_node_connectivity variable {0} was not found in the "
                "reader!".format(triangles))
        return [(nodes[0], lon), (nodes[1], lat),
                (mesh.face_node_connectivity, triangles)]

    def get_coords(self, reader, varo):
        coords = dict(self.test(reader, varo))
        mesh = varo.mesh
        coords[mesh] = reader.variables[mesh]
        return coords


ufuncs = [Ugrid_Triangles(), Icon_Triangles()]

def dimprop(x, doc):
    """Function which creates a dimension property
    Sets up the dimension by using self.nco and the given dimension name"""
    # ---- not used at the moment ----
    def getx(self):
        if getattr(self, x+'names') is None:
            return None
        return self.variables[getattr(self, x+'names')]

    return property(getx, doc=doc)


def dimlist(x):
    """Function which creates a property get the string out of the set
    value, which is also in the variables attribute"""
    def getx(self):
        names = [name for name in getattr(self, '_'+x)
                 if name in self.variables]
        if len(names) == 0:
            names = [None]
        elif len(names) > 1:
            raise ValueError(
                "Found multiple %s in the Reader: %s" % (
                    x, ', '.join(names)))
        return names[0]

    def setx(self, dims):
        if isinstance(dims, (str, unicode)):
            setattr(self, '_'+x, {dims})
        else:
            setattr(self, '_'+x, set(dims))

    def delx(self):
        delattr(self, '_'+x)

    doc = """
        Name of the %s dimension. Set it with a string or list of strings, get
        the dimension name in the reader as string""" % (
            x.replace('names', ''))

    return property(getx, setx, delx, doc)


def datadim(x):
    """Function returning a property that gets and sets values of x from and
    into the dims dictionary"""
    def getx(self):
        try:
            return self.dims[getattr(self, '_DataField'+x)]
        except ValueError:
            return None

    def setx(self, value):
        self.dims[getattr(self, '_DataField'+x)] = value

    def delx(self):
        del self.dims[getattr(self, '_DataField'+x)]
    doc = """Data of the %s dimension. See also dims property"""

    return property(getx, setx, delx, doc)


class GridError(Exception):
    pass

class Variable(object):
    """Variable object for an ArrayReader instance. The structure is
    essentially similar as for the netCDF4.Variable class.

    If var is your Variable instance, the data can be accessed in two ways:
       1) via the data attribute var.data (returns the pure numpy array)
       2) via __getitem__ var[...]
    In case 1), the data is accessed in the usual numpy indexing style (see
    http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html) whereas
    the second case follows the slicing rules of the netCDF4.Variable class,
    allowing one-dimensional boolean and integer sequences. One example (where
    lons might be a one-dimensional longitude array) might be
     >>> var[::2, [1,3,6], 4, lons>0]
    which is not possible for a usual numpy array. The same holds for setting
    the data.
    Note that using 2) will always create a copy of the data.

    Meta informations can be accessed via the explicit attribute (e.g.
    var.long_name) or via var.meta['long_name']. New meta informations can
    be in the same way: var.long_name = 'my long name' or
    var.meta['long_name'] = 'my long name'
    """

    __slots__ = ['data', 'var', 'dimensions', 'meta']

    @property
    def shape(self):
        """Return the shape of data"""
        return self.data.shape

    @property
    def dtype(self):
        """dtype of the variable"""
        return self.data.dtype

    def __init__(self, data=None, var='var', dims=('time', 'lat', 'lon'),
                 meta={}):
        """Initialization method for Variable instance

        Input:
            - data: numpy array
            - var: name of the variable
            - dims: tuple of dimension names (length of dims must match to
                length of data.shape)
            - meta: meta data
        """
        if data is not None and len(np.shape(data)) != len(tuple(dims)):
            try:
                raise ValueError((
                    "Shape of data (%s) and dimensions (%s) do not match!"
                    % (np.shape(data), dims)))
            except:
                raise ValueError((
                    "Shape of data (length %i) and dimensions (length %i) "
                    "do not match!") % (len(np.shape(data)),
                                        len(tuple(dims))))
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        self.data = data
        self.var = var
        self.dimensions = tuple(dims)
        self.meta = OrderedDict(meta).copy()

    def __getitem__(self, keys):
        """Set item method of Variable instance. Keys may be integers, slices
        or one dimensional integer or boolean arrays.

        For example
        >>> tempdat = nco.variables['t2m'][::2, [1,3,6], lats>0, lons>0]"""
        try:
            keys = list(keys)
        except TypeError:  # non-iterable, i.e. only one slice or integer
            return self.data[keys].copy()  # make sure that a copy is returned
        squeeze = []
        for i, key in enumerate(keys):
            if isinstance(key, slice):
                keys[i] = range(*key.indices(self.shape[i]))
            elif isinstance(key, int):
                keys[i] = [key]
                if self.data.ndim > 1:
                    squeeze.append(i)
            elif np.ndim(key) > 1:
                raise IndexError("Index cannot be multidimensional")
        if not squeeze:
            return self.data[np.ix_(*keys)]
        else:
            return np.squeeze(self.data[np.ix_(*keys)], squeeze)


    def __setitem__(self, keys, value):
        """Set item method of Variable instance. Keys may be integers, slices
        or one dimensional integer or boolean arrays.

        For example
        >>> tempdat = nco.variables['t2m'][::2, [1,3,6], lats>0, lons>0]"""
        try:
            keys = list(keys)
        except TypeError:  # non-iterable, i.e. only one slice or integer
            self.data[keys] = value
            return
        squeeze = []
        for i, key in enumerate(keys):
            if isinstance(key, slice):
                keys[i] = range(*key.indices(self.shape[i]))
            elif isinstance(key, int):
                keys[i] = [key]
                squeeze.append(i)
            elif np.ndim(key) > 1:
                raise IndexError("Index cannot be multidimensional")
        if not squeeze:
            self.data[np.ix_(*keys)] = value
        else:
            try:
                s = self.data[np.ix_(*keys)].shape
                self.data[np.ix_(*keys)] = np.reshape(value, s)
            except ValueError:
                self.data[np.ix_(*keys)] = value

    def __len__(self):
        return len(self.data)

    def __getattr__(self, attr):
        try:
            return self.meta[attr]
        except KeyError:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (
                    self.__class__.__name__, attr))

    def __setattr__(self, attr, value):
        if attr not in self.__class__.__slots__:
            getattr(self, 'meta')[attr] = value
        else:
            super(Variable, self).__setattr__(attr, value)

    def __dir__(self):
        return dir(super(Variable, self)) + self.meta.keys()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        strings = [
            super(Variable, self).__repr__(),
            "%s %s(%s)" % (self.data.dtype, self.var,
                           ', '.join(self.dimensions))]
        for item in self.meta.items():
            strings.append("    %s: %s" % item)
        strings.append("current shape: %s" % str(self.shape))
        return '\n'.join(strings)


class DataField(object):
    """Multidimensional Data Field with latitude, longitude, time and level
    dimension.
    This class is a wrapper around a numpy.ma.MaskedArray instance with
    additional grid informations.

    Data can be accessed via
        >>> mydatafield = DataField(...)
        >>> data = mydatafield[:]

    Additional attributes:
      - dims: numpy.ndarray containing grid informations (e.g. longitude data,
          etc.). The information can be accessed via the name of the dimension
          (e.g. data.dims['lon']) or data.dims[['lon', 'lat']]
      - dimensions: tuple of strings where each string stands for the dimension
          the specific axis belongs to

    Additional properties:
      - grid: returns 2-dimensional longitude- and latitude arrays
      - gridweights: Calculate grid weights from longitude and latitude
          informations in self.dims
      - lon: longitude dimension data
      - lat: latitude dimension data
      - time: time dimension data
      - level: level dimension data

    Additional methods:
      - mask_outside: masks the data outside of the region of a
          mpl_toolkits.basemap.Basemap instance
      - shift_data: shifts the data to match the longitude- latitude
          defintions of a mpl_toolkits.basemap.Basemap instance
      - mask_data: masks the data to match a given two-dimensional array with
          the same shape as the longitude and latitude dimension
      - fldmean: Computes the weighted mean over the longitude-latitude
          dimensions (or more)
      - fldstd: Computes the weighted standard deviation over the
          longitude-latitude dimensions (or more)
      - percentiles: Computes the weighted percentile over the
          longitude-latitude dimensions (or more)

    Additional methods:
      - fldmean
    """
    __slots__ = ['__data', '__lon', '__lat', '__time', 'dims', '__spatial',
                    'dimensions', '__level', '__var', 'logger', 'triangles']

    time = datadim('__time')
    level = datadim('__level')
    lon = datadim('__lon')
    lat = datadim('__lat')

    def __init__(self, data, var, dimensions, dims={}, lon='lon', lat='lat',
                 level='level', time='time', spatial_ax=None, triangles=None):
        """Initialization method of DataField2D instance

        Input:
          - data: two-dimensional data array
          - lon: longitude data
          - lat: latitude data
          - time: array with time information as datetime instances
          - level: array with level informations
          - dimensions: dimension names (e.g. ('lat', 'lon')) in the order as
              they appear in the shape of data.
          - spatial_ax: axes numbers with spatial information (used for
              fldmean, etc.)
          - triangles: matplotlib.tri.Triangulation instance for
              unstructered grids"""
        self.set_logger()
        self.__var = var
        self.logger.debug("Input arguments:")
        for arg in ['data', 'dimensions']:
            self.logger.debug("    %s: %s", arg, type(locals()[arg]))
        self.logger.debug("Input dimensions:")
        for key, val in dims.iteritems():
            self.logger.debug("    %s: %s", key, type(val))
        self.__data = data
        self.dimensions = list(dimensions)
        self.dims = self._dict_to_recarray(dims)
        self.check_dims()
        self.__lon = str(lon)
        self.__lat = str(lat)
        self.__level = str(level)
        self.__time = str(time)
        self.__spatial = spatial_ax
        self.triangles = triangles

    @property
    def grid(self):
        """Tuple (lat2d, lon2d), where lat2d is the 2 dimensional latitude and
        lon2d the 2 dimensional longitude corresponding to the data
        Input:
          - ilat: integer. Latitude axis in data.shape
          - ilon: integer. Longitude axis in data.shape
        Returns:
          lat2d, lon2d
        """
        if len(np.shape(self.lat)) > 1 or len(self.__spatial) == 1:
            return self.lat, self.lon
        else:
            ilat = self.dimensions.index(self.__lat)
            ilon = self.dimensions.index(self.__lon)
            if ilat > ilon:
                return np.meshgrid(self.lon, self.lat)
            else:
                return list(
                    np.roll(np.meshgrid(self.lon, self.lat), 1, axis=0))

    @property
    def gridweights(self):
        """Calculates weights from latitude and longitude informations.

        Please note that latitude and longitude are expected to be in
        degrees.
        Input:
          - ilat: integer. Latitude axis in data.shape
          - ilon: integer. Longitude axis in data.shape

        Returns:
          weights: 2 Dimensional array matching to lat2d and lon2d"""
        if len(self.__spatial) == 1:  # return equal weights
            tile_shape = list(self.shape)
            dims = list(self.dimensions)
            ispatial = self.__spatial[0]
            tile_shape[ispatial] = 1
            weights = np.ma.ones(self.shape)
            weights.mask = self.mask
            weights /= weights.sum(ispatial)
            return weights
        lat2d, lon2d = self.grid  # may also be 1d if len(self.__spatial) == 1
        ilat = self.dimensions.index(self.__lat)
        ilon = self.dimensions.index(self.__lon)
        ilat_orig = ilat
        ilon_orig = ilon
        if ilat > ilon:
            latslices = [(slice(None), i) for i in [0, 1, -1, -2]]
            ilat = 1
            lonslices = [(i, slice(None)) for i in [0, 1, -1, -2]]
            ilon = 0
        else:
            latslices = [(i, slice(None)) for i in [0, 1, -1, -2]]
            ilat = 0
            lonslices = [(slice(None), i) for i in [0, 1, -1, -2]]
            ilon = 1
        # interpolate to left and right center
        new_shape = [1 if i == ilon else lon2d.shape[ilat] for i in xrange(2)]
        lon2d = np.append(
            np.insert(
                lon2d, 0, (2*lon2d.__getitem__(lonslices[0]) -
                           lon2d.__getitem__(lonslices[1])), axis=ilon),
            np.reshape((2*lon2d.__getitem__(lonslices[2]) -
                        lon2d.__getitem__(lonslices[3])), new_shape),
            axis=ilon)
        lat2d = np.append(
            np.insert(
                lat2d, 0, (2*lat2d.__getitem__(lonslices[0]) -
                           lat2d.__getitem__(lonslices[1])), axis=ilon),
            np.reshape((2*lat2d.__getitem__(lonslices[2]) -
                        lat2d.__getitem__(lonslices[3])), new_shape),
            axis=ilon)
        # interpolate to upper and lower center
        new_shape = [1 if i == ilat else lat2d.shape[ilon] for i in xrange(2)]
        lon2d = np.append(
            np.insert(
                lon2d, 0, (2*lon2d.__getitem__(latslices[0]) -
                           lon2d.__getitem__(latslices[1])), axis=ilat),
            np.reshape((2*lon2d.__getitem__(latslices[2]) -
                        lon2d.__getitem__(latslices[3])), new_shape),
            axis=ilat)
        lat2d = np.append(
            np.insert(
                lat2d, 0, (2*lat2d.__getitem__(latslices[0]) -
                           lat2d.__getitem__(latslices[1])), axis=ilat),
            np.reshape((2*lat2d.__getitem__(latslices[2]) -
                        lat2d.__getitem__(latslices[3])), new_shape),
            axis=ilat)
        # calculate centered longitude bounds
        lon_bounds = np.array([
            np.mean([lon2d[:-2,:-2], lon2d[1:-1,1:-1]], axis=0),
            np.mean([lon2d[1:-1,1:-1], lon2d[2:,2:]], axis=0)])*np.pi/180.
        # calculate center latitude bounds
        lat_bounds = np.array([
            np.mean([lat2d[:-2,:-2], lat2d[1:-1,1:-1]], axis=0),
            np.mean([lat2d[1:-1,1:-1], lat2d[2:,2:]], axis=0)])*np.pi/180.

        weights = np.abs(lon_bounds[0,:] - lon_bounds[1,:])*(
            np.sin(lat_bounds[0,:]) - np.sin(lat_bounds[1,:]))
        # tile arrays to match
        tile_shape = list(self.shape)
        tile_shape[ilat_orig] = 1
        tile_shape[ilon_orig] = 1
        if hasattr(self, 'mask'):
            # tile
            weights = np.ma.array(np.tile(weights, tile_shape), mask=self.mask)
            # normilize now to consider for the mask
            for ind in self._iter_indices(ilat_orig, ilon_orig):
                weights.__setitem__(
                    ind,
                    weights.__getitem__(ind)/weights.__getitem__(ind).sum())
        else:
            # normalize
            weights /= weights.sum()
            # tile
            weights = np.tile(weights, tile_shape)
        if (weights < 0).any():
            raise ValueError(
                "Found negative weights!")
        return weights

    def _dict_to_recarray(self, dim_data):
        try:
            dim_data = dict(dim_data)
        except TypeError:
            return dim_data
        dims = map(str, frozenset(self.dimensions + dim_data.keys()))
        data = tuple(dim_data.get(dim) for dim in dims)
        dtypes = [np.asarray(dim_data.get(dim)).dtype for dim in dims]
        shapes = [np.asarray(dim_data.get(dim)).shape for dim in dims]
        dtype = zip(dims, dtypes, shapes)
        return np.array(data, dtype)

    def mask_outside(self, mapproj):
        """Mask data outside the boundary of a Basemap instance

        Input:
          - mapproj: mpl_toolkits.basemap.Basemap instance (or another object
              with attributes lonmin, lonmax, latmin and latmax)
        """
        self.logger.debug("Mask data to match %s", type(mapproj))
        for attr in ['lonmin', 'lonmax', 'latmin', 'latmax']:
            self.logger.debug("    %s: %s", attr, getattr(mapproj, attr))
        indices = self._iter_indices(*self.__spatial)
        lat2d, lon2d = self.grid  # may also be 1d if len(self.__spatial) == 1
        londata = self.lon
        latdata = self.lat
        if (londata < mapproj.lonmin).any():
            lonminmax = londata[londata < mapproj.lonmin].max()
        if (londata > mapproj.lonmax).any():
            lonmaxmin = londata[londata > mapproj.lonmax].min()
        if (latdata < mapproj.latmin).any():
            latminmax = latdata[latdata < mapproj.latmin].max()
        if (latdata > mapproj.latmax).any():
            latmaxmin = latdata[latdata > mapproj.latmax].min()
        for indextuple in indices:
            if (londata < mapproj.lonmin).any():
                self.__setitem__(
                    indextuple, np.ma.masked_where(
                        lon2d < lonminmax, self.__getitem__(indextuple),
                        copy=True))
            if (londata > mapproj.lonmax).any():
                self.__setitem__(
                    indextuple,  np.ma.masked_where(
                        lon2d > lonmaxmin, self.__getitem__(indextuple),
                        copy=True))
            if (latdata < mapproj.latmin).any():
                self.__setitem__(
                    indextuple, np.ma.masked_where(
                        lat2d < latminmax, self.__getitem__(indextuple),
                        copy=True))
            if (latdata > mapproj.latmax).any():
                self.__setitem__(
                    indextuple, np.ma.masked_where(
                        lat2d > latmaxmin, self.__getitem__(indextuple),
                        copy=True))
        return self

    def _iter_indices(self, *dims):
        """Returns an iterator over all axes except those specified in *dims"""
        dims = list(dims)
        for i, dim in enumerate(dims):
            try:
                dims[i] = self.dimensions.index(dim)
            except ValueError:
                pass
        iter_dims = [i for i in xrange(self.ndim) if i not in dims]
        for l in imap(list,
                      product(*(range(self.shape[i]) for i in iter_dims))):
            for dim in dims:
                l.insert(i, slice(None))
            yield tuple(l)

    def mask_data(self, mask):
        """Method to mask the data array from a given boolean array. The array
        must match to the shape of the longitude and latitude axis"""
        indices = self._iter_indices(*self.__spatial)
        for indextuple in indices:
            self.__setitem__(
                    indextuple,
                    np.ma.masked_where(mask, self.__getitem__(indextuple),
                                       copy=True))
        return self

    def fldmean(self, weights=None, axis='spatial', weighted=True,
                keepdims=False):
        """Returns the fldmean over the axis specified by the given dimensions.
        Supports masked array.

          - weights: alternative weights to use (if None, the
            self.gridweights property is used). The shape has to match
            self.shape!
          - axis: list of dimensions as they are used in self.dimensions or
            None, or an integer or tuple of integers standing for the array
            axis. If 'spatial', it will be replaced by the spatial axis
          - weighted: True or False. If False, no weighting is used and no
            weights are computed
          - keepdims: bool, optional. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the
            original array.

        Returns:
          The weighted average over the specifid axis.
        """
        if not weighted:
            weights = None
        elif weights is None:
            weights = self.gridweights
        if np.all(axis == 'spatial'):
            axis = self.__spatial
        try:
            axis = list(axis)
            dims = list(self.dimensions)
            for i, ax in enumerate(axis):
                if ax in dims:
                    axis[i] = dims.index(ax)
            axis = tuple(axis)
        except TypeError:
            pass
        mean = np.ma.average(self[:], weights=weights, axis=axis)
        if not keepdims:
            return mean
        else:
            if axis is not None:
                new_shape = list(self.shape)
                try:
                    for i in axis:
                        new_shape[i] = 1
                except TypeError:
                    new_shape[axis] = 1
            else:
                new_shape = [1] * self.ndim
            return mean.reshape(new_shape)

    def fldstd(self, weights=None, axis='spatial', weighted=True,
               keepdims=False):
        """Returns the standard deviation over the axis specified by the given
        dimensions.
        Supports masked array.

          - weights: alternative weights to use (if None, the
            self.gridweights property is used). The shape has to match
            self.shape!
          - axis: list of dimensions as they are used in self.dimensions or
            None, or an integer or tuple of integers standing for the array
            axis. If 'spatial', it will be replaced by the spatial axis
          - weighted: True or False. If False, no weighting is used and no
            weights are computed
          - keepdims: bool, optional. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the
            original array.

        Returns:
          The weighted standard deviation over the specifid axis.
        """
        if not weighted:
            weights = None
        elif weights is None:
            weights = self.gridweights
        if np.all(axis == 'spatial'):
            axis = self.__spatial
        try:
            axis = list(axis)
            dims = list(self.dimensions)
            for i, ax in enumerate(axis):
                if ax in dims:
                    axis[i] = dims.index(ax)
                elif ax < 0:
                    axis[i] += self.ndim
            axis = tuple(axis)
        except TypeError:
            if axis is not None and axis < 0:
                axis += self.ndim
        if weights is None:
            return np.ma.std(self[:], axis=axis)
        wtot = weights.sum(axis)
        mean = self.fldmean(weights, axis, keepdims=True)
        rshape = list(self.shape)
        if axis is not None:
            try:
                for i in xrange(self.ndim):
                    if i not in axis:
                        rshape[i] = 1
            except TypeError:
                for i in xrange(self.ndim):
                    if i != axis:
                        rshape[i] = 1
        mean = np.tile(mean, rshape)
        std = np.ma.sqrt(np.sum(weights*(self[:]-mean)**2, axis=axis)/wtot)
        if not keepdims:
            return std
        else:
            if axis is not None:
                new_shape = list(self.shape)
                try:
                    for i in axis:
                        new_shape[i] = 1
                except TypeError:
                    new_shape[axis] = 1
            else:
                new_shape = [1] * self.ndim
            return std.reshape(new_shape)


    def percentile(self, q, weights=None, axis='spatial',
                    keepdims=False, weighted=True):
        """ Very close to numpy.percentile, but supports weights and masked
        arrays.

        Input:
          - q: float in range of [0,100] (or sequence of floats)
            Percentile to compute which must be between 0 and 100 inclusive.
          - weights: alternative weights to use (if None, the
            self.gridweights property is used). The shape has to match
            self.shape. During the calculation, the weights are normalized
            along the specified axis.
          - axis: int or sequence of strings and int, optional. Axis along
            which the percentiles are computed. Strings must match a name in
            self.dimensions. If axis is None, the percentiles are computed
            along a flattened version of the array. If 'spatial', it will be
            replaced by the spatial axis
          - weighted: True or False. If False, no weighting is used and no
            weights are computed
          - keepdims: bool, optional. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the
            original array.

        Returns
        percentile: scalar or ndarray
        If a single percentile `q` is given and axis=None a scalar is
        returned.  If multiple percentiles `q` are given an array holding
        the result is returned. The results are listed in the first axis.
        (If `out` is specified, in which case that array is returned
        instead).  If the input contains integers, or floats of smaller
        precision than 64, then the output data-type is float64. Otherwise,
        the output data-type is the same as that of the input.

        quantile weights that are used for computing the percentiles, are
        computed via the cumulative sum (nweights are the normalized weights)
            quantile_weights = np.cumsum(nweights, axis=axis) - 0.5 * nweights
        Percentiles are linearly interpolated using the np.interp function.
        """
        data = self[:].copy()
        q = np.array(q, dtype=float).copy()
        if not q.ndim:
            q = np.array([q])
            reduce_shape = True if not keepdims else False
        else:
            reduce_shape = False
        if np.all(axis == 'spatial'):
            axis = self.__spatial
        try:
            axis = list(axis)
            dims = list(self.dimensions)
            for i, ax in enumerate(axis):
                if ax in dims:
                    axis[i] = dims.index(ax)
            axis = tuple(axis)
            reshape = True
        except TypeError:
            reshape = False
            try:
                if axis is not None and axis < 0:
                    axis = data.ndim - axis
            except TypeError:
                pass

        if not weighted:
            weights = np.ma.array(np.ones(data.shape), mask=data.mask)
        elif weights is None:
            weights = self.gridweights
        else:
            if not np.ndim(weights) == data.ndim or not np.all(
                    np.shape(weights) == data.shape):
                raise ValueError(
                    "Shape of weights (%s) has to match the shape of data "
                    "(%s)!" % (np.shape(weights), data.shape))
        weights = np.ma.array(weights, mask=data.mask).copy()

        if not (np.all(q >= 0) and np.all(q <= 100)):
            raise ValueError('q should be in [0, 100]')
        if np.any(weights < 0):
            raise ValueError('Weights must not be smaller than 0!')
        q /= 100.
        if reshape:
            for ax in axis:
                data = np.rollaxis(data[:], ax, 0)
                weights = np.rollaxis(weights[:], ax, 0)
            data = data.reshape([np.product(data.shape[:len(axis)])] + list(
                data.shape[len(axis):]))
            weights = weights.reshape([np.product(weights.shape[:len(axis)])] \
                + list(weights.shape[len(axis):]))
            axis = 0
        elif axis is not None:
            data = np.rollaxis(data[:], axis, 0)
            weights = np.rollaxis(weights[:], axis, 0)
            axis = 0
        if axis is None:
            data = data.ravel()
            weights = weights.ravel()
            sorter = np.ma.argsort(data)
            data = data[sorter]
            weights = weights[sorter]
        else:
            sorter = np.ma.argsort(data, axis=axis)
            indices = map(list, product(*(
                range(ndim) for i, ndim in enumerate(data.shape)
                if i != axis)))
            for ind in indices:
                ind.insert(axis, slice(None))
                data.__setitem__(
                    ind, data.__getitem__(ind)[
                        sorter.__getitem__(ind)])
                weights.__setitem__(
                    ind, weights.__getitem__(ind)[
                        sorter.__getitem__(ind)])

        weights /= weights.sum(axis)  # normalize weights
        weights = np.ma.cumsum(weights, axis=axis) - 0.5 * weights

        if axis is None:
            mask = data.mask == False
            pctl = np.interp(q, weights[mask], data[mask])
        else:
            indices = imap(list, product(*(
                range(ndim) for i, ndim in enumerate(data.shape)
                if i != axis)))
            indices2 = imap(list, product(*(
                range(ndim) for i, ndim in enumerate(data.shape)
                if i != axis)))
            pctl = np.zeros([len(q)] + [
                s for i, s in enumerate(data.shape) if i != axis])
            for ind1, ind2 in izip(indices, indices2):
                ind2.insert(axis, slice(None))
                mask = data.mask.__getitem__(ind2) == False
                pctl.__setitem__(tuple([slice(None)] + ind1), np.interp(
                    q, weights.__getitem__(ind2)[mask],
                    data.__getitem__(ind2)[mask]))
        if reduce_shape:
            pctl = pctl[0]
        return pctl

    def shift_data(self, mapproj):
        """Shift the data to match the Basemap instance boundaries

        Input:
           - mapproj: mpl_toolkits.basemap.Basemap instance (or another object
               with attributes lonmin and lonmax)
        """
        def shift_to_larger(indextuple, val):
            """shift to larger longitudes if lonmin < mapproj.lonmin"""
            if lonmax < val:
                data, lon = iter(bm.shiftgrid(
                lonmax, self.__getitem__(indextuple), lonold))
            else:
                data, lon = self.__getitem__(indextuple), lonold
            shifteddata = iter(bm.shiftgrid(
                val, self.__getitem__(indextuple), lon))
            self.__setitem__(indextuple, next(shifteddata))
            self.lon = next(shifteddata)

        def shift_to_smaller(indextuple, val):
            """shift to smaller longitudes if lonmax > mapproj.lonmax"""
            if lonmin > val:
                data, lon = iter(bm.shiftgrid(
                lonmin, self.__getitem__(indextuple), lonold,
                start=False))
            else:
                data, lon = self.__getitem__(indextuple), lonold
            shifteddata = iter(bm.shiftgrid(
                val, data, lon,
                start=False))
            self.__setitem__(indextuple, next(shifteddata))
            self.lon = next(shifteddata)
        self.logger.debug("Shift data to match %s", type(mapproj))
        # shift data
        if len(self.lon.shape) == 1:
            self.logger.debug("    Longitude is 1d --> shift")
            if len(self.__spatial) == 1:
                self.logger.debug(
                    "    Found one-dimensional data --> ")
                if mapproj.lonmin < 0:
                    self.logger.debug("        decrease all > 180.")
                    self.lon[self.lon > 180.] -= 360.
                elif mapproj.lonmax > 180.:
                    self.logger.debug("        increase all < 0.")
                    self.lon[self.lon < 0.] += 360.
                return self
            # shiftgrid does only support 2 dimensional arrays. Therefore we
            # loop through the other indices
            indices = imap(list, product(*(
                range(ndim) for i, ndim in enumerate(self.shape)
                if self.dimensions[i] not in [self.__lat, self.__lon])))
            lonold = self.lon.copy()
            lonmin = lonold.min()
            lonmax = lonold.max()
            self.logger.debug("    Minimum longitude of data: %s", lonmin)
            self.logger.debug("    Minimum longitude of Basemap: %s",
                              mapproj.lonmin)
            self.logger.debug("    Maximal longitude of data: %s", lonmax)
            self.logger.debug("    Maximal longitude of Basemap: %s",
                              mapproj.lonmax)
            if lonmin <= mapproj.lonmin:
                val = lonold[lonold <= mapproj.lonmin].max()

                self.logger.debug("    --> Shift to the right to %s", val)
                shift = lambda indextuple: shift_to_larger(indextuple, val)
                shift_lon = True
            elif lonmax >= mapproj.lonmax:
                val = lonold[lonold >= mapproj.lonmax].min()
                self.logger.debug("   --> Shift to the left to %s", val)
                shift = lambda indextuple: shift_to_smaller(indextuple, val)
                shift_lon = True
            else:
                self.logger.debug("    Longitude 1d but no shift necessary")
                shift_lon = False
            ilat = self.dimensions.index(self.__lat)
            ilon = self.dimensions.index(self.__lon)
            if shift_lon:
                for i, indextuple in enumerate(indices):
                    indextuple.insert(ilat, slice(None))
                    indextuple.insert(ilon, slice(None))
                    shift(indextuple)
                self.logger.debug("    Performed shifts in total: %i", i+1)
        else:
            self.logger.debug("    Longitude is not 1d --> no shift")
        return self

    def __getitem__(self, key):
            return self.__data[key]

    def __setitem__(self, key, item):
        self.__data[key] = item

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__.keys():
            return getattr(self, attr)
        else:
            return getattr(self.__data, attr)

    def __dir__(self):
        return dir(super(DataField, self)) + dir(self.__data)

    def __len__(self):
        return "%i dimensional DataField instance of %s" % (
            len(self.dimensions), self.__var)

    def __str__(self):
        return repr(self)[1:-1]

    def check_dims(self, raise_error=False):
        """Function to check whether the data, it's shape and the given
        dimensions match."""
        shape = self.__data.shape
        nshape = len(shape)
        ndims = len(self.dimensions)
        self.logger.debug("Checking dimensions and shapes")
        self.logger.debug("Dimensions: %s", ', '.join(self.dimensions))
        if nshape != ndims:
            msg = (
                "Shape of data (%i) does not match to shape of specified "
                "dimensions (%i)!") % (nshape, ndims)
            if raise_error:
                raise ValueError(msg, logger=self.logger)
            else:
                critical(msg, logger=self.logger)

        for idim, dim in enumerate(self.dimensions):
            try:
                dimlen = len(self.dims[dim])
                if shape[idim] != dimlen:
                    msg = (
                        "Length of dimensions data for %s (%i) does not match "
                        "to the shape (%i).") % (dim, dimlen, shape[idim])
                    if raise_error:
                        raise ValueError(msg, logger=self.logger)
                    else:
                        critical(msg, logger=self.logger)
            except (KeyError, TypeError):
                msg = "Did not find dimension %s in dimension data!" % dim
                if raise_error:
                    raise ValueError(msg)
                else:
                    critical(msg, logger=self.logger)

    def set_logger(self, name=None, force=False):
        """This function sets the logging.Logger instance in the MapsManager
        instance.
        Input:
          - name: name of the Logger (if None: it will be named like
             <module name>.<class name>)
          - force: True/False (Default: False). If False, do not set it if the
              instance has already a logger attribute."""
        if name is None:
            name = '%s.%s' % (self.__module__, self.__class__.__name__)
        if not hasattr(self, 'logger') or force:
            self.logger = logging.getLogger(name)
            self.logger.debug('Initializing...')


class ReaderBase(object):
    """Base class defining the principle methods for nc2map.readers

    Parameters
    ----------
    meta: dict
        Global meta data of the ArrayReader instance
    timenames: set of strings
        Dimension and variable names that shall be considered as time
        dimension or variable
    levelnames: set of strings
        Dimension and variable names that shall be considered as level
        dimension or variable
    lonnames: set of strings
        Dimension and variable names that shall be considered as longitude
        dimension or variable
    latnames: set of strings
        Dimension and variable names that shall be considered as latitude
        dimension or variable
    udims: set of strings
        Dimension names that indicates that the variable is defined on an
        unstructured grid
    ufuncs: list
        list containing interpretation instances for unstructered grids
        (see below). Default grid interpretation instances are for
        the ugrid conventions of triangular grids and for the ICON grid.
    **data
            var={'data': arr, 'dims': (dim1, dim2, ...)}}
            var is a string standing for the variable name,
            value of 'data' is the data array of the variable, value of
            'dims' is a list of dimension names. Each dimension name must
            correspond to the specific axes in arr.shape

    Attributes
    ----------
    lonnames: name of the longitude variable
    lon: longitude variable
    latnames: name of the latitude variable
    lat: latitude variable
    timenames: name of the time variable
    time: time variable
    levelnames: name of the level variable
    level: level variable
    udims: set of dimensions that identify an unstructered variable
    ufuncs: list of instances that are used the interpretation of an
        unstructured grid
    variables: dictionary containing the variables


    Notes
    -----
    instances in `ufunc` must have a get_triangles method accepting three
    parameters: a reader, a variable and a boolean. They must furthermore
    return the centered longitudes, latitudes and a
    matplotlib.tri.Triangulation instance with the triangle definitions.

    See Also
    --------
    nc2map.readers.get_triangle_ugrid: interpretation function for UGRID
        convention
    nc2map.readers.get_triangle_icon: ICON interpretation function"""

    # ----- property definitions
    lon = dimprop('lon', "Longitude variable (if found)")
    lat = dimprop('lat', "Latitude variable (if found)")
    time = dimprop('time', "Time variable data (if found)")
    level = dimprop('level', "Level variable data (if found)")

    timenames = dimlist('timenames')
    levelnames = dimlist('levelnames')
    latnames = dimlist('latnames')
    lonnames = dimlist('lonnames')

    @property
    def lola_variables(self):
        """Dictionary with variables containing longitude and latitude
        dimension"""
        return OrderedDict([
            item for item in self.variables.items()
            if (self._lonnames.intersection(item[1].dimensions)
                and self._latnames.intersection(item[1].dimensions)
                and item[0] not in [self.lonnames, self.latnames])])

    @property
    def grid_variables(self):
        """Dictionary with variables being latitude, longitude, time or
        level. Latitude dimension is stored in lat, longitude in lon, time
        in time and level in level."""
        dimensions = frozenset(chain(*(
            var.dimensions for var in self.variables.values())))
        return OrderedDict([
            (var, self.variables.get(var)) for var in dimensions])

    @property
    def time_variables(self):
        """Dictionary with variables containing the time dimension"""
        return OrderedDict([
            item for item in self.variables.items()
            if self.timenames in item[1].dimensions])

    @property
    def level_variables(self):
        """Dictionary with variables containing the time dimension"""
        return OrderedDict([
            item for item in self.variables.items()
            if self.levelnames in item[1].dimensions])

    @property
    def dttime(self):
        """Time array with datetime.datetime instances"""
        time = self.time
        if time is None:
            raise ValueError("Could not find time variable with name %s" %
                             self._timenames)
        return self.convert_time(self.time)

    def __init__(self, meta={}, timenames=defaultnames['timenames'],
                 levelnames=defaultnames['levelnames'],
                 lonnames=defaultnames['lonnames'],
                 latnames=defaultnames['latnames'],
                 udims=defaultnames['udims'], ufuncs=ufuncs,
                 **data):
        """Initialization method for ArrayReader instance

        Parameters
        ----------
        meta: dict
            Global meta data of the ArrayReader instance
        timenames: set of strings
            Dimension and variable names that shall be considered as time
            dimension or variable
        levelnames: set of strings
            Dimension and variable names that shall be considered as level
            dimension or variable
        lonnames: set of strings
            Dimension and variable names that shall be considered as longitude
            dimension or variable
        latnames: set of strings
            Dimension and variable names that shall be considered as latitude
            dimension or variable
        udims: set of strings
            Dimension names that indicates that the variable is defined on an
            unstructured grid
        ufuncs: list
            list containing interpretation instances for unstructered grids
            (see below). Default grid interpretation instances are for
            the ugrid conventions of triangular grids and for the ICON grid.
        **data
             var={'data': arr, 'dims': (dim1, dim2, ...)}}
             var is a string standing for the variable name,
             value of 'data' is the data array of the variable, value of
             'dims' is a list of dimension names. Each dimension name must
             correspond to the specific axes in arr.shape

        Notes
        -----
        instances in `ufunc` must have a get_triangles method that accepts
        three parameters, a reader, a variable and a boolean. They must
        furthermore return the centered longitudes, latitudes and a
        matplotlib.tri.Triangulation instance with the triangle definitions.
        If they cannot interprete the grid, a
        :class:`~nc2map.readers.GridError` should be raised.

        See Also
        --------
        nc2map.readers.get_triangle_ugrid: interpretation function for UGRID
            convention
        nc2map.readers.get_triangle_icon: ICON interpretation function
        """
        self.set_logger()
        self.timenames = timenames
        self.levelnames = levelnames
        self.lonnames = lonnames
        self.latnames = latnames
        self.udims = udims
        self.ufuncs = ufuncs
        self.meta = OrderedDict(meta).copy()
        self.logger.debug("Dimension names:")
        for attr in ['timenames', 'levelnames', 'lonnames', 'latnames',
                     'udims']:
            self.logger.debug("    %s: %s", attr, locals()[attr])
        self.variables = OrderedDict()
        for var, var_dict in data.items():
            vardims = var_dict['dims']
            self.variables[var] = Variable(
                var_dict['data'], var, var_dict['dims'],
                meta=var_dict.get('meta', {}))
        self.logger.debug("Dimensions found:")
        for attr in ['timenames', 'levelnames', 'lonnames', 'latnames']:
            self.logger.debug(
                "    %s as %s dimension.", getattr(self, attr),
                attr.replace('names', ''))

    def get_coords(self, varo):
        """Return the coordinates as a dictionary corresponding to a variable

        Parameters
        ----------
        varo: object
            :class:`~nc2map.readers.Variable` or netCDF4.Variable instance

        Returns
        -------
        dict: dictionary with keys being coordinate names, and values the
            variable"""
        if not self._udim(varo):
            return {item for item in self.grid_variables if item[0] in
                    varo.dimensions}
        else:
            for ini in self.ufuncs:
                self.logger.debug("    Try %s", ini.__class__.__name__)
                try:
                    coords = ini.get_coords(self, varo)
                    for dim in set(
                            varo.dimensions).intersection(self.variables):
                        coords[dim] = self.variables[dim]
                    return coords

                except GridError:
                    self.logger.debug("    Failed.", exc_info=True)
            raise GridError(
                "No class could interprete the unstructered grid!")



    def convert_time(self, times):
        """Converts the time variable instance into array of datetime
        instances.

        Supports relative (e.g. days since 1989-6-15 12:00) and absolute time
        units (day as %Y%m%d.%f)"""
        if isinstance(times[0], dt.datetime):
            return times[:]
        if not hasattr(times, 'units'):
            raise ValueError("Could not determine units of time variable")
        if not hasattr(times, 'calendar'):
            warn("Could not determine calendar. Hence I assume the 'standard' "
                 "calendar.", Nc2MapRuntimeWarning)
            calendar = 'standard'
        else:
            calendar = times.calendar
        try:  # try interpretation of relative time units
            self.logger.debug("Try netCDF4.num2date function")
            dts = nc.num2date(times[:], units=times.units, calendar=calendar)
        except ValueError:  # assume absolute time units
            self.logger.debug("Failed. Test for absolute time...", exc_info=1)
            if not times.units == 'day as %Y%m%d.%f':
                raise ValueError("Could not interprete time units %r" %
                                 times.units)
            days = np.floor(times[:]).astype(int)
            subdays = times[:] - days
            days = np.array(map(lambda x: "%08i" % x, days))
            dts = np.array(
                map(lambda x: (dt.datetime.strptime(x[0], "%Y%m%d") +
                               dt.timedelta(days=x[1])),
                    zip(days, subdays)))
        return np.array(map(np.datetime64, dts))

    def get_triangles(self, varo, convert_spatial,
                      min_circle_ratio=defaults['min_circle_ratio']):
        """Method to get the unstructered grid

        Parameters
        ----------
        varo: object
            variable object containing the data (only used for compatibility)
        convert_spatial: bool, optional
            Default: True. If this is True, and the spatial dimensions
            (latitudes, longitudes) are in radians, they are converted to
            degrees
        min_circle_ratio: float, optional
            Minimal circle ratio. If not 0, the
            maplotlib.tri.TriAnalyzer.get_flat_tri_mask method is used to
            mask very flat triangles. Defaults to
            :attr:`nc2map.defaults.readers`['min_circle_ratio']

        Returns
        -------
        lon: 1D-array of longitudes
        lat: 1D-array of latitudes
        triang: matplotlib.tri.Triangulation instance with the triangle
            definitions

        See Also
        --------
        ufuncs: List of classes that are used for the interpretation of
            unstructered grids, each standing for a different convention.

        Notes
        -----
        This method is used by the
        :meth:`~nc2map.readers.ReaderBase.get_data` method."""
        self.logger.debug("Interprete unstructered grid...")
        for ini in self.ufuncs:
            self.logger.debug("    Try %s", ini.__class__.__name__)
            try:
                lon, lat, triang = ini.get_triangles(self, varo,
                                                     convert_spatial)
                if min_circle_ratio:
                    tria = TriAnalyzer(triang)
                    triang.set_mask(
                        tria.get_flat_tri_mask(min_circle_ratio))
                return lon, lat, triang
            except GridError:
                self.logger.debug("    Failed.", exc_info=True)
        raise GridError(
            "No class could interprete the unstructered grid!")

    def convert_spatial(self, varo, units=None, raise_error=True):
        """Converts radians to degrees

        Parameters
        ----------
        varo: object
            A variable object (e.g. nc2map.readers.Variable or
            netCDF4.Variable)
        raise_error: bool
            Raise an error if `varo` does not have a units attribute

        Returns
        -------
        arr: dimension data in degrees

        Raises
        ------
        ValueError
            If `varo` does not have a units attribute and `raise_error`

        Note
        ----
        This method only calculates if varo.units == 'radian'"""
        self.logger.debug("Converting spatial dimension %s" % varo)
        try:
            units = varo.units
        except AttributeError:
            if units is None:
                raise ValueError(
                    "Could not determine units of the spatial variable")
        if units == 'radian':
            self.logger.debug("    Found radians")
            out = varo[:] * 180./np.pi
        else:
            self.logger.debug("    No radians")
            out = varo[:]
        return out

    def set_logger(self, name=None, force=False):
        """This function sets the logging.Logger instance in the MapsManager
        instance.
        Input:
          - name: name of the Logger (if None: it will be named like
             <module name>.<class name>)
          - force: True/False (Default: False). If False, do not set it if the
              instance has already a logger attribute."""
        if name is None:
            name = '%s.%s' % (self.__module__, self.__class__.__name__)
        if force or not hasattr(self, 'logger'):
            self.logger = logging.getLogger(name)
            self.logger.debug('Initializing...')

    def _udim(self, var):
        """Test if the variable is unstructured"""
        udim = self.udims.intersection(var.dimensions)
        udim = None if not udim else list(udim)[0]
        return udim

    def get_time_slice(self, index):
        """Gets the time slice by using the numpy.datetime64 class and
        returns the index using the numpy.searchsorted function
        Input:
          - Index: list or list of objects suitable for the np.datetime64
              routine. Possibilities are
              -- Integer or slice (than nothing happens and they are
                  returned)
              -- datetime.datetime instances
              -- numpy.datetime64 instances
              -- isoformat ('YYYY-mm-ddTHH:MM:SS') strings or part of them
                  (e.g. '2005' will be interpreted as year 2005, '2005-03'
                  will be interpreted as March, 2005)
        """
        if isinstance(index, (int, slice)):
            return index
        try:
            if isinstance(index[0], (int, slice)):
                if any(not isinstance(idx, (int, slice)) for idx in index):
                    raise ValueError(
                        "Some but not all values are integers or slices!")
                return index
        except (TypeError, IndexError):
            pass
        if self.dttime is None:
            raise ValueError(
                "Could not find (or interpret) time variable in Reader!")
        times = self.dttime
        try:  # try isoformat
            t = np.datetime64(index).astype(times.dtype)
        except ValueError:
            try:
                t = np.array(map(np.datetime64, index)).astype(times.dtype)
            except TypeError:
                raise ValueError("Could not interpret time information!")
        return times.searchsorted(t)

    def merge(self, *args, **kwargs):
        """Merge multiple readers into one.
        Arguments may be instances of the ArrayReader class
        Keyword arguments may be
          - copy: True/False (Default: False). If True, the data is copied.
          - close: True/False (Default: False). If True, the old reader
              instances are closed

        Please note:
          1.) All readers must have the same grid
          2.) Only one of the following can be fullfilled
              a.) Each has different variables
              b.) Each has different time steps
              c.) Each has different levels
        """
        def check_dims(*readers):
            """Checks whether the reader dimensions match to this one and
            prints warnings and raises errors
            Input:
            - reader: ArrayReader instance
            Output:
            - dictionary with matches"""
            # ---- check grid sizes ----
            readers = list(readers) + [self]
            lon_reader = [reader for reader in readers if reader.lon]
            if lon_reader:
                lon_reader = lon_reader[0]
                for reader in readers:
                    if reader == lon_reader or not reader.lon:
                        continue
                    if len(lon_reader.lon) == len(reader.lon):
                        # raise warning if lens match anyway
                        if np.any(lon_reader.lon[:] != reader.lon[:]):
                            critical(
                                "Attention! Only size of longitudinal grid of"
                                " %s matches!" % type(reader))
                    else:
                        raise ValueError(
                            "Longitudes of %s do not match!" % (
                                type(reader)))
            lat_reader = [reader for reader in readers if reader.lat]
            if lat_reader:
                lat_reader = lat_reader[0]
                for reader in readers:
                    if reader == lat_reader or not reader.lat:
                        continue
                    if len(lat_reader.lat) == len(reader.lat):
                        if np.any(lat_reader.lat[:] != reader.lat[:]):
                            # raise warning if lens match anyway
                            critical(
                                "Attention! Only size of latitudinal grid of "
                                "%s matches!" % type(reader))
                    else:
                        raise ValueError(
                            "Latitudes of %s do not match!" % (
                                type(reader)))
            checks = {}
            variables = [
                set(reader.variables.keys()) -
                set(reader.grid_variables.keys())
                for reader in readers]
            all_times = [set(reader.dttime) for reader in readers
                         if reader.time is not None]
            all_levels = [set(reader.level[:]) for reader in readers
                          if reader.level is not None]
            for key, base_dims in [('variables', variables),
                                   ('times', all_times),
                                   ('levels', all_levels)]:
                self.logger.debug("Check if %s match...", key)
                if not base_dims:
                    continue
                for i, dims in enumerate(base_dims):
                    for j, dims2 in enumerate(base_dims):
                        if i == j:
                            continue
                        if dims.isdisjoint(dims2):
                            self.logger.debug(
                                "Reader %i does not match reader %i", i, j)
                            checks[key] = False
                        else:
                            checks[key] = True
                            break
            return checks, readers
        self.logger.debug("Start merging readers...")
        self.logger.debug("Input:")
        for i, reader in enumerate([self] + list(args)):
            self.logger.debug("--------- Reader %i ---------", i)
            self.logger.debug("%s", reader)
        copy = kwargs.get('copy', False)
        close = kwargs.get('close', False)
        checks, readers = check_dims(*args)
        false_checks = len([check for check in checks.values() if not check])
        if false_checks != 1:
            if not false_checks:
                raise ValueError(
                    "Don't now how to merge the readers! The must have either "
                    "all different variables, times or levels!")
            raise ValueError(
                "I can either merge different variables, different times or "
                "different levels, but not different %s simultaneously!" % (
                    ' and '.join(
                        key for key, val in checks.items() if not val)))

        data = {}
        if not checks['variables']:
            for reader in readers:
                for var, obj in reader.variables.items():
                    if copy:
                        data[var] = {
                            'data': obj[:].copy(), 'dims': obj.dimensions[:],
                            'meta': reader.get_meta(var=var).copy()}
                    else:
                        data[var] = {
                            'data': obj[:], 'dims': obj.dimensions[:],
                            'meta': reader.get_meta(var=var)}
        elif not checks['times']:
            stime = self.timenames
            for var, obj in self.variables.items():
                data[var] = {
                    'data': obj[:], 'dims': obj.dimensions[:],
                    'meta': self.get_meta(var=var).copy()}
            for reader in readers[:-1]:
                indices = np.searchsorted(self.dttime, reader.dttime,
                                          sorter=np.argsort(self.dttime))
                for var, obj in self.variables.items():
                    if stime not in obj.dimensions:
                        continue
                    data[var]['data'] = np.insert(
                        data[var]['data'], indices, obj[:],
                        axis=list(obj.dimensions).index(reader.timenames))
        elif not checks['levels']:
            slevel = self.levelnames
            for var, obj in self.variables.items():
                data[var] = {
                    'data': obj[:], 'dims': obj.dimensions[:],
                    'meta': self.get_meta(var=var).copy()}
            for reader in readers[:-1]:
                indices = np.searchsorted(self.level[:], reader.level[:])
                # if levels are reversed --> reverse indices
                if self.level[-1] < self.level[0]:
                    indices = np.searchsorted(self.level[:], reader.level[:],
                                              side='right',
                                              sorter=np.argsort(self.level[:]))
                    indices = len(self.level) - indices
                else:
                    indices = np.searchsorted(self.level[:], reader.level[:])
                for var, obj in self.variables.items():
                    if slevel not in obj.dimensions:
                        continue
                    data[var]['data'] = np.insert(
                        data[var]['data'], indices, obj[:],
                        axis=list(obj.dimensions).index(reader.levelnames))
        return ArrayReader(
            meta=self.get_meta().copy(),
            timenames=self._timenames, levelnames=self._levelnames,
            lonnames=self._lonnames, latnames=self._latnames, **data)

    def dump_nc(self, output, clobber=False, compression={}, close=True,
                missval=None, **kwargs):
        """Method to create netCDF file out the data in the ArrayReader

        Input:
          - output: String. Name of the resulting NetCDF file
          - clobber: Enable clobber (will significantly reduce file size).
              Input must be 'auto' or a list of the chunking parameters (the
              first one corresponds to time, the others to the dimension as
              stored in the netCDF file (usually the second corresponds to
              lat, the third to lon).
              If 'auto' chunking parameters are deterimined such that 1D and
              2D access are balanced. The calculation function is taken from
              http://www.unidata.ucar.edu/staff/russ/public/chunk_shape_3D.py
          - Dictionary with compression parameters for netCDF4 variable
              (determined by netCDF4 package. Possible keywords are zlib,
              complevel, shuffle and least_significant_digit. For documentation
              see
http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4.Variable-class.html
              If compression is not a dictionary, the value will be used for
              the complevel keyword in netCDF4 variables.
          - close: True/False. If True, the NetCDF handler will be closed at
              the end
          - missval: Missing Value. If None, it will be looked for a
              _FillValue attribute in the reader or the FillValue of the
              masked numpy array will be used.
        Returns:
          - nco. netCDF4.Dataset file handler of output
        """
        # docstring is extended below
        # set chunking parameter
        if os.path.exists(output):
            os.remove(output)
        self.logger.debug("Creating NetCDF file %s with..." % output)
        self.logger.debug("    clobber: %s", clobber)
        for item in kwargs.items():
            self.logger.debug("    %s: %s", *item)
        if clobber is not False:
            if clobber == 'auto':
                clobber = chunk_shape_3D(
                    [self.ntimes] + list(
                        self.lola_variables.values()[0].shape))
            nco = NCReader(output, 'w', clobber=True, **kwargs)
        else:
            nco = NCReader(output, 'w', **kwargs)
        if not isinstance(compression, dict):
            compression = {'zlib': True, 'complevel': compression}
        nco.setncatts(self.get_meta())
        created_dims = set()
        for var, obj in self.variables.items():
            self.logger.debug("Creating variable %s" % var)
            if missval is None:
                try:
                    fill_value = obj._FillValue
                except AttributeError:
                    try:
                        fill_value = obj[:].fill_value
                    except AttributeError:
                        fill_value = None
            else:
                fill_value = missval
            for i, dim in enumerate(obj.dimensions):
                if dim not in created_dims:
                    if dim == self.timenames:
                        nco.createDimension(dim, None)
                    else:
                        nco.createDimension(dim, obj.shape[i])
                    created_dims.add(dim)
            if clobber is not False:
                varno = nco.createVariable(
                    var, obj[:].dtype, obj.dimensions,
                    chunksizes=clobber, fill_value=fill_value, **compression
                    )
            else:
                varno = nco.createVariable(
                    var, obj[:].dtype, obj.dimensions,
                    fill_value=fill_value, **compression
                    )
            varno.setncatts(self.get_meta(var=var))
            varno[:] = obj[:]
        if close:
            nco.close()
        return nco

    def get_data(self, var=None, vlst=None, datashape='2D', convert_time=True,
                 rename_dims=True, convert_spatial=True, **dims):
        """Extract data out of the ArrayReader instance
        Please note that either var or vlst must be None

        Input:
          - var: string. Variable name to extract
          - vlst: List of strings. Variable names to extract. If this is not
              None, the fist dimension of the Data output will be set up as
              vlst
          - datashape: string ('1d', '2d', '3d', '4d' or 'any'). Data shape
              which shall be returned.
              -- If 1d, output will be a one-dimensional array. Different from
                  '2d', '3d' and '4d', you must give all dimensions
                  explicitly, there are no default values
              -- If 2d, output will be a two-dimensional array
                  with latitude and longitude dimension (if time and level is
                  not given, the according slices are 0)
              -- If 3d, output will be a three-dimensional array with time,
                  latitude and longitude dimension. (if level dimensions is
                  not given, the according slice will 0)
              -- If 4d: output will be a four-dimensional array with time,
                  level, latitude and longitude dimension.
              -- If any: dimensions will be unchanged and the full
                  slice is returned for dimensions not specified in **dims
              In the case of 2d, 3d or 4d, an error is raised if a dimension
              is found which is not in self.timenames, self.levelnames,
              self.lonnames or self.latnames and not specified by **dims
          - convert_time: Boolean (Default: True). If this is true, the time
              informations are converted to datetime.datetime instances
          - convert_spatial: Boolean (Default: True). If this is True, and the
              spatial dimensions (latitudes, longitudes) are in radians, they
              are converted to degrees
          Further Keyword arguments (**dims) may be of
          <dimension name>=<dimension slice>, where <dimension name> is the
          name of the dimension as stored in the variable instance and
          <dimension slice> the slice (or integer) which shall be extracted.
          If the dimension furthermore is a time dimensions, the index can
          in a numpy.datetime64 compatible style (e.g. '2005-03', see
          get_time_slice method)
        """
        def extract_data(var, dimslices):
            self.logger.debug("Extract %s", var)
            # read data
            self.logger.debug("Extract data with slice %s", dimslices)
            data = self.variables[var].__getitem__(tuple(dimslices))
            if not isinstance(data, np.ma.MaskedArray):
                self.logger.debug("Convert %s to masked array", type(data))
                data = np.ma.masked_array(
                    data, mask=np.ma.make_mask(np.zeros(shape=np.shape(data)),
                                            shrink=False), copy=True)

            # set up grid information for DataField instance
            vardims = list(self.variables[var].dimensions)

            # check whether data has the right shape
            if udim:  # unstructered has only one spatial dimension
                shapelens = {'1d': 1, '2d': 1, '3d': 2, '4d': 3}
            else:
                shapelens = {'1d': 1, '2d': 2, '3d': 3, '4d': 4}
            for shapelen, val in shapelens.items():
                if datashape == shapelen and len(data.shape) != val:
                    raise ValueError((
                        "Wrong dimension length! Expected %i dimensions but "
                        "found %i in %s. Set datashape to 'any' to return "
                        "all.") % (val, len(data.shape), vardims))
            # remove integer slices
            for key, val in dims.items():
                if isinstance(val, int) and key in vardims:
                    self.logger.debug(
                        "Remove dimension %s from dimension list because "
                        "integer slice.", key)
                    vardims.remove(key)
            return data, vardims

        self.logger.debug("Getting data with var %s and vlst %s", var, vlst)
        # set up dimension order to read from the netCDF
        if var is not None and var not in self.variables.keys():
            raise KeyError(
                'Unknown variable %s! Possible variables are %s' % (
                    var, self.variables.keys()))
        if vlst is not None and any(
                var not in self.variables.keys() for var in vlst):
            missing_vars = ', '.join(
                str(var) for var in vlst if var not in self.variables.keys())
            raise KeyError(
                'Unknown variables %s! Possible variables are %s' % (
                    missing_vars, self.variables.keys()))
        if var is not None and vlst is not None:
            raise ValueError(
                "Either var or vlst keyword must be None!")
        if var is None and vlst is None:
            raise ValueError("Either var or vlst must not be None!")
        single_var = True if var is not None else False
        multiple_vars = True if np.all(vlst is not None) else False

        self.logger.debug("Desired datashape: %s", datashape)
        for key, val in dims.items():
            self.logger.debug("Slice for dimension %s: %s", key, val)
        if var is not None:
            dimslices = list(self.variables[var].dimensions)
        elif np.all(vlst is not None):
            var = vlst[0]
            dimslices = list(self.variables[var].dimensions)
        else:
            dimslices = [self.timenames, self.levelnames, self.lonnames,
                         self.latnames]
        self.logger.debug("Dimensions in variable instance: %s",
                          ', '.join(dimslices))
        default_slices = {}
        datashape_slices = {
            '2d': {'time': 0, 'level': 0},
            '3d': {'time': slice(None), 'level': 0}}
        for dshape in ['1d', '4d', 'any']:
            datashape_slices[dshape] = {
                'time': slice(None), 'level': slice(None)}
        for val in datashape_slices.values():
            val['lon'] = slice(None)
            val['lat'] = slice(None)

        datashape = datashape.lower()
        if not datashape in datashape_slices:
            raise ValueError(
                "Wrong datashape %s! Possible values are %s" % (
                    ', '.join(datashape_slices)))

        varo = self.variables[var]
        udim = self._udim(varo)
        for i, dim in enumerate(dimslices):
            try:
                self.logger.debug(
                    "Try to get slice for dimension %s from user settings",
                    dim)
                if dim == self.timenames:
                    dimslices[i] = self.get_time_slice(dims[dim])
                    dims[dim] = dimslices[i]
                else:
                    dimslices[i] = dims[dim]
            except KeyError:
                self.logger.debug("Failed.")
                # if dimension has length 1: take first entry

                for sdim in ['lon', 'lat', 'time', 'level']:
                    failed = True
                    possible_names = getattr(self, '_'+sdim+'names')
                    if dim in possible_names:
                        for d in possible_names:
                            try:
                                dimslice = dims[d]
                                dims[dim] = dims.pop(d)
                            except KeyError:
                                dimslice = datashape_slices[datashape][
                                    sdim]
                        self.logger.debug(
                            "Found dimension %s in standard names for %s "
                            "--> use slice %s", dim, sdim, dimslice)
                        if varo.shape[i] == 1 and datashape == '1d':
                            dimslices[i] = 0
                            dims[dim] = dimslices[i]
                        else:
                            dimslices[i] = dimslice
                            dims[dim] = dimslices[i]
                        failed = False
                        break
                if failed and varo.shape[i] == 1:
                    self.logger.debug(
                        "Dimension %s was not specified but has length 1 --> "
                        "use first step" % dim)
                    if datashape == 'any':
                        dimslices[i] = slice(None)
                        dims[dim] = slice(None)
                    else:
                        dimslices[i] = 0
                        dims[dim] = 0
                elif failed and dim == udim:
                    dimslices[i] = slice(None)
                    dims[dim] = slice(None)
                elif failed and datashape in ['1d', 'any']:
                    if dim != udim:
                        warn("Dimension %s was not specified, therefore I "
                                "return all of that dimension" % dimslices[i])
                    dimslices[i] = slice(None)
                    dims[dim] = slice(None)
                elif failed:
                    self.logger.info(
                        "Use the first step for dimension %s. ", dim)
                    dimslices[i] = 0
                    dims[dim] = 0
        unused_dimensions = [dim for dim in dims if not dim in varo.dimensions]
        if unused_dimensions:
            if set(unused_dimensions) - {'time', 'level'}:
                warn("Did not use slice for dimension %s because not in "
                    "dimension list of variable %s!" % (
                    ', '.join(unused_dimensions), varo.dimensions))
            else:
                self.logger.debug(
                    "Did not use slice for dimension %s because not in "
                    "dimension list of variable %s!",
                    ', '.join(unused_dimensions), varo.dimensions)
            #for dim in unused_dimensions:
                #del dims[dim]

        datakwargs = {}
        standard_names = {'lon': self._lonnames, 'lat': self._latnames,
                          'time': self._timenames, 'level': self._levelnames}
        for dim, dimslice in dims.items():
            self.logger.debug("Try to get data for dimension %s", dim)
            try:
                dim_data = self.variables[dim]
            except KeyError:
                exist = False
                for key, val in standard_names.items():
                    if dim in val:
                        try:
                            dim_data = self.variables[
                                getattr(self, key+'names')]
                            exist = True
                            break
                        except KeyError:
                            pass
                if not exist and dim != udim:
                    warn("Did not find data for dimension %s in the reader" % (
                        dim))
                if not exist:
                    continue
            if dim in self._timenames and convert_time:
                try:
                    dim_data = self.convert_time(dim_data)
                except ValueError as e:
                    warn(e.message, logger=self.logger)
            elif (dim in self._lonnames.union(self._latnames)
                  and convert_spatial):
                dim_data = self.convert_spatial(dim_data, raise_error=False)
            datakwargs[dim] = dim_data[dimslice]

        # consider unstructured data
        if udim:
            lon, lat, triangles = self.get_triangles(varo, convert_spatial)
            if self._lonnames.isdisjoint(datakwargs):
                datakwargs[self.lonnames or 'lon'] = lon
            if self._latnames.isdisjoint(datakwargs):
                datakwargs[self.latnames or 'lat'] = lat
            datakwargs['triangles'] = triangles

            # add unstructured dimension to datakwargs
            if self.udims.isdisjoint(datakwargs):
                iax = list(varo.dimensions).index(udim)
                datakwargs[udim] = np.array(range(varo.shape[iax]))

        if single_var:
            varname = var
            data, vardims = extract_data(var, dimslices)
            dims['var'] = var
        elif multiple_vars:
            varname = '-'.join(vlst)
            data0, vardims = extract_data(vlst[0], dimslices)
            data = np.ma.zeros([len(vlst)] + list(np.shape(data0)))
            data[0, :] = data0
            del data0
            for i, var in enumerate(vlst[1:]):
                data0, vardims0 = extract_data(var, dimslices)
                if not np.all(vardims == vardims0):
                    raise ValueError(
                        "Dimensions do not match! Found dimensions %s for "
                        "variable %s and dimensions %s for variable %s." % (
                            vardims, vlst[0], vardims0, var))
                data[i+1, :] = data0
            vardims.insert(0, varname)
            # avoid a warning in from the DataField.check_dims method
            datakwargs[varname] = range(len(vlst))
            dims['vlst'] = vlst

        # get spatial axis
        if udim:
            spatial_ax = [list(vardims).index(udim)]
        else:
            spatial_ax = [i for i, dim in enumerate(vardims)
                          if dim in self._lonnames.union(self._latnames)]

        # rename dimensions lon, lat, time and level
        if rename_dims:
            for i, dim in enumerate(vardims):
                for key, val in standard_names.items():
                    if dim in val:
                        vardims[i] = key
            for dim in datakwargs:
                for key, val in standard_names.items():
                    if dim in val.intersection(datakwargs):
                        datakwargs[key] = datakwargs.pop(dim)
            kwargs = {}
        else:
            non_standard_names = {'lon': self.lonnames, 'lat': self.latnames,
                              'time': self.timenames,
                              'level': self.levelnames}
            for key in datakwargs:
                if key in non_standard_names:
                    datakwargs[non_standard_names[key]] = datakwargs.pop(key)
            kwargs = {'lon': self.lonnames, 'lat': self.latnames,
                      'time': self.timenames, 'level': self.levelnames}

        if datashape in ['any', '1d']:
            return DataField(data, var=varname, dimensions=vardims,
                             dims=datakwargs, spatial_ax=spatial_ax,
                             triangles=datakwargs.pop('triangles', None),
                             **kwargs)

        else:
            # set up dimensions according to the conventions used in nc2map
            spatial = [udim] if udim else ['lat', 'lon']
            if single_var:
                conventions = {'2d': spatial,
                               '3d': ['time'] + spatial,
                               '4d': ['time', 'level'] + spatial}
            elif multiple_vars:
                conventions = {'2d': [varname] + spatial,
                               '3d': [varname, 'time'] + spatial,
                               '4d': [varname, 'time', 'level'] + spatial}
            vardims = [dim for dim in vardims if dim in conventions[datashape]]
            for dim in vardims:
                if vardims.index(dim) != conventions[datashape].index(dim):
                    data = np.rollaxis(data, vardims.index(dim),
                                       conventions[datashape].index(dim))
                    vardims.remove(dim)
                    vardims.insert(conventions[datashape].index(dim), dim)
            return DataField(data, var=varname, dims=datakwargs,
                             dimensions=conventions[datashape],
                             spatial_ax=spatial_ax,
                             triangles=datakwargs.pop('triangles', None),
                             **kwargs)

    def gen_data(self, var, datashape='2d', **dims):
        """This method returns a data generator for a 2-dimensional data slice
        for the given dimensions dims.

        Any dimension being an 1-dimensional iterable object (list,
        numpy.array) specifies over which dimension it will be looped.

        In this case a generator for dummy 2-dimensional arrays is returned"""
        iterable_dims = []
        for key, val in dims.items():
            try:
                iter(val)
                iterable_dims.append(key)
            except TypeError:
                pass
        if len(iterable_dims) == 0:
            warn('Did not find any interable dimension!')
        elif len(iterable_dims) > 1:
            raise ValueError('Found multiple iterable dimensions!')
        iterable_dim = iterable_dims[0]
        it_vals = iter(dims[iterable_dim])
        next_val = True
        while next_val:
            try:
                dims.update({iterable_dim: next(it_vals)})
                yield self.get_data(var=var, datashape=datashape, **dims)
            except:
                next_val = False

    def set_meta(self, var=None, **meta):
        """Set meta information.
        Input:
          - var: string. Variable name. If None, the meta information is
              regarded as global meta information
          Keyword arguments (meta) describe the key, value pairs for the
          meta informations"""
        if var is not None and var not in self.variables.keys():
            raise KeyError('Unknown variable %s' % var)
        if var is None:
            obj = self
        else:
            obj = self.variables[var]
        obj.meta.update(meta)

    def get_meta(self, var=None):
        """Get meta information.
        Input:
          - var: string. Variable name. If None, the meta information is
              regarded as global meta information"""
        possible_keys = self.variables.keys() + list(self._timenames) + list(
            self._levelnames) + list(self._lonnames) + list(self._latnames)
        if var is not None and var not in possible_keys:
            raise KeyError('Unknown variable %s' % var)
        if var is None:
            obj = self
        elif var in self._lonnames:
            obj = self.lon
        elif var in self._latnames:
            obj = self.lat
        elif var in self._timenames:
            obj = self.time
        elif var in self._levelnames:
            obj = self.level
        else:
            obj = self.variables[var]
        return obj.meta

    def copy(self):
        """Returns an ArrayReader instance with the same attributes as this
        ArrayReader instance"""
        data = {}
        for var, obj in self.variables.items():
            data[var] = {'data': obj[:].copy(), 'dims': obj.dimensions[:],
                         'meta': self.get_meta(var=var).copy()}
        reader =  ArrayReader(
            meta=self.get_meta().copy(),
            timenames=self._timenames, levelnames=self._levelnames,
            lonnames=self._lonnames, latnames=self._latnames, **data)
        try:
            reader._grid_file = self._grid_file
        except AttributeError:
            pass
        return reader

    def close(self):
        """Closes the ArrayReader instance and deletes the stored variables"""
        for variable in self.variables:
            try:
                del variable.data
            except AttributeError:
                pass
            del self.variables[variable]

    def selname(self, *args, **kwargs):
        """Method to return an ArrayReader instance with only the grid
        variables

        Keyword arguments may be
          - copy: Boolean. (Default: False). If True, the data will sign to
              the same array as before. Otherwise everything will be copied."""
        copy = kwargs.get('copy')
        data = {}
        var_items = [
            item for item in self.variables.items() if item[0] in args]
        dimensions = set(chain(*[self.get_coords(obj)
                                 for var, obj in var_items]))
        dim_items = [
            item for item in self.variables.items() if item[0] in dimensions]
        for var, obj in var_items + dim_items:
            if copy:
                data[var] = {'data': obj[:].copy(), 'dims': obj.dimensions[:],
                             'meta': self.get_meta(var=var).copy()}
            else:
                data[var] = {'data': obj[:], 'dims': obj.dimensions[:],
                             'meta': self.get_meta(var=var)}
        if copy:
            meta = self.get_meta().copy()
        else:
            meta = self.get_meta()
        return ArrayReader(
            meta=meta,
            timenames=self._timenames, levelnames=self._levelnames,
            lonnames=self._lonnames, latnames=self._latnames, **data)

    def expand_dims(self, var=None, vlst=None, default=0, **dims):
        """Expands the dimensions to match the dimensions in the reader
        variable"""
        # dimensions in the variable (without longitude and latitude)
        if var is None:
            dims['vlst'] = vlst
            var = self.variables[vlst[0]]
        else:
            dims['var'] = var
            var = self.variables[var]
        vdims = set(var.dimensions) - self._latnames - self._lonnames \
            - self.udims
        ndims = len(set(dims) & self._levelnames & self._timenames &
                    set(var.dimensions))
        if ndims == len(vdims):
            return dims
        # standard names from reader._levelnames, etc.
        levelnames = self._levelnames
        timenames = self._timenames
        vdims -= set(dims)  # remove dims that match already
        if not levelnames.isdisjoint(dims):
            vdims -= levelnames
        if not timenames.isdisjoint(dims):
            vdims -= timenames
        for dim in vdims:
            if (dim == self.levelnames and 'level' in levelnames):
                dim = 'level'
            elif (dim == self.timenames and 'time' in timenames):
                dim = 'time'
            self.logger.info("Use %s for dimension %s. ", default, dim)
            dims[dim] = default
        return dims

    def extract(self, var=None, vlst=None, **kwargs):
        """Method to extract the data variable specified by **dims and returns
        an ArrayReader instance with only this data plus the grid
        informations. This method will return a new copy of the ArrayReader
        instance

        Keyword arguments (kwargs) are determined by the get_data method, where
        datashape is by default set to 'any' and convert_time to False.
        """
        kwargs.setdefault('datashape', 'any')
        kwargs.setdefault('convert_time', False)

        for key, val in kwargs.items():
            if isinstance(val, int) and key not in [
                'datashape', 'convert_time']:
                kwargs[key] = slice(val, val+1)
            elif key in self._timenames:
                val = self.get_time_slice(val)
                if isinstance(val, int):
                    kwargs[key] = slice(val, val+1)
                else:
                    kwargs[key] = val

        if var is not None:
            vlst = [var]
        full_vlst = set(vlst + list(chain(*(
            self.variables[var].dimensions for var in vlst)))).intersection(
                set(self.variables.keys()))
        reader = self.selname(*full_vlst, copy=True)
        data = reader.get_data(rename_dims=False, var=var,
                               vlst=vlst if var is None else None, **kwargs)
        for dim in data.dims.dtype.fields:
            try:
                reader.variables[dim].data = data.dims[dim].copy()
            except KeyError:
                pass
        if var is not None:
            reader.variables[var].data = data[:].copy()
            reader.variables[var].dimensions = data.dimensions[:]
        else:
            for i, var in enumerate(vlst):
                reader.variables[var].data = data[i, :].copy()
                reader.variables[var].dimensions = data.dimensions[1:]
        return reader

    def _arithmetics(self, value, func):
        """Basic function performing arithmetics with readers. Value may be a
        another ArrayReader instance, func is the function that defines the
        arithmetics method (e.g. lambda x, y: x + y)
        This method is called by __iadd__, __imulc__, etc."""

        def check_reader(reader):
            """Checks whether the reader dimensions and variables match to this
            one and prints warnings and raises errors
            Input:
            - reader: ArrayReader instance
            Output:
            - dictionary with matches"""
            checks = self._check_variables(reader)
            checks.update(checks['base']._check_dims(checks['new']))
            return checks

        try:  # try first simply to add the value or array to each variable
            for var in set(self.variables) - set(self.grid_variables):
                obj = self.variables[var]
                try:  # try __getitem__ (in case of array)
                    func(obj, slice(None), value[:])
                except TypeError:  # try float
                    func(obj, slice(None), value)
            base = self
        except TypeError:  # now assume a reader instance
            checks = check_reader(value)
            if not checks['base']:
                return self
            dims = ['level', 'time']
            base = checks['base']
            new = checks['new']
            if all(checks[dim] for dim in dims):
                for base_var, new_var in izip(checks['base_vars'],
                                              checks['new_vars']):
                    func(base.variables[base_var], slice(None),
                         new.variables[new_var][:])
            else:
                for base_var, new_var in izip(checks['base_vars'],
                                              checks['new_vars']):
                    dimslices = ['base_', 'new_']
                    dims_gen = product(
                        izip(checks['base_time'], checks['new_time']),
                        izip(checks['base_level'], checks['new_level']))
                    print base_var, new_var
                    for times, levels in dims_gen:
                        for i, var in enumerate([base.variables[base_var],
                                                 new.variables[new_var]]):
                            obj = base if not i else new
                            vardims = np.array(var.dimensions)
                            dimslices[i] = list(var.dimensions)
                            for j, dim in enumerate(dimslices[i]):
                                if dim == obj.timenames:
                                    dimslices[i][j] = times[i]
                                elif dim == obj.levelnames:
                                    dimslices[i][j] = levels[i]
                                else:
                                    dimslices[i][j] = slice(None)
                        func(
                            base.variables[base_var], tuple(dimslices[0]),
                            new.variables[new_var].__getitem__(tuple(
                                dimslices[1])))
        return base

    def _check_dims(self, reader):
        """Checks whether the reader dimensions match to this one and
        prints warnings and raises errors
        Method is used by _arithmetics to determine how to perform
        arithmetics between readers
        Input:
        - reader: ArrayReader instance
        Output:
        - dictionary with matches"""
        # ---- check grid sizes ----
        if (np.all(self.lon[:] != reader.lon[:])
                or np.all(self.lat[:] != reader.lat[:])):
            # raise warning if lens match anyway
            if all(len(getattr(reader, dim)) == len(getattr(self, dim))
                    for dim in ['lat', 'lon']):
                critical(
                    "Attention! Only grid size of %s matches!" % type(
                        reader))
            else:
                raise ValueError(
                    "Grid of %s does not match!" % type(reader))

        # ---- check dimensions ----
        checks = {}
        for dim in ['level', 'time']:
            my_dim = getattr(self, dim)
            rd_dim = getattr(reader, dim)
            if my_dim is None or rd_dim is None:
                self.logger.debug("%s in self is None: %s", dim,
                                my_dim is None)
                self.logger.debug("%s in reader is None: %s", dim,
                                rd_dim is None)
                if my_dim is None and rd_dim is None:
                    self.logger.debug("    --> both None")
                    checks[dim] = True
                else:
                    checks[dim] = False
                    checks['base_'+dim] = cycle([None]) if my_dim is None \
                        else xrange(len(my_dim))
                    checks['new_'+dim] = cycle([None]) if rd_dim is None \
                        else xrange(len(rd_dim))
            elif len(my_dim) == len(rd_dim):
                self.logger.debug(
                    "Dimension size for %s is the same (%i)", dim,
                    len(my_dim))
                if not np.all(my_dim[:] == rd_dim[:]):
                    warn("%s informations are not the same!" % dim)
                checks[dim] = True
                checks['base_'+dim] = [slice(None)]
                checks['new_'+dim] = [slice(None)]
            elif len(my_dim) == 1 or len(rd_dim) == 1:
                self.logger.debug(
                    "%s size in self: %i", dim, len(my_dim))
                self.logger.debug(
                    "%s size in reader: %i", dim, len(rd_dim))
                checks[dim] = False
                if len(my_dim) == 1:
                    checks['base_'+dim] = cycle([0])
                    checks['new_'+dim] = xrange(len(rd_dim))
                else:
                    checks['base_'+dim] = xrange(len(my_dim))
                    checks['new_'+dim] = cycle([0])
            else:
                raise ValueError(
                    "%s dimensions do not match!" % dim)
        return checks

    def _check_variables(self, reader):
        """Function to check if variables match
        Method is used by _arithmetics to determine how to perform
        arithmetics between readers
        """
        checks = {}
        # ---- check variables ----
        my_var_keys = sorted(set(self.variables) - set(self.grid_variables))
        rd_var_keys = sorted(
            set(reader.variables) - set(reader.grid_variables))
        my_nvars = len(my_var_keys)
        rd_nvars = len(rd_var_keys)
        # if no lola_variables in value: return
        if not my_nvars or not rd_nvars:
            if not rd_nvars:
                warn("Found no longitude-latitude variables in %s!" % (
                    type(reader)))
            if not my_nvars:
                warn("Found no longitude-latitude variables in self!")
            checks['base'] = False
        # if both have same lenghts --> calculate
        elif my_nvars == rd_nvars:
            self.logger.debug("Found same number of variables: %i",
                                my_nvars)
            # check if variable definitions match
            if (not (my_nvars == 1 and rd_nvars == 1)
                    and not np.all(my_var_keys == rd_var_keys)):
                raise ValueError(
                    "Variables of the first reader (%s) do not match "
                    "to those of the second (%s)!" % (
                        ', '.join(my_var_keys), ', '.join(rd_var_keys)))
            checks['base'] = self
            checks['new'] = reader
            checks['base_vars'] = my_var_keys
            checks['new_vars'] = rd_var_keys
        # if one has length 1 --> fill up stream
        elif my_nvars == 1 or rd_nvars == 1:
            self.logger.debug("Number of variables in %s: %i",
                                type(reader), rd_nvars)
            self.logger.debug("Number of variables in self: %i", my_nvars)
            if my_nvars == 1:
                checks['base'] = reader
                checks['new'] = self
                self.logger.debug("    --> Filling up self")
                checks['base_vars'] = rd_var_keys
                checks['new_vars'] = cycle(my_var_keys)
            else:
                checks['base'] = self
                checks['new'] = reader
                self.logger.debug("    --> Filling up %s" % type(reader))
                checks['base_vars'] = my_var_keys
                checks['new_vars'] = cycle(rd_var_keys)
        return checks

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __iadd__(self, value):
        """Self agglomeration method of ArrayReader class"""
        def _iadd(v, s, y):
            v[s] += y
            return v
        return self._arithmetics(value, _iadd)

    def __add__(self, value):
        """Agglomeration method of ArrayReader class"""
        reader = self.copy()
        reader += value
        return reader

    def __imul__(self, value):
        """Self multiplication method of ArrayReader class"""
        def _imul(v, s, y):
            v[s] *= y
            return v
        return self._arithmetics(value, _imul)

    def __mul__(self, value):
        """Multiplication method of ArrayReader class"""
        reader = self.copy()
        reader *= value
        return reader

    def __idiv__(self, value):
        """Self division method of ArrayReader class"""
        def _idiv(v, s, y):
            v[s] /= y
            return v
        return self._arithmetics(value, _idiv)

    def __div__(self, value):
        """Division method of ArrayReader class"""
        reader = self.copy()
        reader /= value
        return reader

    def __isub__(self, value):
        """Self subtraction method of ArrayReader class"""
        def _isub(v, s, y):
            v[s] -= y
            return v
        return self._arithmetics(value, _isub)

    def __sub__(self, value):
        """Subtraction method of ArrayReader class"""
        reader = self.copy()
        reader -= value
        return reader

    def __ipow__(self, value):
        """Self power method of ArrayReader class"""
        def _ipow(v, s, y):
            v[s] **= y
            return v
        return self._arithmetics(value, _ipow)

    def __pow__(self, value):
        """Subtraction method of ArrayReader class"""
        reader = self.copy()
        reader **= value
        return reader

    def __abs__(self):
        reader = self.copy()
        for varo in reader.variables.values():
            varo[:] = abs(varo[:])
        return reader

    def __str__(self):
        strings = [super(ReaderBase, self).__repr__()]
        for item in self.get_meta().items():
            strings.append("    %s: %s" % item)
        strings.append("    variables(dimensions): %s" % (
            ', '.join(
                "%s %s(%s)" % (item[1][:].dtype, item[0],
                               ', '.join(item[1].dimensions))
                for item in self.variables.items())))
        return '\n'.join(strings)


class ArrayReader(ReaderBase):
    """Enhanced ReaderBase with methods to rename and create Variables"""
    def renameAttribute(self, oldname, newname):
        """Renames the meta attribute 'oldname' to 'newname'"""
        try:
            self.meta[newname] = self.meta.pop(oldname)
        except KeyError:
            raise KeyError(
                "Variable %s does not exist in reader! Possible variables are "
                "%s." % (oldname, ', '.join(self.meta)))

    def renameVariable(self, oldname, newname):
        """Renames the variable 'oldname' to 'newname' (but not
        corresponding dimension! Use the renameDimension method for that.)"""
        try:
            self.variables[newname] = self.variables.pop(oldname)
            self.variables[newname].var = newname
        except KeyError:
            raise KeyError(
                "Variable %s does not exist in reader! Possible variables are "
                "%s." % (oldname, ', '.join(self.variables)))

    def renameDimension(self, oldname, newname):
        """Renames the dimension 'oldname' to 'newname' (but not
        corresponding variables! Use the renameVariable method for that.)"""
        exist = False
        for varo in self.variables.values():
            if oldname in varo.dimensions:
                exist = True
                dims = list(var.dimensions)
                dims.insert(dims.index(oldname), newname)
                varo.dimensions = tuple(dims)
        if not exist:
            dims = set(chain(*(
                varo.dimensions for varo in self.variables.values())))
            warn("Dimension %s not found in reader! Possible dimensions are "
                "%s" % (oldname, ', '.join(dims)))

    def createVariable(self, data=None, var='var', dims=('time', 'lat', 'lon'),
                       meta={}, delete=False):
        """Creates a new nc2map.readers.Variable in this ArrayReader

        Input:
            - data: numpy array with data
            - var: name of the variable
            - dims: tuple of dimension names (length of dims must match to
                length of data.ndim)
            - meta: dictionary containing meta informations (e.g. long_name,
                units, etc.)
            - delete: True/False. If False and the variable name var is already
                in use, a ValueError is raised.
        Returns:
            The created nc2map.readers.Variable instance
        """
        if delete and var in self.variables:
            raise ValueError("Variable %s already exists in the Reader!")
        self.variables[var] = Variable(data=data, var=var, dims=dims,
                                       meta=meta)
        return self.variables[var]

    def to_NCReader(self, *args, **kwargs):
        """Dumps the data in the ArrayReader instance into a NetCDF file,
        returns the open handler (if not close=False is set) and closes this
        ArrayReader instance.
        *args and **kwargs are determined by the dump_nc method.
        """
        kwargs.setdefault('close', False)
        nco = self.dump_nc(*args, **kwargs)
        self.close()
        return nco

    def __getattr__(self, attr):
        try:
            return self.meta[attr]
        except KeyError:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (
                    self.__class__.__name__, attr))

    def __dir__(self):
        return dir(super(ArrayReader, self)) + self.meta.keys()


class NCReader(ReaderBase):
    """nc2map compatible netCDF4.Dataset class

    The netCDF4.Dataset instance is stored in nco attribute.
    For initialization see __init__ method"""

    nco_base = nc.Dataset

    def __init__(self, *args, **kwargs):
        """Initialization method of NCReader instance

        Parameters
        ----------
        *args
            Determined by the netCDF4.Dataset class
        **kwargs
            Determined by the netCDF4.Dataset class (despite of the parameters
            below)

        Other Parameters
        ----------
        timenames: set of strings
            Dimension and variable names that shall be considered as time
            dimension or variable
        levelnames: set of strings
            Dimension and variable names that shall be considered as level
            dimension or variable
        lonnames: set of strings
            Dimension and variable names that shall be considered as longitude
            dimension or variable
        latnames: set of strings
            Dimension and variable names that shall be considered as latitude
            dimension or variable
        udims: set of strings
            Dimension names that indicates that the variable is defined on an
            unstructured grid
        ufuncs: list
            list containing interpretation functions for unstructered grids
            (see below). Default grid interpretation functions are for
            the ugrid conventions of triangular grids and for the ICON grid.

        See Also
        --------
        nc2map.readers.ReaderBase: Basic class for reader interpretation"""
        # docstring is extended below
        self.nco = None
        self.set_logger()
        self.logger.debug('Initialization arguments:')
        for arg in args:
            self.logger.debug('    %s', arg)
        self.logger.debug('Initialization keyword arguments:')
        for item in kwargs.items():
            self.logger.debug('    %s: %s', *item)

        # set timenames, levelnames, lonnames and latnames
        dimnames = {'timenames', 'levelnames', 'lonnames', 'latnames'}
        # convert from string to list
        for key in dimnames:
            if key in kwargs and isinstance(kwargs[key], str):
                kwargs[key] = [kwargs[key]]
        for key, val in defaultnames.items():
            setattr(self, key, set(kwargs.get(key, val)))
        self.ufuncs = kwargs.pop('ufuncs', ufuncs)

        # delete timenames, levelnames, lonnames and latnames key from kwargs
        kwargs = {key: val for key, val in kwargs.items()
                  if key not in dimnames}
        # save args and kwargs for initialization
        self._init_args = args[:]

        self._init_kwargs = kwargs.copy()

        # init netCDF.MFDataset
        self.nco = self.nco_base(*args, **kwargs)
        self._set_grid_file(*args, **kwargs)

    def _set_grid_file(self, *args, **kwargs):
        """Sets the _grid_file from kwargs['filename'], kwargs['files'][0],
        args[0] and self.filepath()"""
        try:
            try:
                self._grid_file = kwargs.pop('filename')
            except KeyError:
                self._grid_file = kwargs.pop('files')
        except KeyError:
            self._grid_file = args[0]
        except IndexError:
            # use netCDF4.Dataset.filepath method
            self._grid_file = self.filepath()
        except ValueError:
            warn("Could not get file name of grid file!")
            self.logger.debug(exc_info=True)
            return None
        try:
            self._grid_file = glob.glob(self._grid_file)[0]
        except TypeError:
            self._grid_file = glob.glob(self._grid_file[0])[0]
        except:
            warn("Could not get file name of grid file!")

    def to_ArrayReader(self):
        """Same as copy method but closes this instance as well"""
        newreader = self.copy()
        self.close()
        return newreader

    def set_meta(self, var=None, **meta):
        """Set meta information.
        Input:
          - var: string. Variable name. If None, the meta information is
              regarded as global meta information
          Keyword arguments (meta) describe the key, value pairs for the
          meta informations"""
        if var is not None and var not in self.variables.keys():
            raise KeyError('Unknown variable %s' % var)
        if var is None:
            self.nco.setncatts(meta)
        else:
            self.nco.variables[var].setncatts(meta)

    def get_meta(self, var=None):
        """Get meta information.
        Input:
          - var: string. Variable name. If None, the meta information is
              regarded as global meta information"""
        possible_keys = self.variables.keys() + list(self._timenames) + list(
            self._levelnames) + list(self._lonnames) + list(self._latnames)
        if var is not None and var not in possible_keys:
            raise KeyError('Unknown variable %s' % var)
        if var is None:
            obj = self
        elif var in self._lonnames:
            obj = self.lon
        elif var in self._latnames:
            obj = self.lat
        elif var in self._timenames:
            obj = self.time
        elif var in self._levelnames:
            obj = self.level
        else:
            obj = self.variables[var]
        return OrderedDict([(key, getattr(obj, key)) for key in obj.ncattrs()])

    def close(self):
        """Close the NCReader instance"""
        self.nco.close()
        self.nco = None

    def __dir__(self):
        return dir(super(NCReader, self)) + dir(self.nco)

    def __getattr__(self, attr):
        """Tries to get attribute defined by this class and if not available,
        return attribute from nco attribute"""
        if attr in self.__dict__.keys():
            return getattr(self, attr)
        elif hasattr(self.nco, attr):
            return getattr(self.nco, attr)
        else:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (
                    self.__class__.__name__, attr))


class MFNCReader(NCReader):
    """nc2map compatible netCDF4.MFDataset class

    Designed to manage a multifile dataset"""

    nco_base = nc.MFDataset

    def __init__(self, *args, **kwargs):
        """Initialization method of MFNCReader instance

        Keyword arguments (kwargs) may be
          - levelnames: set of strings: Gives the name of the level dimension
              for which will be searched in the netCDF file
          - timenames: set of strings: Gives the name of the time dimension
              for which will be searched in the netCDF file
          - lonnames: set of strings: Gives the name of the longitude dimension
              for which will be searched in the netCDF file
          - latnames: set of strings: Gives the name of the latitude dimension
              for which will be searched in the netCDF file
        Further args and kwargs are determined by the netCDF4.MFDataset
        instance:
        """
        super(MFNCReader, self).__init__(*args, **kwargs)

    def set_meta(self, var=None, **meta):
        """Set meta information.
        Input:
          - var: string. Variable name. If None, the meta information is
              regarded as global meta information
          Keyword arguments (meta) describe the key, value pairs for the
          meta informations"""
        raise ValueError(
            "nc.MFDataset does not support setting of meta data information!")


class FlexibleReader(MFNCReader):
    """Class to handle unstructered grids that vary with time

    This class is intended to manage 2D flexible mesh topologies, see
    :ref:`https://github.com/ugrid-conventions/ugrid-conventions/blob/v0.9.0/ugrid-conventions.md#2d-flexible-mesh-mixed-triangles-quadrilaterals-etc-topology`
    It is assumed that each file contains exactly one time step on a flexible
    mesh. The get_data method does not accept '3d' and '4d' data shapes

    ``*args`` and ``**kwargs`` are the same as for :class:`MFNCReader`

    Attributes
    ----------
    unlimited: string, name of the unlimited dimension
    """
    @property
    def unlimited(self):
        """Name of the unlimited dimension in the files"""
        return next(dim for dim, obj in self.nco.dimensions.items()
                    if obj.isunlimited())

    @property
    def unlimiteddim(self):
        """Variable corresponding to the unlimited dimension"""
        return self.__nco.variables[self.unlimited]

    @property
    def unlimitedlist(self):
        """Alternative list of names for the unlimited dimension in the
        files"""
        unlimited = self.unlimited
        for l in (self._levelnames, self._timenames, self._lonnames,
                  self._latnames):
            if unlimited in l:
                return l
        return {unlimited}

    def __init__(self, *args, **kwargs):
        # same docstring as for MFNCReader.__init__
        super(FlexibleReader, self).__init__(*args, **kwargs)
        # the nco attribute will be overwritten when get_data is called
        self.__nco = self.nco

    def get_data(self, *args, **kwargs):
        # same docstring as for MFNCReader.get_data
        unlimited = set(kwargs) & self.unlimitedlist
        if unlimited:
            unlimited = list(unlimited)[0]
            if unlimited in self._timenames:
                dim_slice = self.get_time_slice(kwargs[unlimited])
            else:
                dim_slice = kwargs[unlimited]
        else:
            unlimited = self.unlimited
            dim_slice = 0
        if isinstance(dim_slice, slice):
            dim_slice = range(*dim_slice.indices(len(self.unlimiteddim)))
        try:
            if len(dim_slice) > 1:
                raise ValueError(
                    "It is impossible to return more than one step of the "
                    "unlimited variable %s. Use the gen_data method "
                    "instead." % unlimited)
            self.nco = self.__nco._cdf[dim_slice[0]]
            kwargs[unlimited] = [0]
        except TypeError:
            self.nco = self.__nco._cdf[dim_slice]
            kwargs[unlimited] = 0
        try:
            return super(FlexibleReader, self).get_data(*args, **kwargs)
        except:
            raise
        finally:
            self.nco = self.__nco

    __init__.__doc__ = MFNCReader.__init__.__doc__
    get_data.__doc__ = MFNCReader.get_data.__doc__

# ------------ modify docstrings here ------------------
auto_set_reader.__doc__ += ', '.join(readers)
