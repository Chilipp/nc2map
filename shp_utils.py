# coding: utf-8
"""Shapefile utils for the nc2map python module

This module contains the definition of the PolyWriter class which allows
the efficient extraction and dissolving of records and shapes in a polygon
shapefile (plus some other features)

Requirements are the python shapefile and shapely module"""
import os
import sys
import tempfile as tmp
import logging
from itertools import izip, chain, imap, compress, starmap, izip_longest
import shapefile as shp
from shapely.ops import cascaded_union
from shapely.geometry import MultiPolygon, Polygon
from numpy import nan, array, unique, isnan, mean
from .warning import warn

logger = logging.getLogger(__name__)


def get_bbox(ifile, *args, **kwargs):
    """Gets the sourrounding box of specified shapes

    Input:
      - ifile: path to a shape file or shapefile.Reader like object
      Arguments may be list-like objects, where the order matches to
      the fields in ifile.
      Keyword arguments may be any field identifier in ifile as key and
      list-like objects as values
    Output:
      The lower-left and upper-right corners of the surrounding box
      over all shapes [lon_min, lat_min, lon_max, lat_max]
    """
    logger.debug("Calculate box with...")
    logger.debug("    Arguments:")
    for arg in args:
        logger.debug("        %s", arg)
    logger.debug("    Keyword arguments:")
    for item in kwargs.items():
        logger.debug("        %s: %s", *item)
    sf = open_reader(ifile)
    if not args and not kwargs:
        logger.debug(
            "    No args and kwargs specified --> return full extent %s",
            sf.bbox)
        return sf.bbox
    logger.info("Calculate boundary box...")
    w = PolyWriter()
    boxes = array(
        [shape.bbox for shape in w.extract_shapes(sf, *args, **kwargs)])
    if not len(boxes):
        raise ValueError(
            "Could not find any matching records in the shapefile!")
    logger.debug("    Found %i matching shapes", len(boxes))
    bbox = [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(),
            boxes[:, 3].max()]
    logger.debug("    Return value: %s", bbox)
    return bbox


def get_fnames(ifile):
    """Return possible field names in a shape file

    Parameters
    ----------
    ifile: str or shapefile.Reader
        path to the shape file or a shapefile.Reader instance of the python
        shapefile module

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

    sf = open_reader(ifile)
    return [f[0] for f in sf.fields if not f[0].startswith("Deletion")]


def get_unique_vals(ifile, *args, **kwargs):
    """Get unique values in a shape file

    Parameters
    ----------
    ifile: str or shapefile.Reader
        path to the shape file or a shapefile.Reader instance of the python
        shapefile module
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
    sf = open_reader(ifile)
    fnames = get_fnames(sf)
    if not args:
        selectors = [1]*len(fnames)
    else:
        for key in [arg for arg in args if arg not in fnames]:
            raise KeyError(
                "%s key not found in field names! Possible keys are %s" % (
                    key, ", ".join(fnames)))
        indices = map(fnames.index, args)
        selectors = [1 if idx in indices else 0 for idx in range(len(fnames))]
    if not kwargs:
        records = sf.records()
    else:
        records = PolyWriter().extract_records(sf, **kwargs)
    return map(unique, compress(izip(*records), selectors))


def open_reader(ifile):
    logger.debug("Open reader...")
    try:
        logger.debug("Try shapefile.Reader")
        return Reader(ifile)
    except shp.ShapefileException:
        logger.debug("Failed. --> Assume shapefile.Reader", exc_info=True)
        return ifile


class Reader(shp.Reader):
    """Same as shapefile.Reader but with additional close method"""
    def close(self):
        """Closes all open files (i.e. self.dbf, self.shx and self.shp)"""
        for attr in ['shp', 'shx', 'dbf']:
            try:
                getattr(self, attr).close()
            except AttributeError:
                pass


class PolyWriter(shp.Writer):
    """Shapefile writer designed to create and modify polygon shapes"""

    def __init__(self, shapeType=5):
        # documentation is set below
        self.set_logger()
        try:
            super(PolyWriter, self).__init__(shapeType)
        except TypeError:  # shapefile.Writer is oldstyle class
            shp.Writer.__init__(self, shapeType)

    def extract_records(self, ifile, exact_matches=False, *args, **kwargs):
        """Returns an iterator with all shapes and shape records
        matching to *args and **kwargs.

        Input:
            - ifile: path to a shape file or shapefile.Reader like object
            - exact_matches: True/False. If False, only one of the named
                fields specified by *args and **kwargs must match to return
                the full record, otherwise all fields must match
        Arguments may be list-like objects, where the order matches to
        the fields in ifile.
        Keyword arguments may be any field identifier in ifile as key and
        list-like objects as values

        Returns:
          Iterator of records. The shape of each
          object can be accessed via the shape attribute, the record via the
          record attribute.
        """
        sf, test = self._extract(ifile, *args, **kwargs)
        return (rec for rec in sf.iterRecords() if test(rec))

    def _extract(self, ifile, exact_matches=False, *args, **kwargs):
        """Method called by extract_records and extract_shapeRecords which
        returns a shapefile and a testing function that can be used to test
        a record.
        - exact_matches: True/False. If False, only one of the named
                fields specified by *args and **kwargs must match to return
                the full record, otherwise all fields must match
        Arguments and keyword arguments are the same as for extract_records,
        extract_shapeRecords and extract_shapes
        Returns:
          - sf: open shapefile
          - test: Function that takes one record in the shapefile and
              evaluates to True if it matches args and kwargs.
        """
        if exact_matches:
            def test(rec):
                return all(
                    not len(vals) or r in vals
                    for r, vals in izip(rec, values))
        else:
            def test(rec):
                return any(
                    len(vals) and r in vals for r, vals in izip(rec, values))
        sf = open_reader(ifile)
        if not args and not kwargs:
            return sf, lambda rec: True
        fnames = get_fnames(sf)
        if len(args) > len(fnames):
            raise ValueError(
                "Number of arguments (%i) exceed number of fields in "
                "the shapefile (%i)!" % (len(args), len(fnames)))
        if any(isinstance(arg, (str, unicode)) for arg in args):
            warn("Found string arguments in args! Usually I expect lists!")
        for key in (key for key in kwargs if key not in fnames):
            raise KeyError(
                "%s key not found in field names! Possible keys are %s" % (
                    key, ", ".join(fnames)))
        if any(isinstance(val, (str, unicode)) for val in kwargs.itervalues()):
            warn("Found string arguments in kwargs! Usually I expect lists!")
        values = [list(arg) for arg in args] + [
            [] for _ in xrange(len(fnames) - len(args))]
        for key, val in kwargs.items():
            values[fnames.index(key)] += list(val)
        return sf, test


    def extract_shapeRecords(self, ifile, exact_matches=False, *args,
                             **kwargs):
        """Returns an iterator with all shapes and shape records
        matching to *args and **kwargs.

        Input:
            - ifile: path to a shape file or shapefile.Reader like object
            - exact_matches: True/False. If False, only one of the named
                fields specified by *args and **kwargs must match to return
                the full record, otherwise all fields must match
        Arguments may be list-like objects, where the order matches to
        the fields in ifile.
        Keyword arguments may be any field identifier in ifile as key and
        list-like objects as values

        Returns:
          Iterator of shapefile._ShapeRecord instances. The shape of each
          object can be accessed via the shape attribute, the record via the
          record attribute.
        """
        sf, test = self._extract(ifile, *args, **kwargs)
        return (shp._ShapeRecord(shape, rec)
                for shape, rec in izip(sf.iterShapes(), sf.iterRecords())
                if test(rec))

    def extract_shapes(self, ifile, exact_matches=False, *args, **kwargs):
        """Same as extract_shapeRecords method but returns an iterator of the
        shapes instead over shapefile._ShapeRecord instances"""
        return (sr.shape for sr in self.extract_shapeRecords(
            ifile, exact_matches, *args, **kwargs))

    def shift_to_0_360(self, ifile, append=False, eps=0.001, add_fields=True):
        """Shifts longitudes in ifile from [-180, 180] to [0, 360].

        Input:
        - ifile: path to the original shape file with longitudes from -180 to
            180 or shapefile.Reader like object
        - append: True/False. If True, old points with longitudes from -180 to
            0 are kept.
        - eps: very small number. This number is added (or subtracted) to the
            points at the right (left) boarder, since these polygons are merged
            together.
        - add_fields: True/False. If True, all fields from ifile are added to
            the PolyWriter instance fields
        """
        logger = self.logger
        logger.debug("Shifting from [-180, 180] to [0, 360]")
        logger.debug("    Input file: %s", ifile)
        logger.debug("    append:     %s", append)
        logger.debug("    eps:        %s", eps)
        logger.debug("    add_fields: %s", add_fields)

        sf = open_reader(ifile)
        # define fields in new shape file
        if add_fields:
            for field in sf.fields:
                self.field(*field)
        boarders = sf.bbox[::2]
        if append:
            test_old = lambda lon: True
        else:
            test_old = lambda lon: lon >= 0
        for shape, record in izip(sf.iterShapes(), sf.iterRecords()):
            parts = [[] for _ in shape.parts]
            new_parts = [[] for _ in shape.parts]
            parts_to_merge = []
            try:
                indices = enumerate(izip(shape.parts, chain(shape.parts[1:],
                                                            [None])))
            except IndexError:
                indices = enumerate(izip(shape.parts, [None]))
            for i, (j, k) in indices:
                old_part = shape.points[slice(j, k)]
                if any(point[0] in boarders for point in old_part):
                    parts_to_merge.append([])
                    parts_to_merge.append([])
                    part = parts_to_merge[-2]
                    new_part = parts_to_merge[-1]
                    append_right = lambda lon: lon if lon != boarders[1] \
                        else lon + eps
                    append_left = lambda lon: lon if lon != boarders[0] \
                        else lon - eps
                else:
                    part = parts[i]
                    new_part = new_parts[i]
                    append_right = lambda lon: lon
                    append_left = lambda lon: lon
                for point in shape.points[slice(j, k)]:
                    if test_old(point[0]):
                        part.append([append_right(point[0]), point[1]])
                    if point[0] <= 0:
                        new_part.append(
                            [360 + append_left(point[0]), point[1]])
            # filter empty lists out
            parts = filter(lambda a: a, parts)
            new_parts = filter(lambda a: a, new_parts)
            parts_to_merge = filter(lambda a: a, parts_to_merge)
            # merge to avoid boarder conflicts
            if parts_to_merge:
                new_polygons = cascaded_union([
                    Polygon(part) for part in parts_to_merge])
            else:
                new_polygons = []

            # create new polygon
            try:
                self.poly(parts + new_parts +
                    [poly.exterior.coords[:] for poly in new_polygons])
            except TypeError:
                self.poly(
                    parts + new_parts + [new_polygons.exterior.coords[:]])

            self.record(*record)
        return self

    def _shape_to_parts(self, shape):
        """Returns an interator over the parts in shape"""
        it = izip_longest(shape.parts, shape.parts[1:])
        return (shape.points[slice(i, j)] for i, j in it)


    def shapes_to_poly(self, shapes, polyo=Polygon, **kwargs):
        """Returns an iterator of polygons over the specified shapes or a
        Multipolygon if return_multi.
        Input:
          - shapes: iterable (e.g. list) of shapes in a shapefile.Reader
          - polyo: The polygon creater which takes a single part as argument.
              Defaults to the shapely.geometry.Polygon class, but may also
              be the matplotlib.patches.Polygon class
        Further keyword arguments (**kwargs) are passed to the polyo call.
        Returns:
          itertools.imap iterator containing the single polygons in shapes
        """
        return imap(lambda part: polyo(part, **kwargs),
                    self._shape_to_parts(shapes))

    def dissolve_shapes(self, shapes):
        """Dissolves the given shapes and returns the parts"""
        polygons = cascaded_union(
            list(chain(*map(self.shapes_to_poly, shapes))))
        try:
            return [poly.exterior.coords[:] for poly in polygons]
        except TypeError:
            return [polygons.exterior.coords[:]]

    def dissolve(self, ifile, fields=[], add_fields=True, **kwargs):
        """Dissolves the shapes in ifile

        Input:
          - ifile: path to a shape file or shapefile.Reader like object
          - fields: Fields by which to dissolve. If no fields are given,
              all fields are dissolved
          - add_fields: True/False. If True, all fields from ifile are added
              to the PolyWriter instance fields
          - delete_fields: List of field names that shall not be considered

        Further keyword arguments can be used for a specific selection (see
        extract_shapeRecords method)
        """
        from pandas import DataFrame
        self.logger.info("Dissolving shapes...")
        sf = open_reader(ifile)
        fnames = get_fnames(sf)
        for key in [fname for fname in fields if fname not in fnames]:
            raise KeyError(
                "%s key not found in field names! Possible keys are %s" % (
                    key, ", ".join(fnames)))

        if not fields:
            # use all fields
            c = lambda rec: rec
        else:
            # use only selected fields
            selectors = [1 if fname in fields else 0 for fname in fnames]
            c = lambda rec: compress(rec, selectors)

        if add_fields:
            for field in c(
                    [f for f in sf.fields if not f[0].startswith("Deletion")]):
                self.field(*field)

        if not kwargs:
            srs = starmap(shp._ShapeRecord,
                          izip(sf.iterShapes(), sf.iterRecords()))
        else:
            srs = self.extract_shapeRecords(sf, **kwargs)
        # if no fields are specified, simply dissolve all shapes
        if not fields:
            try:
                sr = next(srs)
            except StopIteration:
                raise ValueError("Found no shapes to dissolve")
            self.record(*c(sr.record))
            self.poly(self.dissolve_shapes(chain(
                [sr.shape], (sr.shape for sr in srs))))
            return
        # otherwise use the pandas groupby method
        df = DataFrame(
            [list(chain(c(sr.record),
                        [sr.shape])) for sr in srs],
            columns=list(c(fnames)) + ['_shape'])
        for rec, shapes in df.groupby(fields)['_shape'].unique().iteritems():
            self.poly(self.dissolve_shapes(shapes))
            if len(fields) == 1:
                self.record(rec)
            else:
                self.record(*rec)
        return

    def copy(self, ifile, add_fields=True, delete_fields=[], **kwargs):
        """Copies all shapes (or those selected) and records from ifile to
        this Writer

        Input:
          - ifile: path to the original shape file with longitudes from -180 to
              180 or shapefile.Reader like object
          - add_fields: True/False. If True, all fields from ifile are added to
              the PolyWriter instance fields
          - delete_fields: List of field names that shall not be considered

        Further keyword arguments can be used for a specific selection (see
        extract_shapeRecords method)"""
        sf = open_reader(ifile)
        fnames = get_fnames(sf)
        if not delete_fields:
            # use all fields
            c = lambda rec: rec
        else:
            # use only selected fields
            selectors = [1 if fname not in delete_fields else 0
                         for fname in fnames]
            c = lambda rec: compress(rec, selectors)
        if add_fields:
            for field in c(
                    [f for f in sf.fields if not f[0].startswith("Deletion")]):
                self.field(*field)
        if not kwargs:
            shapeRecords = starmap(shp._ShapeRecord,
                                   izip(sf.iterShapes(), sf.iterRecords()))
        else:
            shapeRecords = self.extract_shapeRecords(sf, **kwargs)
        for sr in shapeRecords:
            if len(sr.shape.parts) == 1:
                self.poly([sr.shape.points])
            else:
                self.poly(
                    [sr.shape.points[i:j] for i, j in zip(
                        sr.shape.parts[:-1], sr.shape.parts[1:])] +
                    [sr.shape.points[sr.shape.parts[-1]:]])
            self.record(*c(sr.record))


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

    __init__.__doc__ = shp.Writer.__init__.__doc__


class NamedTemporaryFiles(dict):
    """Create and return a dictionary with multiple temporary files.

    The arguments are the same as for the tempfile.NamedTemporaryFile
    function, only the suffix keyword changed.
    Arguments:
    'prefix', 'dir' -- as for tempfile.mkstemp.
    'suffix' -- list-like (unique and non-empty) suffixes for each file
    'mode' -- the mode argument to os.fdopen (default "w+b").
    'bufsize' -- the buffer size argument to os.fdopen (default -1).
    'delete' -- whether the file is deleted on close (default True).
    The files are created as tempfile.mkstemp() would do it.

    Returns an dictionary-like whose values have a file-like interfaces; the
    name of the files is accessible as file.name. The files will be
    automatically deleted when it is closed unless the 'delete' argument is
    set to False.
    For a suffix s in the 'suffix' argument, the file name can be accessed via
    the corresponding key s (i.e. access the file for suffix 'txt' via
    files['txt'] and the file name via files['txt'].name.
    All files share the same basename, only the suffix changed.

    """
    def __init__(self, mode='w+b', bufsize=-1, suffix=["shp", "dbf", "shx"],
                 prefix=tmp.template, dir=None, delete=True):
        from collections import Counter
        super(NamedTemporaryFiles, self).__init__()
        # first check whether no invalid suffix is part
        suffix = list(suffix)
        for s in (s for s in suffix if not s):
            raise ValueError("Empty suffixes are not allowed!")
        # now check whether one suffix occurs multiple times
        for item in (item for item in Counter(suffix).items() if item[1] > 1):
            raise ValueError("Found suffix %s multiple (%i) times!" % item)
        suffix = ['.'+s for s in suffix]
        files = [0] * len(suffix)

        success = False
        for _ in xrange(tmp.TMP_MAX):
            f = tmp.NamedTemporaryFile(
            mode=mode, bufsize=bufsize, suffix=suffix[0], prefix=prefix,
            dir=dir, delete=delete)
            basename = os.path.splitext(f.name)[0]
            success = all(not os.path.exists(basename+s) for s in suffix[1:])
            if success:
                break

        if not success:
            f.close()
            raise IOError("No usable temporary file name found")

        if 'b' in mode:
            flags = tmp._bin_openflags
        else:
            flags = tmp._text_openflags

        # Setting O_TEMPORARY in the flags causes the OS to delete
        # the file when it is closed.  This is only supported by Windows.
        if os.name == 'nt' and delete:
            flags |= os.O_TEMPORARY

        files[0] = f
        for i, s in enumerate(suffix[1:], start=1):
            fd = os.open(basename+s, flags, 0600)
            tmp._set_cloexec(fd)
            f = os.fdopen(fd, mode, bufsize)
            files[i] = tmp._TemporaryFileWrapper(f, basename+s, delete=delete)

        self.name = basename
        self.update(zip([s[1:] for s in suffix], files))


    def close(self):
        """Closes all files in self.values()"""
        for f in self.values():
            try:
                f.close()
            except AttributeError:
                pass

    def names(self):
        """Return the file names for all files in self.values()"""
        return [f.name for f in self.values()]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
