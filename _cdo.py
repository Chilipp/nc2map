# -*- coding: utf-8 -*-
"""_cdo module of the nc2map module

This module contains the Cdo class, a subclass of the original cdo.Cdo class,
enhanced by some functions to better incorporate into the nc2map module"""
from functools import wraps
from cdo import Cdo as CdoBase
from readers import NCReader
from mapos import FieldPlot, LinePlot
from _maps import Maps

CDF_MOD_NCREADER= 'ncreader'


class Cdo(CdoBase):
    """Subclass of the original cdo.Cdo class in the cdo.py module

    Requirements are a working cdo binary and the installed cdo.py python
    module.

    For a documentation of an operator, use the python help function, for a
    list of operators, use the builtin dir function.
    Further documentation on the operators can be found here:
    https://code.zmaw.de/projects/cdo/wiki/Cdo%7Brbpy%7D
    and on the usage of the cdo.py module here:
    https://code.zmaw.de/projects/cdo

    For a demonstration script on how cdos are implemented, see the
    nc2map.demo.cdo_demo.py file

    Compared to the original cdo.Cdo class, the following things changed:
      - default cdf handler is the nc2map.readers.NCReader
      - implemented returnMap, returnMaps, returnData and
          returnLine keywords are implemented for all the operators.
          -- returnMaps takes None, a string or list of strings (the variable
              names) or a dictionary (keys and values are determined by the
              nc2map.Maps.__init__ method). None will open maps for all
              variables that have longitude and latitude dimensions in it.
              An open nc2map.Maps instance is returned.
          -- returnMap takes a string (the variable name) or a dictionary
              (keys and values are determined by the
              nc2map.mapos.FieldPlot.__init__ method). An open FieldPlot
              instance is returned
          -- returnLine takes a string or list of strings (the variable
              names) or a dictionary (keys and values are determined by the
              nc2map.mapos.LinePlot.__init__ method). An open LinePlot
              instance is returned
          -- returnData takes a string or list of strings (the variable
              names) or a dictionary (keys and values are determined by
              the nc2map.readers.ArrayReaderBase.get_data method).
              It returns a nc2map.readers.DataField instance of the
              specified variables, with datashape='any'."""
    def __init__(self, *args, **kwargs):
        """Initialization method of nc2map.Cdo class.
        args and kwargs are the same as for Base Class __init__ with the
        only exception that cdfMod is set to CDF_MOD_NCREADER by default"""
        kwargs.setdefault('cdfMod', CDF_MOD_NCREADER)
        super(Cdo, self).__init__(*args, **kwargs)
        self.loadCdf()

    def loadCdf(self, *args, **kwargs):
        """Load data handler as specified by self.cdfMod"""
        if self.cdfMod == CDF_MOD_NCREADER:
            self.cdf = NCReader
        else:
            super(Cdo, self).loadCdf(*args, **kwargs)

    def __getattr__(self, method_name):
        def my_get(get):
            """Wrapper for get method of Cdo class to include MapBase and
            Maps instance output"""
            @wraps(get)
            def wrapper(self, *args, **kwargs):
                if any(x in kwargs for x in ['returnMap', 'returnLine',
                                             'returnMaps']):
                    try:
                        map_ops = kwargs.pop('returnMap')
                        obj = FieldPlot
                        var_key = 'var'
                    except KeyError:
                        try:
                            map_ops = kwargs.pop('returnLine')
                            obj = LinePlot
                            var_key = 'vlst'
                        except KeyError:
                            map_ops = kwargs.pop('returnMaps')
                            obj = Maps
                            var_key = 'vlst'
                    kwargs['returnCdf'] = True
                    try:  # try mappable map_ops
                        return obj(get(*args, **kwargs), **map_ops)
                    except TypeError:  # assume variable name
                        if not map_ops:
                            return obj(get(*args, **kwargs))
                        else:
                            return obj(get(*args, **kwargs),
                                       **{var_key: map_ops})
                elif 'returnData' in kwargs:
                    var_ops = kwargs.pop('returnData')
                    kwargs['returnCdf'] = True
                    reader = get(*args, **kwargs)
                    try:
                        return reader.get_data(**var_ops)
                    except TypeError:
                        if isinstance(var_ops, (str, unicode)):
                            return reader.get_data(
                                var=var_ops, datashape='any')
                        else:
                            return reader.get_data(
                                vlst=var_ops, datashape='any')
                else:
                    return get(*args, **kwargs)
            return wrapper
        if method_name == 'cdf':
            # initialize cdf module implicitly
            self.loadCdf()
            return self.cdf
        else:
            get = my_get(super(Cdo, self).__getattr__(method_name))
            setattr(self.__class__, method_name, get)
            return get.__get__(self)

