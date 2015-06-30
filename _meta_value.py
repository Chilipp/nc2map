# -*- coding: utf-8 -*-
"""Module containing the MetaValue class

This script is part of the nc2map module
!!!Currently not in use because unsave!!!"""

class MetaValue(object):
    """Simple (but very crude) wrapper assigning user defined meta data to
    user defined data

    String Example
      Initialization
        >>> meta_str = MetaValue('my string', meta={'description': 'example'})
      Get the value
        >>> meta_str
        'my string'
      Get the meta data
        >>> meta_str.meta
        {'description': 'example'}
        >>> meta_str.description
        'example'
      Access attributes of the value
        >>> meta_str.split()
        ['my', 'string']
      Add something
        >>> new_meta_str = meta_str + ', okay?'
        >>> new_meta_str
        'my string, okay?'
        >>> new_meta_str.description
        'example'
    """
    def __init__(self, data, meta):
        self.__data = data
        self.__meta = meta

    def __repr__(self):
        return repr(self.__data)

    def __str__(self):
        return str(self.__data)

    def __get__(self):
        return self.__data

    def __getattr__(self, attr):
        if attr == '__meta':
            return self.__meta
        else:
            return getattr(self.__data, attr)

    def __call__(self, *args, **kwargs):
        return self.__data(*args, **kwargs)

    def __len__(self):
        return len(self.__data)

    def __delattr__(self, attr):
        """Deletes the attr from self.__data"""
        delattr(self.__data, attr)

    def __getitem__(self, item):
        return self.__data[item]

    def __setitem__(self, key, value):
        self.__data[item] = value

    def __delitem__(self, key):
        del self.__data[key]

    def __iter__(self):
        return iter(self.__data)

    def __reversed__(self):
        return reversed(self.__data)

    def __contains__(self, item):
        return item in self.__data

    def __lt__(self, value):
        return self.__data < value

    def __le__(self, value):
        return self.__data <= value

    def __eq__(self, value):
        return self.__data == value

    def __ne__(self, value):
        return self.__data != value

    def __ge__(self, value):
        return self.__data >= value

    def __gt__(self, value):
        return self.__data > value

    def __cmp__(self, value):
        return cmp(self.__data, value)

    def __hash__(self):
        return hash(self.__data)

    def __nonzero__(self):
        return bool(self.__data)

    def __unicode__(self):
        return unicode(self.__data)

    def __mod__(self, other):
        return self.__class__(self.__data % other, self.__meta)

    def __rmod__(self, other):
        return self.__class__(other % self.__data, self.__meta)

    def __divmod__(self, other):
        return self.__class__(divmod(self.__data, other), self.__meta)

    def __rdivmod__(self, other):
        return self.__class__(divmod(other, self.__data), self.__meta)

    # adding methods
    def __add__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data+value, self.__meta)

    def __radd__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value+self.__data, self.__meta)

    # mul methods
    def __mul__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data*value, self.__meta)

    def __rmul__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value*self.__data, self.__meta)

    # div methods
    def __div__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data/value, self.__meta)

    def __rdiv__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value/self.__data, self.__meta)

    def __floordiv__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data // value, self.__meta)

    def __rfloordiv__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value // self.__data, self.__meta)

    def __truediv__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data.__truediv__(value), self.__meta)

    def __rtruediv__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value.__truediv__(self.__data), self.__meta)

    def __lshift__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data << value, self.__meta)

    def __rlshift__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value << self.__data, self.__meta)

    def __rshift__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data >> value, self.__meta)

    def __rrshift__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value >> self.__data, self.__meta)

    def __and__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data & value, self.__meta)

    def __rand__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value & self.__data, self.__meta)

    def __or__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data | value, self.__meta)

    def __ror__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value | self.__data, self.__meta)

    def __xor__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data ^ value, self.__meta)
    def __rxor__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value ^ self.__data, self.__meta)

    def __rxor__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value ^ self.__data, self.__meta)

    def __sub__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(self.__data-value, self.__meta)

    def __rsub__(self, value):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(value-self.__data, self.__meta)

    def __pow__(self, value, *args, **kwargs):
        if hasattr(value, '_%s__data' % self.__class__.__name__):
            value = getattr(value, '_%s__data' % self.__class__.__name__)
        return self.__class__(pow(value, *args, **kwargs), self.__meta)

    def __neg__(self):
        return self.__class__(self.__data.__neg__(), self.__meta)

    def __pos__(self):
        return self.__class__(self.__data.__pos__(), self.__meta)

    def __abs__(self):
        return self.__class__(abs(self.__data), self.__meta)

    def __invert__(self):
        return self.__class__(self.__data.__invert__(), self.__meta)

    def __complex__(self):
        return self.__class__(complex(self.__data), self.__meta)

    def __int__(self):
        return self.__class__(int(self.__data), self.__meta)

    def __long__(self):
        return self.__class__(long(self.__data), self.__meta)

    def __float__(self):
        return self.__class__(float(self.__data), self.__meta)

    def __oct__(self):
        return  oct(self.__data)

    def __hex__(self):
        return hex(self.__data

    def __index__(self):
        return self.__data.__index__()

    def __coerce__(self, value):
        return self.__data.__coerce__(value)
