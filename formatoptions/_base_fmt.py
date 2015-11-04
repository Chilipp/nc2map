# -*- coding: utf-8 -*-
"""This file contains a class for basic formatoptions of labels
"""
import logging
import numpy as np
from collections import OrderedDict
from difflib import get_close_matches
from matplotlib.rcsetup import validate_bool
from _basefmtproperties import BaseFmtProperties
from ..defaults import BaseFormatter as default
from ..defaults import texts
from ..warning import warn

__author__ = "Philipp Sommer (philipp.sommer@studium.uni-hamburg.de)"
__version__ = '0.0'

_props = BaseFmtProperties()

class FormatoptionMeta(type):
    """Meta class for formatoptions to provide a uniform documentation"""
    def __new__(cls, clsname, bases, dct):
        """Assign an automatic documentation to the formatoption"""
        required = {'_summary', '_ext_summary', '_possible', '_default',
                    '_key'}
        for key in required.difference(dct):
            succeded = False
            for base in bases:
                try:
                    dct[key] = getattr(base, key)
                    succeded = True
                    break
                except AttributeError:
                    pass
            if succeded:
                continue
            else:
                raise ValueError(
                "A formatoption class needs %s to be defined!" % key)
        doc = "**%s**\n" % dct['_summary']
        key = dct['_key']
        try:
            possible = dct['_possible'] % {'default': dct['_default'][key]}
        except KeyError:
            possible = dct['_possible']
        if dct['_ext_summary'].startswith('\n'):
            doc += dct['_ext_summary']
        else:
            doc += "\n" + dct['_ext_summary']
        doc += "\n\n**Possible values**\n%s" % possible
        if dct.get('_warning'):
            if dct['_warning'].startswith('\n'):
                doc += "\n\nWarning\n-------%s" % dct['_warning']
            else:
                doc += "\n\nWarning\n-------\n%s" % dct['_warning']
        if dct.get('_examples'):
            if dct['_examples'].startswith('\n'):
                doc += "\n\nExamples\n--------%s" % dct['_examples']
            else:
                doc += "\n\nExamples\n--------\n%s" % dct['_examples']
        if dct.get('_note'):
            if dct['_note'].startswith('\n'):
                doc += "\n\nNotes\n-----%s" % dct['_note']
            else:
                doc += "\n\nNotes\n-----\n%s" % dct['_note']
        if dct.get('_see'):
            if dct['_see'].startswith('\n'):
                doc += "\n\nSee Also\n--------%s" % dct['_see']
            else:
                doc += "\n\nSee Also\n--------\n%s" % dct['_see']
        dct['__doc__'] = doc #+ dct.get('__doc__', '')
        return super(FormatoptionMeta, cls).__new__(cls, clsname, bases, dct)

class Formatoption(object):
    """Base class for formatoptions descriptor, defining dummy properties that
    should be covered by each formatoption class"""
    __metaclass__ = FormatoptionMeta  # defines documentation
    # necessary documention attributes
    _summary = "Formatoption Base class"  # Long name of fmt
    _key = "fmtkey"  # unique formatoption key
    _ext_summary = "Description." # What is this formatoption doing?
    _possible = """
    - str: Description for string
    - dict: Description for dict"""  # The possible values for the formatoption

    # optional documentation attributes
    _warning = "Warning"  # optional warning
    _examples = "Examples"  # optional examples
    _note = "Note"  # optional note
    _see = "other stuff"  # optional, see also other stuff

    # priority of the formatoption:
    #  10: at the end (labels, etc)
    #  20: before plotting (colormap, etc)
    #  30: before getting the data (basemap options)
    priority = 10

    _default = default  # dictionary with default values (may be overwritten)

    # dependencies. List of formatoptions that have to be updated before this
    # one is updated.
    dependencies = []

    # optional, has to be True if the formatoption has a make_plot method to
    # make the plot and a remove method to remove the plot.
    plot = False

    # group name of the formatoption keyword
    group = 'misc'

    # long name of the group
    @property
    def groupname(self):
        """Long name of the group this formatoption belongs too."""
        return groups[self.group]

    def __set__(self, instance, value):
        """Set method"""
        value = self.__class__.validate(value)
        setattr(instance, '_' + self._key, value)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, '_' + self._key)

    def __delete__(self, instance, owner):
        setattr(instance, '_' + self._key, default[self._key])

    @staticmethod
    def validate(value):
        """Function to validate the input"""
        return value  # no validation

    def initialize(self, ploto):
        """Method that is called during initialization of a MapBase instance

        Parameters
        ----------
        plot: :class:`nc2map.plotos.BasePlot` instance that is initialized"""
        pass

    def update(self, ploto):
        """Method that is call to update the formatoption on the axes

        Parameters
        ----------
        ploto: :class:`nc2map.plotos.BasePlot` instance to update"""
        pass

    def get_value(self, ploto):
        """Method to return the formatoption value corresponding to this fmt

        Parameters
        ----------
        :class:`nc2map.plotos.BasePlot` instance to update"""
        return getattr(ploto.fmt, self._key)

class Enable(Formatoption):
    """enable formatoption"""
    _summary = "Enable plotting"
    _key = "enable"
    _ext_summary = "Formatoption to enable or disable the plotting."
    _possible = """bool
    Default: %(default)s"""
    _examples = """
Disable the wind plots in your :class:`~nc2map.Maps` instance::

    mymaps.update(enable=False, windonly=True)"""

    validate = staticmethod(validate_bool)

class Tight(Formatoption):
    """tight formatoption"""
    _summary = "Automatic subplot adjusting"
    _key = "tight"
    _ext_summary = """
    Automatically adjust subplots. This formatoption uses plt.tight_layout to
    automatically adjust the plot boundaries"""
    _possible = """bool
     Default: %(default)s"""
    _warning = """
There is no update method to undo what happend after this formatoption is
set!"""

    validate = staticmethod(validate_bool)

_replace_note = """
        Metadata keys (var, time, level, or netCDF attributes like long_name,
        units, ...) maybe replaced via %%(key)s. If the time information in the
        reader (i.e. NetCDF file) is stored in relative (e.g. hours since ...)
        or absolute (day as %%Y%%m%%d.f) units, directives like %%Y for year
        or %%m for the month as given by the python datetime package, are also
        replaced by the specific time information.
        There are furthermore some special keys which are replaced when you
        insert '{key}' in your text (e.g. {tinfo}). Those are
            %s
        Those special keys are defined in the
        nc2map.defaults.texts['labels'] dictionary."""  % (
            '\n            '.join(
                map(lambda item: "%s: %s" % item, texts['labels'].items())))

class BaseFormatter(object):
    enable = Enable()
    tight = Tight()
    grid = _props.default(
        'grid', """
        Enables the plotting of the grid on the axes if not None (Default:
        %s).""" % default['grid'])

    # fontsizes and fontweights
    fontsize = _props.fontsize(
        'fontsize', """
        string or float (Default: %s). Defines the default size of ticks, axis
        labels and title. Strings might be 'xx-small', 'x-small', 'small',
        'medium', 'large', 'x-large', 'xx-large'. Floats define the absolute
        font size, e.g., 12""" % default['fontsize'])
    ticksize = _props.ticksize(
        'ticksize', """
        string or float (Default: %s). Defines the size of the ticks
        (see fontsize for possible values)""" % default['ticksize'])
    figtitlesize = _props.figtitlesize(
        'figtitlesize', """
        string or float (Default: %s). Defines the size of the subtitle of
        the figure (see fontsize for possible values). This is the title of
        this specific axes! For the title of the figure see figtitlesize""" %
        default['figtitlesize'])
    titlesize = _props.titlesize(
        'titlesize', """
        string or float (Default: %s). Defines the size of the
        title (see fontsize for possible values)""" % default['titlesize'])
    labelsize = _props.labelsize(
        'labelsize', """
        string or float (Default: %s). Defines the size of x- and y-axis labels
        (see fontsize for possible values)""" % default['labelsize'])
    fontweight = _props.fontweight(
        'fontweight', """
        A numeric value in the range 0-1000 or string (Default: %s).
        Defines the fontweight of the ticks. Possible strings are one of
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold',
        'black'.""" % default['fontweight'])
    tickweight = _props.tickweight(
        'tickweight', """
        Fontweight of ticks (Default: Defined by fontweight property).
        See fontweight above for possible values.""")
    figtitleweight = _props.figtitleweight(
        'figtitleweight', """
        Fontweight of the figure suptitle (Default: Defined by fontweight
        property). See fontweight above for possible values.""")
    titleweight = _props.titleweight(
        'titleweight', """
        Fontweight of the title (Default: Defined by fontweight property).
        See fontweight above for possible values. This is the title of this
        specific axes! For the title of the figure see figtitleweight""")
    labelweight = _props.labelweight(
        'labelweight', """
        Fontweight of axis labels (Default: Defined by fontweight property).
        See fontweight above for possible values.""")

    # axis colors
    axiscolor = _props.axiscolor(
        'axiscolor', """
        string or color for axis or dictionary (Default: %s). If string
        or color this will set the default value for all axis. If
        dictionary, keys must be in ['right', 'left', 'top', 'bottom'] and the
        values must be a string or color to set the color for 'right', 'left',
        'top' or 'bottom' specificly.""" % default['axiscolor'])

    # labels
    figtitle = _props.default(
        'figtitle', """
        string (Default: %s). Defines the figure suptitle of the
        plot.""" %  default['figtitle'])
    title = _props.default(
        'title', """
        string (Default: %s). Defines the title of the plot.%s
        This is the title of this specific axes! For the title of the figure
        see figtitle""" % (default['title'], _replace_note))

    text = _props.text(
        'text', """
        String, tuple or list of tuples (x,y,s[,coord.-system][,options]])
        (Default: %s).
          - If string s: this will be used as (1., 1., s, {'ha': 'right'})
              (i.e. a string in the upper right corner of the axes).
          - If tuple or list of tuples, each tuple defines a text instance on
              the plot. 0<=x, y<=1 are the coordinates. The coord.-system can
              be either the data coordinates (default, 'data') or the axes
              coordinates ('axes') or the figure coordinates ('fig'). The
              string s finally is the text. options may be a dictionary
              to specify format the appearence (e.g. 'color', 'fontweight',
              'fontsize', etc., see matplotlib.text.Text for possible keys).
        To remove one single text from the plot, set (x,y,'') for the text at
        position (x,y); to remove all set text=[].""" % (
            default['text']) + _replace_note)

    def __init__(self, **kwargs):
        self.set_logger()
        # dictionary for default values
        self._default = default.copy()
        # list of label parameters properties (tick, label, etc., etc.)
        self._text_props = []
        self._tickops = {}  # settings for tick labels
        self._labelops = {}  # settings for axis labels
        self._titleops = {}  # settings for title
        self._figtitleops = {}  # settings for figure suptitle
        self._updating = 0
        # set default values
        for key, val in self._default.items():
            setattr(self, key, val)
        # update for kwargs
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update formatoptions property by the keywords defined in **kwargs.
        All key words of initialization are possible."""
        self._updating = 1
        kwargs = self._removeoldkeys(kwargs)
        for key, val in kwargs.items():
            self.check_key(key)
            setattr(self, key, val)
        self._updating = 0

    def asdict(self):
        """Returns the non-default FmtBase instance properties as a
        dictionary"""
        fmt = {key[1:]: val for key, val in self.__dict__.items() if key[1:] in
               self._default.keys() and (np.all(val != self._default[key[1:]])
                                         or isinstance(val, dict))}
        return fmt

    def check_key(self, key, raise_error=True, possible_keys=None):
        """Check the key whether it is in possible_keys.

        Input:
          - key: string. Keyword to check
          - raise_error: True/False. If True, raise an error if the key is not
              found, else print a warning.
          - possible_keys: list of possible keys to look in. If None, defaults
              to self._default.keys()
        """
        if possible_keys is None:
            possible_keys = self._default.keys()
        if key not in possible_keys:
            similarkeys = get_close_matches(key, possible_keys)
            if similarkeys == []:
                msg = (
                    "Unknown formatoption keyword %s! See function "
                    "show_fmtkeys for possible formatopion keywords") % (
                        key)
                if raise_error:
                    raise KeyError(msg)
                else:
                    self.logger.warning(msg)
            else:
                msg = (
                    'Unknown formatoption keyword %s! Possible similiar '
                    'frasings are %s.') % (
                        key,
                        ', '.join(key for key in similarkeys))
                if raise_error:
                    raise KeyError(msg)
                else:
                    self.logger.warning(msg)

    def get_fmtdocs(self, *args):
        """
        Function which returns a dictionary containing all possible
        formatoption settings as keys and their documentation as value"""
        return OrderedDict((
            (key, getattr(self.__class__, key).__doc__) for key in
                self.get_fmtkeys(*args)))

    def get_fmtkeys(self, *args):
        """
        Function which returns a list containing all possible
        formatoption settings as keys and their documentation as value"""
        args = [arg for arg in args if arg not in ['wind', 'windonly', 'xy',
                                                   'simple']]
        if args == []:
            return sorted(self._default.keys())
        else:
            for arg in (arg for arg in args
                        if arg not in self._default.keys()):
                self.check_key(arg, raise_error=False)
                args.remove(arg)
            return list(args)

    def _get_fmtdocs_formatted(self, *args):
        """
        Function which returns a formatted string with the keys and
        documentations of all formatoption keywords in a readable manner"""
        doc = self.get_fmtdocs(*args)
        return '\n\n'.join(map(": ".join, doc.items())).encode('utf-8')

    def show_fmtdocs(self, *args):
        """
        Function which prints the keys and documentations of all
        formatoption keywords in a readable manner"""
        print(self._get_fmtdocs_formatted(*args))

    def _get_fmtkeys_formatted(self, *args, **kwargs):
        """
        Function which prints the formatoption keywords in a readable
        manner
        Parameters
        ----------
        *args
            Any formatoption keyword (if None, all are printed)
        indent: int, optional
            Indent of the lines (Default: 0)"""
        keys = self.get_fmtkeys(*args)
        indent = " " * kwargs.get('indent', 0)
        nkeys = 4  # keys per line
        bars = indent + ("="*18 + "  ")*nkeys
        lines = (''.join(key.ljust(20) for key in keys[i:i+nkeys])
                 for i in xrange(0, len(keys), nkeys))
        #empty = nkeys - len(keys) % nkeys
        #lines[-1] += "..".ljust(20)*empty
        text = bars + "\n" + indent + ("\n" + indent ).join(
            lines).encode('utf-8')
        return text + "\n" + bars

    def show_fmtkeys(self, *args):
        """
        Function which prints the formatoption keywords in a readable
        manner"""
        print(self._get_fmtkeys_formatted(*args))

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

    def _removeoldkeys(self, entries):
        """Method to remove wrong keys and modify them"""
        if 'tight' in entries and self.tight != self._default['tight']:
            warn("There is no update method for 'tight'! You have to "
                 "reset the plot!")
        return entries
