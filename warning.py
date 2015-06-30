# coding: utf-8
"""Warning module of the nc2map python module

This module controls the warning behaviour of the module via the python
builtin warnings module and introduces two new warning classes:
Nc2MapWarning and Nc2MapCritical"""
import warnings
import logging


# disable a warning about "comparison to 'None' in backend_pdf which occurs
# in Maps.output method
warnings.filterwarnings(
    'ignore', 'comparison', FutureWarning, 'matplotlib.backends.backend_pdf',
    2264)


logger = logging.getLogger(__name__)


class Nc2MapRuntimeWarning(RuntimeWarning):
    """Runtime warning that appears only ones"""
    pass

class Nc2MapWarning(UserWarning):
    """Normal UserWarning for nc2map module"""
    pass


class Nc2MapCritical(UserWarning):
    """Critical UserWarning for nc2map module"""
    pass


warnings.simplefilter('always', Nc2MapWarning, append=True)
warnings.simplefilter('always', Nc2MapCritical, append=True)


def disable_warnings(critical=False):
    """Function that disables all warnings and all critical warnings (if
    critical evaluates to True) related to the nc2map Module.
    Please note that you can also configure the warnings via the
    nc2map.warning logger (logging.getLogger(nc2map.warning))."""
    warnings.filterwarnings('ignore', '\w', Nc2MapWarning, 'nc2map', 0)
    if critical:
        warnings.filterwarnings('ignore', '\w', Nc2MapCritical, 'nc2map', 0)


def warn(message, category=Nc2MapWarning, logger=None):
    """wrapper around the warnings.warn function for non-critical warnings.
    logger may be a logging.Logger instance"""
    if logger is not None:
        message = "[Warning by %s]\n%s" % (logger.name, message)
    warnings.warn(message, category, stacklevel=2)


def critical(message, category=Nc2MapCritical, logger=None):
    """wrapper around the warnings.warn function for critical warnings.
    logger may be a logging.Logger instance"""
    if logger is not None:
        message = "[Warning by %s]\n%s" % (logger.name, message)
    warnings.warn(message, category, stacklevel=2)


old_showwarning = warnings.showwarning


def customwarn(message, category, filename, lineno, *args, **kwargs):
    """Use the nc2map.warning logger for categories being out of
    Nc2MapWarning and Nc2MapCritical and the default warnings.showwarning
    function for all the others."""
    if category is Nc2MapWarning:
        logger.warning(warnings.formatwarning(
            "\n%s" % message, category, filename, lineno))
    elif category is Nc2MapCritical:
        logger.critical(warnings.formatwarning(
            "\n%s" % message, category, filename, lineno))
    else:
        old_showwarning(message, category, filename, lineno, *args, **kwargs)


warnings.showwarning = customwarn
