# -*- coding: utf-8 -*-
"""This file contains a class for basic formatoptions of simple x-y plot
"""
import matplotlib.pyplot as plt
from _basefmtproperties import BaseFmtProperties
from _base_fmt import BaseFormatter
from ..defaults import SimpleFmt as default

__author__ = "Philipp Sommer (philipp.sommer@stud.uni-hamburg.de)"
__version__ = '0.0'


props = BaseFmtProperties()


class SimpleFmt(BaseFormatter):
    ylabel = props.default(
        'ylabel', """
        string (Default: %s). Defines the y-axis label""" %
        default['ylabel'])
    xlabel = props.default(
        'xlabel', """
        string (Default: %s). Defines the x-axis label""" %
        default['xlabel'])
    xrotation = props.default(
        'xrotation', """
        float (Default %s). Degrees between 0 and 360 for which the
        xticklabels shall be rotated""" % default['xrotation'])
    yrotation = props.default(
        'yrotation', """
        float (Default %s). Degrees between 0 and 360 for which the
        xticklabels shall be rotated""" % default['yrotation'])
    ylim = props.default(
        'ylim', """
        tuple (Default: %s). Specifies the limits of the y-axis""" %
        default['ylim'])
    xlim = props.default(
        'xlim', """
        tuple (Default: %s). Specifies the limits of the x-axis""" %
        default['xlim'])
    scale = props.default(
        'scale', """
        string ('logx', 'logy', 'logxy') (Default: %s). Sets the scale
        of the x and/or y-axis (See also 'xformat' and 'yformat' for
        scientific axis description)""" % default['scale'])
    yticks = props.ticks_and_labels(
        'yticks', """
        integer, 1D-array or dictionary (Default: %s). Defines the y-ticks.
          - If None, the automatically calculated y-ticks will be used.
          - If integer i, every i-th tick of the automatically calculated ticks
              will be used.
          - If 1D-array, those will be used for the yticks.
          - If dictionary, possible keys are 'minor' for minor ticks and
              'major' for major ticks. Values can be in any of the styles
              described above. Another possible key is 'pad' to define the
              vertical difference between minor and major ticks. By default,
              those are calculated from the ticksize formatoption""" % (
                  default['yticks']))
    xticks = props.ticks_and_labels(
        'xticks', """
        integer, 1D-array or dictionary (Default: %s). Defines the x-ticks.
          - If None, the automatically calculated x-ticks will be used.
          - If integer i, every i-th tick of the automatically calculated ticks
              will be used.
          - If 1D-array, those will be used for the xticks.
          - If dictionary, possible keys are 'minor' for minor ticks and
              'major' for major ticks. Values can be in any of the styles
              described above. Another possible key is 'pad' to define the
              vertical difference between minor and major ticks. By default,
              those are calculated from the ticksize formatoption""" % (
                  default['xticks']))
    yticklabels = props.ticks_and_labels(
        'yticklabels', """
        format string, 1D-array or dictionary (Default: %s). Defines the
        y-axis ticklabels.
          - If None, the automatically calculated y-ticklabels will be used.
          - If format string (e.g. '%%0.0f' for integers, '%%1.2e' for
              scientific or '%%b' for the month if time is plotted on the
              axis.
          - If 1D-array, those will be used for the yticklabels. (Note: The
              length should match to the used yticks
          - If dictionary, possible keys are 'minor' for minor ticks and
              'major' for major ticks. Values can be in any of the styles
              described above.
       Note: To enable minor ticks, you use the yticks formatoption""" % (
           default['yticklabels']))
    xticklabels = props.ticks_and_labels(
        'xticklabels', """
        format string, 1D-array or dictionary (Default: %s). Defines the
        y-axis ticklabels.
          - If None, the automatically calculated y-ticklabels will be used.
          - If format string (e.g. '%%0.0f' for integers, '%%1.2e' for
              scientific or '%%b' for the month if time is plotted on the
              axis.
          - If 1D-array, those will be used for the yticklabels. (Note: The
              length should match to the used yticks
          - If dictionary, possible keys are 'minor' for minor ticks and
              'major' for major ticks. Values can be in either of the styles
              described above.
       Note: To enable minor ticks, you use the xticks formatoption""" % (
           default['xticklabels']))
    legend = props.legend(
        'legend', """
        location value or dictionary (Default: %s). Draw a legend on the axes.
        If string or integer, this will be used for the location keyword. If
        dictionary, the settings of this dictionary will be used. Possible
        keys for the dictionary are given in the following parameter list:\n"""
        % (default['legend']) +
        '\n'.join(map(lambda x: '        ' + x, plt.legend.__doc__[
            plt.legend.__doc__.find('Parameters'):].splitlines())))

    def __init__(self, **kwargs):
        """initialization and setting of default values.
        Key word arguments may be any names of a property. Use show_fmtkeys
        for possible keywords and their documentation"""
        super(SimpleFmt, self).__init__()
        self._default.update(default)

        # set default values
        for key, val in default.items():
            setattr(self, key, val)

        # update for kwargs
        self.update(**kwargs)
