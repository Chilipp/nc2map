# -*- coding: utf-8 -*-
"""Module containing default values

This module contains the default values for the respective formatoption class
in nc2map.formatoptions"""
import os
from collections import OrderedDict
from .data.boxes import lonlatboxes

BaseFormatter = OrderedDict([
    ('enable', True),
    ('tight', False),
    ('grid', False),

    # Fontsize properties
    ('ticksize', 'small'),
    ('labelsize', 'medium'),
    ('titlesize', 'large'),
    ('figtitlesize', 12),
    ('fontsize', None),
    ('tickweight', None),
    ('labelweight', None),
    ('titleweight', None),
    ('figtitleweight', None),
    ('fontweight', None),

    # axis color
    ('axiscolor', {'right': None,
                   'left': None,
                   'top': None,
                   'bottom': None}),

    # Label properties
    ('title', None),
    ('figtitle', None),
    ('text', [])
    ])

SimpleFmt = OrderedDict([
    ('ticksize', 'medium'),
    ('ylabel', None),
    ('xlabel', None),
    ('ylim', None),
    ('xlim', None),
    ('scale', None),
    ('yticks', None),
    ('yticklabels', None),
    ('xticks', None),
    ('xticklabels', None),
    ('xrotation', 0),
    ('yrotation', 0),
    ('legend', None)
    ])

FmtBase = OrderedDict([
    # General properties
    ('plotcbar', ['b']),
    ('cmap', 'jet'),
    ('cticksize', 'medium'),
    ('ctickweight', None),

    ('ticks', None),
    ('ticklabels', None),
    ('extend', 'neither'),
    ('rasterized', True),
    ('latlon', True),

    # Colorcode properties
    ('bounds', ('rounded', 11)),
    ('norm', 'bounds'),
    ('opacity', None),

    # labels
    ('clabel', None),

    # basemap properties
    ('lonlatbox', [-180., 180., -90., 90.]),
    ('lineshapes', None),
    ('merilabelpos', None),
    ('paralabelpos', None),
    ('meridionals', 7),
    ('parallels', 5),
    ('proj', 'cyl'),
    ('lsm', True),
    ('countries', False),
    ('land_color', None),
    ('ocean_color', None),

    # masking properties
    ('mask', None)
    ])

FieldFmt = OrderedDict([
    # masking propert
    ('maskleq', None),
    ('maskless', None),
    ('maskgreater', None),
    ('maskgeq', None),
    ('maskbetween', None),
    ('plottype', 'quad'),
    ('grid', None),
    ('windplot', {})
    ])

WindFmt = OrderedDict([
    ('arrowsize', 1.),
    ('arrowstyle', '-|>'),
    ('scale', 1.0),
    ('density', 1.0),
    ('linewidth', 0),
    ('color', 'k'),
    ('streamplot', False),
    ('reduceabove', None),
    ('lengthscale', 'lin'),
    #('legend', None)  # currently not implemented
    ])

readers = {
    'dimnames':{
        'timenames': {'time'},
        'levelnames': {'level', 'lvl', 'lev'},
        'lonnames': {'lon', 'longitude', 'x', 'clon'},
        'latnames': {'lat', 'latitude', 'y', 'clat'},
        'udims': {'ncells', 'nMesh2_face', 'nMesh2_node'}
        },
    'min_circle_ratio': 0.05,
    }

shapes = {
    'boundaryfile': "%s%sdata%scountries_and_continents_vmap0" % (
        os.path.dirname(os.path.abspath(__file__) ), os.path.sep, os.path.sep),
    'default_field': 'COUNTRY',
    }
shapes['shapes'] = shapes['boundaryfile']
shapes['lsm'] = os.path.dirname(shapes['shapes']) + os.path.sep + 'lsmask'

texts = {
    'labels': {
        'tinfo': '%B %d, %Y. %H:%M',
        'dinfo': '%B %d, %Y',
        },
    'default_position': (1., 1.)}
