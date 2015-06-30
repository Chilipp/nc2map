#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script creates a demo plot for the nc2map user manual

Plot types are:
  1. temperature field
  2. temperature field with overlayed quiver plot
  3. quiver plot
  4. stream plot
"""
import nc2map
ncfile = '../../demo/demo-t2m-u-v.nc'
fmt = {
    'clabel': '%(long_name)s [%(units)s]',
    'mapo0': {
        'lonlatbox': [-35, 70, 30, 80],
        'lineshapes': {'COUNTRY': ['Germany'], 'color': 'r', 'linewidth': '3'},
        'enable': True,
        'cmap': 'coolwarm',
        'windplot': {
            'enable': True,
            #'density': .50,
            }
        },
    'mapo1': {
        'enable': True,
        'cmap': 'coolwarm',
        'windplot': {
            'enable': False,
            }
        },
    'mapo2': {
        'lonlatbox': 'Europe',
        'lineshapes': {'COUNTRY': ['Germany'], 'color': 'r', 'linewidth': '3'},
        'meridionals': 5,
        'enable': False,
        'windplot': {
            #'density': .50,
            'color': 'absolute',
            'clabel': 'Wind speed [%(units)s]'
            }
        },
    'mapo3': {
        'lonlatbox': 'Saudi Arabia|China|India',
        'meridionals': 5,  # draw only five merdionals on the map
        'enable': False,
        'land_color': 'coral',
        'ocean_color': 'aqua',
        'windplot': {
            'linewidth': 'absolute',  # scale linewidth with absolute speed
            'scale': 2.0,  # scaling factor for linewidths
            'density': 4.0,  # increase density of arrows
            'streamplot': True,  # enable streamplot
            'color': 'blue',  # use blue arrows
            }
    }}

mymaps = nc2map.Maps(ncfile, fmt=fmt, ax=(2,2), u='u', v='v', vlst='t2m',
                     time=[0]*4)
mymaps.output('figures/demo-plot-types.pdf', dpi=300)
