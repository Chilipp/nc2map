#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example script for an automatically created colorbar for multiple plots

This script is part of the nc2map Python module, version 0.0b.
The nc2map.CbarManager allows you to control the colorbar settings (color map,
bounds, etc.) for multiple plots at the same time. This is especially useful if
you want to use automatically calculated bounds.
This script creates a pdf onecbar-demo.pdf.
Hint: If you want to see how to apply the formatoptions: Use the
nc2map.show_fmtkeys and nc2map.show_fmtdocs functions.
It requires the NetCDF file 'demo-t2m-u-v.nc' which you can find in the
nc2map/demo directory"""
import nc2map

output = 'onecbar-demo.pdf'  # final output file of this script
ncfile = 'demo-t2m-u-v.nc'  # name of the netCDF file with the data
fmt = {'figtitle': 'One colorbar demo plot, initial state',
       # use the variable name, month and year as plot title
       'title': '%(var)s, %B %Y',
       'plotcbar': False,  # do not plot colorbars under the maps
       }
onecbar = [
    # first the colorbar options for temperature (both time steps)
    {'var': 't2m',
     'cmap': 'white_blue_red',
     'clabel': '%(long_name)s [%(units)s]',
    },
    # now colorbar options for u and v (Note: we could also use
    # 'var': ['u', 'v'] instead of 'level': 1
    {'level': 1,
     'cmap': 'winter',
     'plotcbar': 'r'  # plot colorbar on the right side of the figure
     },
    # now colorbar options which apply to all of them
    {'clabel': '%(long_name)s [%(units)s]'}
    ]

# now we set up exactly which time steps and so on we want to show.
# Therefore we specify the dimensions directly via their names, in order to
# use two different time steps for temperature but not for u and v)
names = {
    'mapo0': {           # name of the plot instance
        'var': 't2m'},   # variable name
    'mapo1': {           # second plot
        'var': 't2m',
        'time': 1},      # time defaults to 0
    'mapo2': {           # third plot
        'var': 'u',
        'level': 1},     # vertical level step (defaults to 0)
    'mapo3': {           # fourth plot
        'var': 'v',
        'level': 1}}


mymaps = nc2map.Maps(ncfile,
                     ax=(2,2),  # plot all variables into one figure
                     names=names,
                     fmt=fmt,
                     onecbar=onecbar)

# mymaps.show()  # show the figure

# save figure at current state
pdf = mymaps.output(output, returnpdf=True)

# to update now the colorbar, we have to use the update_cbar method
mymaps.update(figtitle='One colorbar demo plot, first update: t2m cbar')
mymaps.update_cbar(var='t2m', bounds='minmax')
mymaps.output(pdf)  # save the figure in the current state
mymaps.update(figtitle='One colorbar demo plot, second update: wind cbar')
mymaps.update_cbar(var='u',  # select all colorbars that control maps of u
                   bounds='roundedsym',  # draw rounded symmetric bounds
                   cmap='blue_white_red'  # change colorbar
                   )
mymaps.output(pdf)  # save the figure in the current state

pdf.close()  # finally save the pdf

mymaps.close()  # close the Maps instance
