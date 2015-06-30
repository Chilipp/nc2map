#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example script for the visualization of wind data

This script is part of the nc2map Python module, version 0.0b.
The nc2map.mapos.FieldPlot allows you to plot a scalar variable (e.g.
temperature) and a vector variable (e.g. wind) above. The
nc2map.mapos.WindPlot controls the appearance of windplots.
This script creates a pdf wind-demo-map.pdf.
It requires the NetCDF file 'demo-t2m-u-v.nc' which you can find in the
nc2map/demo directory
Hint: If you want to see how to apply the formatoptions: Use
nc2map.show_fmtkeys('wind') and nc2map.show_fmtdocs('wind').
"""
import nc2map

output = 'wind-demo-map.pdf'
ncfile = 'demo-t2m-u-v.nc'  # name of the netCDF file
fmt = {'figtitle': 'Wind demo plot',
       'figtitlesize': 'x-large',
       'clabel': '%(long_name)s [%(units)s]'}

# ------ combined field and wind plot --------
fmt['quiver'] = {
    'title': 'Quiver demo plot',
    'lonlatbox': 'Europe',  # focus on Europe
    # draw a red shape around Germany
    'lineshapes': {'COUNTRY': ['Germany'], 'color': 'r', 'linewidth': '3'},
    'meridionals': 5,  # draw only five merdionals on the map
    # set enable to False if you do not want to show the underlying temperature
    # field
    'enable': True,
    'windplot': {  # wind plot specific options.
        # See nc2map.show_fmtdocs('wind', 'windonly')
        # set enable to False if you do not want to show the overlayed
        # wind field (same holds of course for the quiver plot)
        'enable': True,
        'density': 1.0,  # reduce number of arrows by 50 percent
        }
    }

fmt['stream'] = {
    'title': 'Stream demo plot',
    # focus on Middle East, China and India
    'lonlatbox': 'Saudi Arabia|China|India',
    'meridionals': 5,  # draw only five merdionals on the map
    'windplot': {
        'linewidth': 'absolute',  # scale linewidth with absolute speed
        'scale': 2.0,  # scaling factor for linewidths
        'density': 4.0,  # increase density of arrows
        'streamplot': True,  # enable streamplot
        'color': 'blue',  # use blue arrows
        }
    }

mymaps = nc2map.Maps(ncfile, fmt=fmt,
                     u='u',  # variable name of zonal wind component in ncfile
                     v='v',  # variable name of merdionaly wind component
                     vlst='t2m',  # dummy name for wind variable
                     time=[0, 0],  # plot two times the first timestep
                     names=['quiver', 'stream'],  # names of the WindPlots
                     ax=(1,2))

# save figure at current state
pdf = mymaps.output(output, returnpdf=True)
mymaps.close()

# ------ stand alone wind plot --------
# We can also avoid the FieldPlot (here 't2m') completely and plot the wind
# arrows alone. This has the advantage, that we avoid the additional 'windplot'
# option.
# Therefore we now use the same settings as above with additional ocean color
# and land color for the stream plot and a color coding for the quiver plot.

# set windplot options to main formatoption dictionary
fmt['quiver'].update(fmt['quiver'].pop('windplot'))
fmt['stream'].update(fmt['stream'].pop('windplot'))
del fmt['clabel']

# use color coding for quiver plot
fmt['quiver'].update({
    'color': 'absolute',
    'clabel': 'Wind speed [%(units)s]'})

# color land and oceans for stream plot
fmt['stream'].update({
    'land_color': 'coral',  # color land
    'ocean_color': 'aqua'  # color oceans
    })

# open nc2map.Maps instance with windonly keyword
mymaps = nc2map.Maps(ncfile, fmt=fmt, u='u', v='v', time=[0, 0],
                     names=['quiver', 'stream'], ax=(1,2), windonly=True)

mymaps.output(pdf)
pdf.close()
#mymaps.close()
