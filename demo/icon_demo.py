#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple example script how to make plots of netCDF files on the ICON
grid with the nc2map module

This script is part of the nc2map Python module, version 0.0b.
The nc2map.Maps class is the basic class which can be used for a plot.
This script shows you an easy example how to open it and how to save a pdf
and make a movie. Look also into the update_demo.py. This is only one feature
of the nc2map module. The most important, the interactive feature, is shown in
the update_demo.py script.
This script creates a pdf icon_demo.pdf. It requires the NetCDF file
'icon_demo.nc' which you can find in the nc2map/demo directory."""
import nc2map
output = 'icon_demo.pdf'
ncfile = 'icon_demo.nc'  # name of the netCDF file

# now we set some formatoptions for the plot. You can however set any
# formatoption interactively via the update method (see update_demo.py)
fmt = {'figtitle': 'Icon demo plot',
       # use the variable name, month and year as plot title
       'title': '%(var)s, {dinfo}',
       'cmap': 'white_blue_red',  # colormap
       # colorlabel like 'Temperature [K]'
       'clabel': '%(long_name)s [%(units)s]',
       'proj': 'robin'  # use robinson projection
       }

# open the Maps instance. First argument is the NetCDF file, vlst denotes the
# variable names (as list or string) and fmt gives the formatoptions.
mymaps = nc2map.Maps(ncfile, vlst='t2m', fmt=fmt)

# mymaps.show()  # show the figures
mymaps.output(output)  # save the figure
mymaps.close()  # close the Maps instance

# some additional tips:
# nc2map.show_fmtkeys()                     # show format option keywords
# nc2map.show_fmtdocs('lsm')                # show docu for keyword
# nc2map.show_colormaps()                   # show available colormaps
# to reverse colormaps: use extend the colormap by '_r',
# e.g. 'white_blue_red_r' --> red_blue_white'
# mymaps.update(fmtkey=value)  # updates the plot with the given
# formatoptions, e.g.
# mymaps.update(title="new Plot")           # update map with title
# mymaps.update(cmap='white_red_blue')      # set colormap
# mymaps.update(proj='hammer')              # set projection
