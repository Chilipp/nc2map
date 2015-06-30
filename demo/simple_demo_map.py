#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple example script how to make plots with the nc2map module

This script is part of the nc2map Python module, version 0.0b.
The nc2map.Maps class is the basic class which can be used for a plot.
This script shows you an easy example how to open it and how to save a pdf
and make a movie. Look also into the update_demo.py. This is only one feature
of the nc2map module. The most important, the interactive feature, is shown in
the update_demo.py script.
This script creates a pdf simple-demo-map.pdf and a movie simple-demo-map.gif.
It requires the NetCDF file 'demo-t2m-u-v.nc' which you can find in the
nc2map/demo directory.
Hint: If you want to see how to apply the formatoptions: Use the
nc2map.show_fmtkeys and nc2map.show_fmtdocs functions."""
import nc2map
output = 'simple-demo-map.pdf'
movie_output = 'simple-demo-map.gif'
ncfile = 'demo-t2m-u-v.nc'  # name of the netCDF file

fmt = {'figtitle': 'Demo plot',
       # use the variable name, month and year as plot title
       'title': '%(var)s, {dinfo}',
       'cmap': 'white_blue_red',  # colormap
       # colorlabel like 'Temperature [K]'
       'clabel': '%(long_name)s [%(units)s]',
       }

mymaps = nc2map.Maps(ncfile,
                     ax=(2,2),  # plot all variables into one figure
                     fmt=fmt)

# furthermore we create a line plot over the first three levels, at 0 degree
# east and 0 degree west
fmt = {
    'title': '%(long_name)s at longitude %(lon)s, latitude %(lat)s',
    'ylabel': '%(long_name)s [%(units)s]',  # ylabel of the plot
    'legend': 'best',  # draw a legend
    'xticks': 2,  # use every second xtick from the default settings
    'xticklabels': '%B, %d',  # format with "Month, day" style
    }
mymaps.addline(ncfile, vlst='u', lon=0, lat=0, level=range(3), fmt=fmt,
               names=['surface', 'level 1', 'level 2'])


# mymaps.show()  # show the figures
mymaps.output(output)  # save the figures as pdf
# make a movie with the figure that contains the maps
mymaps.make_movie(movie_output, fps=1, mode='maps')
mymaps.close()  # close the Maps instance
