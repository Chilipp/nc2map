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
nc2map/demo directory
Hint: If you want to see how to apply the formatoptions: Use the
nc2map.show_fmtkeys and nc2map.show_fmtdocs functions.
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nc2map

output = 'update-demo.pdf'
ncfile = 'demo-t2m-u-v.nc'  # name of the netCDF file
fmt = {'figtitle': 'Demo plot',
       # use the variable name, month and year as plot title
       'title': '%(var)s, %B %Y',
       'cmap': 'white_blue_red',  # colormap
       # colorlabel like 'Temperature [K]'
       'clabel': '%(long_name)s [%(units)s]',
       }

mymaps = nc2map.Maps(ncfile,
                     ax=(2,2),  # plot all variables into one figure
                     fmt={'figtitle': 'Initial state'})
# save figure at current state (the returnpdf keyword makes sure that we can
# save another figure in the same pdf file)
pdf = mymaps.output(output, returnpdf=True)

# mymaps.show()  # show the maps and make it all interactive
# update figtitle, title and colorbar label for all
mymaps.update(figtitle='First update',
              title='%(var)s, level %(level)s',
              clabel='%(long_name)s [%(units)s]')
mymaps.output(pdf)
# update colormap for temperature
mymaps.update(figtitle='Second update',
              cmap='white_blue_red',
              var='t2m')
mymaps.output(pdf)
# update colormap and level for u and v variable
mymaps.update(figtitle='Third update',
              cmap='winter', # use predefined winter color map
              fmt={'level': 1},  # use second vertical level
              var=['u', 'v']  # update only u and v variable
              )
pdf.savefig(plt.gcf())
pdf.close()
mymaps.close()  # close the Maps instance
