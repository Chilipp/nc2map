#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example script for calculations inside the nc2map module

This script is part of the nc2map Python module, version 0.0b.
It shall introduce you into the data management within the nc2map module
and show you how per efficiently perform arithmetics and visualization at
the same time.
Within this script, we will calculate the absolute wind speed in two
different ways. It produces a NetCDF file calculation-demo.nc and a plot
calculation-demo.pdf.
It requires the NetCDF file 'demo-t2m-u-v.nc' which you can find in the
nc2map/demo directory"""
import nc2map
import os
ncfile = 'demo-t2m-u-v.nc'
output = 'calculation-demo.pdf'
output_nc = 'calculation-demo.nc'

# open nc2map.Maps instance
mymaps = nc2map.Maps(ncfile, vlst=['u', 'v'], ax=(1,2),
                     fmt={'clabel': '%(long_name)s [%(units)s]'})

# get the MapBase instance with zonal wind velocity
mapo_u = mymaps.get_maps(var='u')[0]
mapo_v = mymaps.get_maps(var='v')[0]

# calculate the speed. The return is again a MapBase instance with the
# formatoptions of mapo_u. It will plot into a new subplot.
mapo_speed = (mapo_u*mapo_u + mapo_v*mapo_v)**0.5

# add it to the current Maps instance (to change the meta information, look
# at the end of the script)
mymaps.addmap(mapo_speed)

# save the plots into a pdf
pdf = mymaps.output(output, returnpdf=True)

# However, the calculation with MapBase instances will always only consider the
# specific level, time, etc. that are shown by the MapBase instance. Hence if
# you now try
# >>> mymaps.update(maps=mapo_speed, fmt={'time': 1})
# you will get an IndexError. This is behaviour is very efficient if you have
# large datasets. However, if you want to consider all the data, we have to
# go one level deeper, to the reader level
reader = mapo_u.reader

# a reader (an instance of the nc2map.readers.ArrayReaderBase) is the
# representation of the NetCDF file in the nc2map module. It has essentially
# the same structure as the netCDF4.Dataset class (in fact, it uses this
# class). All the meta data is stored in this reader.
# To calculate the wind speed for all times and levels, we can extract the
# specific variable and calculate as we did above
speed_reader = (reader.selname('u')**2 + reader.selname('v')**2)**0.5
# if you want to be more specific on the times and levels, use the
# ArrayReaderBase.extract method in place of the selname method

# The new variable contains all meta informations of the 'u' variable in the
# old reader. We rename the variable name in the speed reader
speed_reader.renameVariable('u', 'speed')
# and change the global meta information
speed_reader.set_meta(history='Calculated speed with nc2map module')
# and the local long_name attribute of the 'speed' variable
speed_reader.set_meta('speed', long_name='Absolute wind speed')

# now we can add a new MapBase instance to our Maps
mymaps.addmap(speed_reader, vlst='speed', fmt={
    'clabel': '%(long_name)s [%(units)s]',
    'title': 'New mapo with full reader'})

# save the figure
mymaps.output(pdf, var='speed')
# and why not merge the new variable with the existing NetCDF file and save it
speed_reader.merge(reader).dump_nc(output_nc, compression=4)

# NOTE: In the first calculation when creating the mapo_speed MapBase
# instance, we did not change any meta information, it contains everything
# from the reader and the 'u' variable.
# Here is how you can do it
mapo_speed.reader.renameVariable('u', 'speed')
mapo_speed.reader.set_meta('speed', long_name='Absolute wind speed')
mapo_speed.var = 'speed'
mapo_speed.update()

#mymaps.close()
