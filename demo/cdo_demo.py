#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example script for using climate data operators with the nc2map module

This script is part of the nc2map Python module, version 0.0b.
It shall introduce you how to use the combined features of the Climate Data
Operators and the nc2map module. It requires, that you have a working cdo
binary and the cdo.py python module installed.
You can find a detailed documentation of the cdo operators at
https://code.zmaw.de/projects/cdo
It requires the NetCDF file 'demo-t2m-u-v.nc' which you can find in the
nc2map/demo directory."""
import nc2map
cdo = nc2map.Cdo()

ncfile = 'demo-t2m-u-v.nc'

# first create a figure
# create a nc2map.Maps instance from a cdo operator (here timmean, i.e. the
# mean over all timesteps)
mymaps = cdo.timmean(input=ncfile, returnMaps=['t2m', 'u'])
# you can also pass in a dictionary with initialization keywords for the
# nc2map.Maps instance instead of a list ['t2m', 'u'] or simply None, to
# visualize all variables.
# change the title such that we can recognize it later
mymaps.update(title='Map %(name)s from initialization')

# now lets create some additional maps and lines:
fig, ax = nc2map.subplots(2,2)
ax = iter(ax.ravel())

# return a MapBase instance and add it to mymaps
mapo = cdo.timmean(input=ncfile, returnMap='v')
mapo.ax = next(ax)  # assign the axes subplot
mymaps.addmap(mapo)  # add the map and make the plot
# calculate the seasonal mean and add the second season the our Maps instance
mapo_kwargs = {'var': 't2m', 'time': 1,  # show second season of 't2m'
               'fmt': {'title': '%(season)s',
                       'figtitle': 'Some cdo demo maps and lines'},
               'meta': {'season': 'MAM'}}  # set the meta attribute 'season'
mapo = cdo.seasmean(input=ncfile, returnMap=mapo_kwargs)
mapo.ax = next(ax)
mymaps.addmap(mapo)

# visualize the fldmean of the first level as a lineplot
line = cdo.fldmean(input='-sellevidx,1 ' + ncfile, returnLine='t2m')
line.ax = next(ax)
mymaps.addline(line)
# visualize the fldmean of all levels as a lineplot
line_kwargs = {'vlst': 't2m',                 # variable list
               'level': range(4),             # one line per level
               'fmt': {'legend': 'best'},     # draw a legend
               'pyplotfmt': {'label': 'level %(level)s'}}  # label with level
line = cdo.fldmean(input=ncfile, returnLine=line_kwargs)
line.ax = next(ax)
mymaps.addline(line)

# now something a bit more advanced mimiking the
# nc2map.evaluators.FldMeanEvaluator (which does not need cdos)
# the following command for the cdo.merge operator does the following
#  1. calculate global mean for t2m
#  2. calculate global standard deviation for t2m and renames it to 'std'
fig, ax = nc2map.subplots(1,2)
ax = iter(ax)
cmd = '-fldmean -selname,t2m %s -setname,std -fldstd -selname,t2m %s' % (
    ncfile, ncfile)
line_kwargs = {'vlst': 't2m',                 # variable list
               'pyplotfmt': {'fill': 'std',   # use standard deviation as error
                             'label': '%B'},  # use month as label
               'time': range(5),              # one line per time step
               'fmt': {'legend': 'best',
                       'title': '%(name)s'},     # draw a legend
               'name': 'my-first-fldmean'}    # name of the LinePlot instance
line = cdo.merge(input=cmd, returnLine=line_kwargs)
line.ax = next(ax)
mymaps.addline(line)
# which is essentially the same as
reader = cdo.merge(input=cmd, returnCdf=True)
line_kwargs['name'] = 'my-second-fldmean'  # change the name
mymaps.addline(reader, ax=next(ax), **line_kwargs)
# update ylabel
mymaps.update_lines(ylabel='%(long_name)s [%(units)s]', xticks=3)
mymaps.update_lines(name=['my-first-fldmean', 'my-second-fldmean'],
                    # use long_name as xlabel
                    xlabel='%(dim_long_name)s [%(dim_units)s]')

# save everything
mymaps.output('cdo-demo.pdf')

# Just for completion:
# To get an nc2map.reader.DataField instance
data = cdo.timmean(input=ncfile, returnData='t2m')

# get a nc2map.readers.NCReader from a cdo
reader = cdo.timmean(input=ncfile, returnCdf=True)
