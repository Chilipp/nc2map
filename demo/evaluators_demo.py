#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example script for evaluation with the nc2map module

This script is part of the nc2map Python module, version 0.0b.
It's a quite advanced script showing you the main features in the (currently
implemented) fldmean and violin plot evaluation routines.
It creates the output NetCDF file fldmean-t2m-u-v.nc and the plots
evaluators-demo.pdf.
See also the calculation_demo.py file showing you how to perform
arithmetics within the nc2map module and the cdo_demo.py.
It requires the NetCDF file 'demo-t2m-u-v.nc' which you can find in the
nc2map/demo directory"""
import nc2map
output = 'evaluators-demo.pdf'
ncfile = 'demo-t2m-u-v.nc'  # name of the netCDF file

regions = {  # dictionary defining the regions to evaluate
    # keys are the name of the region, values are lists
    # [ncfile, variable-name, value]. If value is not given, it is masked
    # everywhere, where the mask is 0 (or missing value).
    'Global': [],  # global evaluations (with the current mask, i.e. no mask)
    'All regions': ['focus_regions.nc', 'mask'],
    'Pacific': ['focus_regions.nc', 'mask', 1],
    'Amazon': ['focus_regions.nc', 'mask', 2]}

# create subplots with two rows and three columns
fig, ax = nc2map.subplots(2,3, gridspec_kw={'wspace': 0.4, 'bottom': 0.15},
                          figsize=(12,6))
# NOTE: Using the nc2map.subplots method instead of the plt.subplots method
# will save the settings from the initialization. This is useful for the
# Maps.save method

fmt = {'title': '%(long_name)s', 'clabel': '%(var)s [%(units)s]',
     'figtitle': '{dinfo}', 'figtitlesize': 'xx-large'}
# create plots and plot the maps in the first row
mymaps = nc2map.Maps(ncfile, ax=ax[0], time=0, level=0, fmt=fmt)

# ---- make fldmean plots
# plot fldmeans of the specified regions below them and use the standard
# deviation as error range
fmt = {
    # enable minor xticks and use every third of the default ones.
    'xticks': {'major': 2, 'minor': 3},
    'xticklabels': {'major': '%B',  # label major ticks with month
                    'minor': '%d'},  # label minor ticks with day
    'title': '',  # disable the title (defaults to the corresponding map name),
    # draw a legend below the figure (keywords are determined by plt.legend,
    # this may be improved in the future)
    'legend': {'loc': 'center', 'bbox_to_anchor': (0.5, 0.04), 'frameon': True,
               'bbox_transform': fig.transFigure, 'ncol': 4,
               'fontsize': 'large'}
    }
mymaps.evaluate('fldmean', mymaps.maps, ax=ax[1], regions=regions,
                error='std', fmt=fmt,
                merge=True,  # merge the new data into one single reader
                # dump the data into a NetCDF file
                ncfiles=['fldmeans-of-t2m-u-v.nc']
                )
# the new fldmeans are saved as a new ArrayReader instance. The merge keyword
# made sure, that the fldmean evaluator created only one single reader.
# We can now simply dump the new data into a NetCDF file with the Maps.dump_nc
# method, or why not save the whole figure setting, such that it can be
# reloaded with the nc2map.load function?
mymaps.save('evaluators-demo.pkl')
# now we could restore the whole project with exactly the same settings with
# mymaps = nc2map.load('evaluators-demo.pkl')

# ---- make violin plots of the regions specified
# we make two rows of violin plots. One for temperature, one for u and v
fig, ax = nc2map.subplots(2, len(regions), sharey='row')

# -- violin plots of temperature
violin_evaluator = mymaps.evaluate(
    'violin', mymaps.get_maps(var='t2m', mode='maps'), regions=regions,
    ax=ax[0], names='')
# NOTE: setting names='' disables xticklabels
# NOTE: The ViolinEvaluator does not create new readers as the FldMeanEvaluator
# does. Therefore the data of the ViolinPlots cannot be restored when the plots
# are closed

# -- violin plots of u and v
fmt = {
    # change ylabel of most left plot
    'All regions': {'ylabel': 'Wind speed [%(units)s]'},
    # manually disable title (otherwise it will default to %(region)s
    'title': None,
    }
# now make the evaluation with xticklabels being the variable name
violin_evaluator = mymaps.evaluate(
    'violin', mymaps.get_maps(var=['u','v'], mode='maps'), regions=regions,
    ax=ax[1], fmt=fmt, names='%(var)s')

# The violin evaluators by default disables the ViolinPlot since it does not
# work with the Maps.update method. Instead use the ViolinEvaluator.update
# method. However, for the output, we enable them.
violinplots = mymaps.get_disabled()  # store disabled ViolinPlot instances
mymaps.enable_maps(maps=violinplots)

# save everything and return the PdfPages instance for further usage
mymaps.output(output)

# disable ViolinPlot instances again
mymaps.disable_maps(maps=violinplots)
