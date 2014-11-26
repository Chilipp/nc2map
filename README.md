nc2map.py
=========

Module to plot netCDF files (interactively)

This module is attempted to handle netCDF files with the use of
python package netCDF4 and to plot the with the use of python
package matplotlib.
Requirements:
   - matplotlib version, 1.3.1
   - numpy
   - netCDF4, version 1.0.9
   - Python 2.7
   (May even work with older packages, but without warranty.)
Main class for usage is the maps object class. A helper function 
for the formatoption keywords is show_settingdocs, displaying the
documentation of all formatoption keywords.
If you find any bugs, please do not hesitate to contact the authors.
This is nc2map version 0.0beta, so there might be some bugs.

Example usage:
  Load package via
  >>> import nc2map
  Assume you have a netCDF file named "myncfile.nc" with the variables
  "t2m" (temperature) and "pr" (pressure) and 4 timesteps.
  Simplest usage:
    >>> mymaps = nc2map.maps("myncfile.nc")
  will open two figures, one for each variable and for the first time
  step (time=0).
  To plot each variable in a single figure but with the first and third 
  timesteps variable with all timesteps into a single figure use the ax
  and times keywords:
    >>> mymaps = nc2map.maps("myncfile.nc", times = [0,2], ax = (1,2))
  which will open two figures with one row of subplots.
  To modify the colorbar label right from the initialization you can use
  the formatoption keyword:
    >>> mymaps = nc2map.maps("myncfile.nc", times = [0,2], ax = (1,2), \
        formatoptions = {'clabel':'My colorbar label'})
  You can also do this interactively after opening the figures via the
  update method:
    >>> mymaps.update(clabel='My colorbar label')
  To undo changes made use the undo method:
    >>> mymaps.undo()
  To make a movie out of the netCDF file, use the make_movie method:
    >>> mymaps.make_movie("mymovie_<<<var>>>.gif")
  More formatoptions are explained in function
    >>> nc2map.show_settingdocs()
