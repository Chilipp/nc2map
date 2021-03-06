DEPRECATED
=========
[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

This module is not maintained. Please use the [psyplot](https://github.com/Chilipp/psyplot) module instead!

Module to plot netCDF files (interactively)

This module is attempted to handle netCDF files with the use of
python package netCDF4 and to plot the with the use of python
package matplotlib.
Requirements:
   - matplotlib version, 1.3.1
   - mpl_toolkits.basemap, version 1.07
   - netCDF4, version 1.1.3
   - Python 2.7
   (May even work with older packages, but without warranty.)

Optional Requirements:
   - seaborn (for violin evaluator)
   - shapely (for lineshapes formatoption keyword)
   - shapefile (for lineshapes formatoption keyword)

Main class for usage is the Maps object class. Please look into nc2map/demo
for demonstration scripts and into docs/user_manual/user_manual.pdf for a
rough documentation. For a more detailed documentation use the python help
function.

If you find any bugs, please do not hesitate to contact the authors.
This is nc2map version 0.0beta, so there might be some bugs. Furthermore please
note that there will be significant changes to the API in the near future.

Example usage:

Load package via

    import nc2map

Assume you have a netCDF file named "myncfile.nc" with the variables
"t2m" (temperature) and "pr" (pressure) and 4 timesteps.

Simplest usage:

    mymaps = nc2map.Maps("my-ncfile.nc")

will open two figures, one for each variable and for the first time
step (time=0).
To plot each variable in a single figure but with the first and third 
timesteps variable with all timesteps into a single figure use the ax
keywords and specify the indices for the time dimension:

    mymaps = nc2map.Maps("myncfile.nc", time=[0,2], ax=(1,2))

which will open two figures with one row of subplots.
To modify the colorbar label right from the initialization you can use
the formatoption keyword:

    mymaps = nc2map.maps("myncfile.nc", time=[0, 2], ax=(1,2), \
        fmt={'clabel':'My colorbar label'})

You can also do this interactively after opening the figures via the
update method:

    mymaps.update(clabel='My colorbar label')

To undo changes made use the undo method:

    mymaps.undo()

To make a movie out of the netCDF file, use the make_movie method:

    mymaps.make_movie("mymovie.gif")

More formatoptions are explained by the function

    nc2map.show_fmtkeys()
    
and
    nc2map.show_fmtdocs()
    
