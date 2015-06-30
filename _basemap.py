# -*- coding: utf-8 -*-
"""mpl_toolkits.basemap extension for the nc2map module.

This module contains the Basemap class, a subclass of the original
mpl_toolkits.basemap.Basemap class, enhanced by the support of irregular
grids"""
import functools
from mpl_toolkits.basemap import Basemap as BasemapBase
from mpl_toolkits.basemap import _cylproj, _pseudocyl, latlon_default
from matplotlib.tri import Triangulation


def _transformtri(plotfunc):
    # shift data and longitudes to map projection region, then compute
    # transformation to map projection coordinates.
    @functools.wraps(plotfunc)
    def with_transform(self,*args, **kwargs):
        args, kwargs = self._transform1d(*args, **kwargs)
        return plotfunc(self,*args, **kwargs)
    return with_transform

class Basemap(BasemapBase):
    def _transform1d(self, *args, **kwargs):
        # input coordinates are latitude/longitude, not map projection coords.
        if kwargs.pop('latlon', latlon_default):
            # shift data to map projection region for
            # cylindrical and pseudo-cylindrical projections.
            x = args[0].x if isinstance(args[0], Triangulation) else args[0]
            y = args[0].y if isinstance(args[0], Triangulation) else args[1]
            # data is not shifted but converted
            if self.projection in _cylproj or self.projection in _pseudocyl:
                if self.lonmin < 0:
                    x[x > 180.] -= 360.
                elif self.lonmax > 180.:
                    x[x < 0.] += 360.
            # convert lat/lon coords to map projection coords.
            x, y = self(x,y)
            args = list(args)
            if isinstance(args[0], Triangulation):
                args[0].x = x
                args[0].y = y
            else:
                args[0] = x
                args[1] = y
        return args, kwargs

    @_transformtri
    def tripcolor(self, *args, **kwargs):
        """Create a pseudocolor plot of an unstructured triangular grid.

        The triangulation can be specified in one of two ways; either::

        tripcolor(triangulation, ...)

        where triangulation is a matplotlib.tri.Triangulation object, or

        ::

        tripcolor(x, y, ...)
        tripcolor(x, y, triangles, ...)
        tripcolor(x, y, triangles=triangles, ...)
        tripcolor(x, y, mask=mask, ...)
        tripcolor(x, y, triangles, mask=mask, ...)

        in which case a Triangulation object will be created.  See
        matplotlib.tri.Triangulation for a explanation of these
        possibilities.

        The next argument must be `C`, the array of color values, either
        one per point in the triangulation if color values are defined at
        points, or one per triangle in the triangulation if color values
        are defined at triangles. If there are the same number of points
        and triangles in the triangulation it is assumed that color
        values are defined at points; to force the use of color values at
        triangles use the kwarg `facecolors`=C instead of just `C`.

        `shading` may be 'flat' (the default) or 'gouraud'. If `shading`
        is 'flat' and C values are defined at points, the color values
        used for each triangle are from the mean C of the triangle's
        three points. If `shading` is 'gouraud' then color values must be
        defined at points.  `shading` of 'faceted' is deprecated;
        please use `edgecolors` instead.

        The remaining kwargs are the same as for matplotlib.axes.Axes.pcolor.

        Notes
        -----
        This method solves a problem of the mpl_toolkits.basemap.Basemap.pcolor
        method that produces error when the ``latlon`` keyword is set.

        Additional kwargs: hold = [True|False] overrides default hold state"""
        ax, plt = self._ax_plt_from_kw(kwargs)
        # allow callers to override the hold state by passing hold=True|False
        b = ax.ishold()
        h = kwargs.pop('hold', None)
        if h is not None:
            ax.hold(h)
        try:
            ret =  ax.tripcolor(*args, **kwargs)
        except:
            ax.hold(b)
            raise
        ax.hold(b)
        # reset current active image (only if pyplot is imported).
        if plt:
            plt.sci(ret)
        # clip for round polar plots.
        if self.round: ret, c = self._clipcircle(ax, ret)
        # set axes limits to fit map region.
        self.set_axes_limits(ax=ax)
        if self.round:
            # for some reason, frame gets turned on.
            ax.set_frame_on(False)
        return ret

    @_transformtri
    def triplot(self, *args, **kwargs):
        """Draw a unstructured triangular grid as lines and/or markers.

        The triangulation to plot can be specified in one of two ways;
        either::

        triplot(triangulation, ...)

        where triangulation is a :class:`matplotlib.tri.Triangulation`
        object, or

        ::

        triplot(x, y, ...)
        triplot(x, y, triangles, ...)
        triplot(x, y, triangles=triangles, ...)
        triplot(x, y, mask=mask, ...)
        triplot(x, y, triangles, mask=mask, ...)

        in which case a Triangulation object will be created.  See
        :class:`~matplotlib.tri.Triangulation` for a explanation of these
        possibilities.

        The remaining args and kwargs are the same as for
        :meth:`~matplotlib.axes.Axes.plot`.

        Return a list of 2 :class:`~matplotlib.lines.Line2D` containing
        respectively:

        - the lines plotted for triangles edges
        - the markers plotted for triangles nodes

        Notes
        -----
        This method solves a problem of the mpl_toolkits.basemap.Basemap.pcolor
        method that produces error when the ``latlon`` keyword is set.

        Additional kwargs: hold = [True|False] overrides default hold state"""
        ax, plt = self._ax_plt_from_kw(kwargs)
        # allow callers to override the hold state by passing hold=True|False
        b = ax.ishold()
        h = kwargs.pop('hold', None)
        if h is not None:
            ax.hold(h)
        try:
            ret =  ax.triplot(*args, **kwargs)
        except:
            ax.hold(b)
            raise
        ax.hold(b)
        # reset current active image (only if pyplot is imported).
        if plt:
            plt.sci(ret)
        # clip for round polar plots.
        if self.round: ret, c = self._clipcircle(ax, ret)
        # set axes limits to fit map region.
        self.set_axes_limits(ax=ax)
        if self.round:
            # for some reason, frame gets turned on.
            ax.set_frame_on(False)
        return ret


    def quiver(self, *args, **kwargs):
        """
        Notes
        -----
        This method is the same as for mpl_toolkits.basemap.Basemap method but
        has a better support for irregular grids."""
        # docstring is set below
        try:
            return super(Basemap, self).quiver(*args, **kwargs)
        except ValueError:
            args, kwargs = self._transform1d(*args, **kwargs)
            return super(Basemap, self).quiver(*args, **kwargs)

    quiver.__doc__ = BasemapBase.quiver.__doc__ + quiver.__doc__
