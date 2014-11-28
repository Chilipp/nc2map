# -*- coding: utf-8 -*-
"""Module to plot netCDF files (interactively)

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
for the formatoption keywords is show_fmtkeys, displaying the
documentation of all formatoption keywords.
If you find any bugs, please do not hesitate to contact the authors.
This is nc2map version 0.0beta, so there might be some bugs.
"""

__version__ = "0.00b"
__author__  = "Philipp Sommer (philipp.sommer@studium.uni-hamburg.de)"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.basemap as bm
import sys
import netCDF4 as nc
from itertools import izip, chain

currentmap = None
openmaps = []
cmapnames = { # names of self defined colormaps (see get_cmap function below)
             'red_white_blue':[(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (1, 1., 1), (1, 1., 1), (0, 1, 1), (0, 0.5, 1), (0, 0, 1)], # symmetric water fluxes
             'blue_white_red':[(0, 0, 1), (0, 0.5, 1), (0, 1, 1), (1, 1., 1), (1, 1., 1), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)], # symmetric temperature
             'white_blue_red':[(1, 1., 1), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)], # temperature
             'white_red_blue':[(1, 1., 1), (1, 0, 0), (1, 1, 0), (0, 1, 1), (0, 0, 1)] # water fluxes
            }


def get_cmap(cmap, N=11):
  """Returns a colormap. Extended version of pyplots get_cmap function
  via additional colormaps (see below and cmapnames)
  Input:
    - cmap: string or colormap (e.g. matplotlib.colors.LinearSegmentedColormap).
      If cmap is a colormap, nothing will happen. Otherwise if cmap is a string,
      a colorbar will be chosen. Possible strings are
      -- 'red_white_blue' (e.g. for symmetric precipitation colorbars)
      -- 'blue_white_red' (e.g. for symmetric temperature colorbars)
      -- 'white_blue_red' (e.g. for asymmetric temperature colorbars)
      -- any standard colorbar name from pyplot
    - N: Integer (Default: 11). Number of increments in the colormap
  
  Output:
    - colormap instance
  """
  if cmap in cmapnames:
    defcmap = mpl.colors.LinearSegmentedColormap.from_list(name=cmap, colors =cmapnames[cmap],N=N)
  elif cmap in plt.cm.datad:
    defcmap = plt.get_cmap(cmap, N)
  else:
    defcmap = cmap
  return defcmap

def show_colormaps(*args):
  """Script to show standard colormaps from pyplot. Taken from
  http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
  and slightly adapted in November 2014.
  *args may be any names as strings of standard colorbars (e.g. 'jet',
  'Greens', etc.)."""
  # This example comes from the Cookbook on www.scipy.org.  According to the
  # history, Andrew Straw did the conversion from an old page, but it is
  # unclear who the original author is."""
  import numpy as np
  import matplotlib.pyplot as plt
  from difflib import get_close_matches
  a = np.linspace(0, 1, 256).reshape(1,-1)
  a = np.vstack((a,a))
  # Get a list of the colormaps in matplotlib.  Ignore the ones that end with
  # '_r' because these are simply reversed versions of ones that don't end
  # with '_r'
  for arg in (arg for arg in args if arg not in plt.cm.datad.keys() + cmapnames.keys()): 
    similarkeys = get_close_matches(arg, plt.cm.datad.keys()+cmapnames.keys())
    if similarkeys != []: print("Colormap " + arg + " not found in standard colormaps. Similar colormaps are " + ', '.join(key for key in similarkeys))
    else: print("Colormap " + arg + " not found in standard colormaps. Run function without arguments to see all colormaps")
  if args == (): maps = sorted(m for m in plt.cm.datad.keys()+cmapnames.keys() if not m.endswith("_r"))
  else: maps = sorted(m for m in plt.cm.datad.keys()+cmapnames.keys() if m in args)
  nmaps = len(maps) + 1
  fig = plt.figure(figsize=(5,10))
  fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
  for i,m in enumerate(maps):
      ax = plt.subplot(nmaps, 1, i+1)
      plt.axis("off")
      plt.imshow(a, aspect='auto', cmap=get_cmap(m), origin='lower')
      pos = list(ax.get_position().bounds)
      fig.text(pos[0] - 0.01, pos[1], m, fontsize=10, horizontalalignment='right')
  plt.show()

def returnbounds(data, bounds):
  """Returns automatically generated bounds.
  Input:
  - data: Array. Data used for generation of the bounds
  - 'bounds' 1D-array or tuple (Default:('rounded', 11, True):
     Defines the bounds used for the colormap. Possible types are
    + 1D-array: Defines the bounds directly by giving the values
    + tuple (string, N): Compute the bounds automatically. N gives
      the number of increments whereas <string> can be one of the
      following strings
       ++ 'rounded': Rounds min and maxvalue of the data to the next
          lower (in case of minimum) or higher (in case of maximum)
          0.5-value with respect to the exponent with base 10 of the
          maximal range (i.e. if minimum = -1.2e-4, maximum = 1.8e-4,
          min will be -1.5e-4, max 2.0e-4) using round_to_05 function.
       ++ 'roundedsym': Same as 'rounded' but symmetric around zero
          using the maximum of the data maximum and (absolute value of)
          data minimum.
       ++ 'minmax': Uses minimum and maximum of the data (without roun-
          ding)
       ++ 'sym': Same as 'minmax' but symmetric around 0 (see 'rounded'
          and 'roundedsym').
    + tuple (string, N, percentile): Same as (string, N) but uses the
      percentiles defined in the 1D-list percentile as maximum. percen-
      tile must have length 2 with [minperc, maxperc]
    + string: same as tuple with N automatically set to 11
  """
  from copy import deepcopy
  exp=np.floor(np.log10(abs(np.max(data)-np.min(data))))
  if type(bounds) is str: bounds = (bounds, 11)
  if isinstance(bounds[0], str):
    N=bounds[1]
    # take percentiles as boundary definitions
    if len(bounds) == 3:
      perc = deepcopy(bounds[2])
      if perc[0] == 0:	perc[0] = np.min(data)
      else:	perc[0] = np.percentile(data,perc[0])
      if perc[1] == 100:	perc[1] = np.max(data)
      else:	perc[1] = np.percentile(data,perc[1])
      if perc[1] == perc[0]:	print('Attention!: Maximum and Minimum bounds are the same! Using max value for maximal bound.'); perc[1]=np.max(data)
      data = deepcopy(np.ma.masked_outside(data, perc[0], perc[1], copy = True))
    if bounds[0] == 'rounded':
      cmax = np.max((round_to_05(np.max(data), exp, np.ceil), round_to_05(np.max(data), exp, np.floor)))
      cmin = np.min((round_to_05(np.min(data), exp, np.floor), round_to_05(np.min(data), exp, np.ceil)))
    elif bounds[0] == 'minmax':
      cmax = np.max(data)
      cmin = np.min(data)
    elif bounds[0] == 'roundedsym':
      cmax = np.max((np.max((round_to_05(np.max(data), exp, np.ceil), round_to_05(np.max(data), exp, np.floor))), np.abs(np.min((round_to_05(np.min(data), exp, np.floor), round_to_05(np.min(data), exp, np.ceil))))))
      cmin = - cmax
    elif bounds[0] == 'sym':
      cmax = np.max((np.max(data), np.abs(np.min(data))))
      cmin = - cmax
    bounds = np.linspace(cmin,cmax,N,endpoint=True)
  return bounds

def round_to_05(n, exp=None, func=round):
  """Applies the round function specified in func to round n to the
  next 0.5-value with respect to its exponent with base 10 (i.e.
  1.3e-4 will be rounded to 1.5e-4) if exp is None or with respect
  to the given exponent in exp.
  Input:
    - n: Float, number to round
    - exp: Integer. Exponent for rounding
    - func: Rounding function
  Output:
    - Rounded n
  """
  from math import log10, floor
  if exp is None:
    exp=floor(log10(abs(n))) # exponent for base 10
  ntmp=np.abs(n)/10.**exp # mantissa for base 10
  if np.abs(func(ntmp) - ntmp) >= 0.5:
    return np.sign(n)*(func(ntmp) - 0.5)*10.**exp
  else:
    return np.sign(n)*func(ntmp)*10.**exp

def gcm():
  """Return the current maps instance"""
  return currentmap

def scm(mymaps):
  """Sets the current maps instance"""
  currentmap = mymaps

def close():
  """close all open maps instances"""
  for mymaps in openmaps: mymaps.close()
  openmaps = []

def update(fmt={}, add = True, delete = True, todefault = False, **kwargs):
  """Shortcut to the current maps instance update function"""
  currentmap.update(fmt,add,delete,todefault, **kwargs)

def show_fmtkeys(*args):
  """Function which prints the keys and documentations in a readable manner.
  Arguments (*args) may be any keyword of the formatoptions (without: Print all);
  Keyword arguments may be wind = True if the wind options shall be displayed (has only an effect when windonly is not True)"""
  # docstring is just for information and will be replaced below with the docstring of maps instance
  if not 'wind' in args: myfmt = fieldfmt()
  else: myfmt = windfmt()
  myfmt.show_fmtkeys(*args)

def get_fmtkeys(*args):
  """Function which returns a dictionary containing all possible formatoption settings as keys and their documentation as value.
  Arguments (*args) may be any keyword of the formatoptions;
  Keyword arguments may be wind = True if the wind options shall be displayed (has only an effect when windonly is not True)"""
  # docstring is just for information and will be replaced below with the docstring of maps instance
  if not 'wind' in args: myfmt = fieldfmt()
  else: myfmt = windfmt()
  return myfmt.get_fmtkeys(*args)

def get_docs(*args):
  """shortcut for get_fmtkeys"""
  return get_fmtkeys(*args)

class fmtproperties():
  """class containg property definitions of formatoption containers fmtBase and subclasses fieldfmt and windfmt"""
  def default(self,x,doc):
    """default property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def bmprop(self,x,doc):
    """default basemap property (currently not in use)"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._default.setdefault(x, getattr(self, '_' + x))
      if x not in self._bmprops: self._bmprops.append(x)
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def cmap(self, x, doc):
    """Property controlling the colormap. The setter is disabled if self._enablebounds is False"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      if self._enablebounds:
        setattr(self, '_' + x, value)
      else: print("Setting of colormap is disabled. Use the update_cbar function or removecbars first.")
      self._default.setdefault(x, getattr(self, '_' + x))
      if x not in self._cmapprops: self._cmapprops.append(x)
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def cmapprop(self,x,doc):
    """default colormap property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._default.setdefault(x, getattr(self, '_' + x))
      if x not in self._cmapprops: self._cmapprops.append(x)
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def bounds(self,x,doc):
    """bound property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,bounds):
      if self._enablebounds:
        if type(bounds) is str: setattr(self,'_' + x,(bounds, 11))
        else: setattr(self, '_' + x, bounds)
        self._default.setdefault(x, getattr(self, '_' + x))
      else: print("Setting of bounds is disabled. Use the update_cbar function or removecbars first.")
      if x not in self._cmapprops: self._cmapprops.append(x)
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def lonlatbox(self,x,doc):
    """lonlatbox property to automatically configure projection after lonlatbox is changed"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      # update properties depending on lonlatbox
      setattr(self,'_box', value)
      if hasattr(self, 'proj'): self.proj = self.proj
      self._default.setdefault(x, getattr(self, '_' + x))
      if x not in self._bmprops: self._bmprops.append(x)
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def proj(self,x,doc):
    """Projection property to automatically configure projops"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,projection):
      setattr(self,'_' + x,projection)
      if projection == 'northpole':
        self._defaultrange = [-180.,180.,0.,90]
        if self._box == self.glob: self._box = self._defaultrange
        self._projops= {'projection':'npstere', 'lon_0':270, 'boundinglat':self._box[2], 'llcrnrlon':self._box[0], 'urcrnrlon':self._box[1], 'llcrnrlat':self._box[2], 'urcrnrlat':self._box[3]} # projection options for basemap
        self.meridionals = self.meridionals
        if self.merilabelpos is None: self._meriops['labels'] = [1,1,1,1]
        self.parallels   = self.parallels
        if self.paralabelpos is None: self._paraops['labels'] = [0,0,0,0]
      elif projection == 'southpole':
        self._defaultrange = [-180.,180.,-90.,0.]
        if self._box == self.glob: self._box = self._defaultrange
        self._projops= {'projection':'spstere', 'lon_0':90, 'boundinglat':self._box[3], 'llcrnrlon':self._box[0], 'urcrnrlon':self._box[1], 'llcrnrlat':self._box[2], 'urcrnrlat':self._box[3]} # projection options for basemap
        # update meridionals and parallels
        self.meridionals = self.meridionals
        if self.merilabelpos is None: self._meriops['labels'] = [1,1,1,1]
        self.parallels   = self.parallels
        if self.paralabelpos is None: self._paraops['labels'] = [0,0,0,0]
      elif projection == 'cyl':
        # update meridionals and parallels
        self._defaultrange = self.glob
        self._projops= {'projection':'cyl', 'llcrnrlon':self._box[0], 'urcrnrlon':self._box[1], 'llcrnrlat':self._box[2], 'urcrnrlat':self._box[3]} # projection options for basemap
        # update meridionals and parallels
        self.meridionals = self.meridionals
        if self.merilabelpos is None: self._meriops['labels'] = [0,0,0,1]
        self.parallels   = self.parallels
        if self.paralabelpos is None: self._paraops['labels'] = [1,0,0,0]
      self._default.setdefault(x, getattr(self, '_' + x))
      if x not in self._bmprops: self._bmprops.append(x)
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def meridionals(self,x,doc):
    """property to configure and initialize meridional plotting options"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,values):
      setattr(self, '_'+x, values)
      if type(values) is int: self._meriops.update({x: np.linspace(self._box[0],self._box[1],values, endpoint = True),'fontsize':self.ticksize, 'latmax':90})
      else: self._meriops.update({x:values,'fontsize':self.ticksize, 'latmax':90})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def parallels(self,x,doc):
    """property to configure and initialize parallel plotting options"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,values):
      setattr(self,'_'+x,values)
      if type(values) is int: self._paraops.update({x:np.linspace(self._box[2],self._box[3],values, endpoint = True),'fontsize':self.ticksize, 'latmax':90})
      else: self._paraops.update({x:values,'fontsize':self.ticksize, 'latmax':90})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def merilabelpos(self,x,doc):
    """property to define axes where to plot meridional labels"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self, value):
      setattr(self,'_'+x,value)
      if value is not None:
        self._meriops.update({'labels':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def paralabelpos(self,x,doc):
    """property to define axes where to plot parallel labels"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self, value):
      setattr(self,'_'+x,value)
      if value is not None:
        self._paraops.update({'labels':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def windplotops(self, x, doc):
    """wind plot property updating wind plot options"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      if hasattr(self, '_streamplot'): self._windplotops.update({x:value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def streamplot(self, x, doc):
    """streamplot property to configure windplotops"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._windplotops = {}
      if value is True: # set streamplot options
        for attr in ['arrowsize', 'arrowstyle', 'density', 'linewidth', 'color', 'cmap']: setattr(self, attr, getattr(self,attr))
      else: # set quiver options
        for attr in ['color', 'cmap', 'rasterized', 'scale', 'linewidth']: setattr(self, attr, getattr(self,attr))
        if self.linewidth is None: self.linewidth = 0
        self._windplotops['units']='xy'
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def fontsize(self,x,doc):
    """default fontsize options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      if not value is None:
        for label in ['figtitle', 'tick', 'title', 'label']:
          if getattr(self, label + 'size') == self._default[label+'size']:
            getattr(self,'_'+label+'ops').update({'fontsize':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def ticksize(self,x,doc):
    """tick fontsize options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._tickops.update({'fontsize':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def figtitlesize(self,x,doc):
    """figtitle fontsize options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._figtitleops.update({'fontsize':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def titlesize(self,x,doc):
    """title fontsize options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._titleops.update({'fontsize':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def labelsize(self,x,doc):
    """axes label fontsize options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._labelops.update({'fontsize':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def fontweight(self,x,doc):
    """default fontweight options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      for label in ['figtitle', 'tick', 'title', 'label']:
        if getattr(self, label + 'weight') == self._default[label+'weight']:
          getattr(self,'_'+label+'ops').update({'fontweight':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def tickweight(self,x,doc):
    """tick fontweight options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._tickops.update({'fontweight':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def figtitleweight(self,x,doc):
    """figtitle fontweight options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._figtitleops.update({'fontweight':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def titleweight(self,x,doc):
    """title fontweight options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._titleops.update({'fontweight':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def labelweight(self,x,doc):
    """axes label fontweight options property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._labelops.update({'fontweight':value})
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def maskprop(self,x,doc):
    """property for masking options like maskabove or maskbelow"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      setattr(self, '_' + x, value)
      self._maskprops.append(x)
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def text(self,x,doc):
    """property to add text to the figure"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      if value == []: oldtexts = value; textstoupdate = []
      else:
        if isinstance(value, tuple): value = [value]
        textstoupdate = []
        for text in value:
          if all(trans not in text for trans in ['axes','fig','data']): text = tuple(list(value).insert(3,'data'))
          oldtexts = getattr(self,'_'+x)
          append = True
          for oldtext in oldtexts:
            if all(oldtext[i] == text[i] for i in [0,1,3]):
              if text[2] == '': oldtexts.remove(oldtext)
              else: oldtexts[oldtexts.index(oldtext)] = text; textstoupdate.append(text)
              append = False
          if append: oldtexts.append(text); textstoupdate.append(text)
      self._textstoupdate = (text for text in textstoupdate)
      setattr(self, '_' + x, oldtexts)
      self._default.setdefault(x, getattr(self, '_' + x))
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)
  def plotcbar(self,x,doc):
    """default colormap property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      if value == True: value = 'b'
      elif value in [False, None]: value = ''
      setattr(self, '_' + x, value)
      self._default.setdefault(x, getattr(self, '_' + x))
      if x not in self._cmapprops: self._cmapprops.append(x)
    def delx(self): setattr(self, '_'+x,self._default[x])
    return property(getx,setx,delx,doc)

class fmtBase(object):
  """Base class of formatoptions. Documented properties (plotcbar, rasterized, etc.)
  are what can be set as formatoption keywords in any mapBase and maps instance."""
  props = fmtproperties() # container containing methods for property definition
  # ------------------ define properties here -----------------------
  # General properties
  plotcbar      = props.plotcbar('plotcbar', """String, tuple or list of strings (Default: 'b'). Determines
                     where to plot the colorbar. Possibilities are
                     'b' for at the bottom of the plot, 'r' for at
                     the right side of the plot, 'sh' for a horizontal
                     colorbar in a separate figure, 'sv' for a vertical
                     colorbar in a separate figure. For no colorbar use '', or None, or False.""")
  rasterized    = props.default('rasterized', """Boolean (Default: True). Rasterize the pcolormesh
                     (i.e. the mapplot) or not.""")
  cmap          = props.cmap('cmap',"""string or colormap (e.g.
                     matplotlib.colors.LinearSegmentedColormap) (Default: plt.cm.jet). 
                     Defines the used colormap. If cmap is a colormap, nothing will
                     happen. Otherwise if cmap is a string, a colorbar will be chosen.
                     Possible strings are
                       - 'red_white_blue' (e.g. for symmetric precipitation color-
                          bars)
                       - 'white_red_blue' (e.g. for asymmetric precipitation color-
                          bars)
                       - 'blue_white_red' (e.g. for symmetric temperature colorbars)
                       - 'white_blue_red' (e.g. for asymmetric temperature colorbars)
                       - any other name of a standard colorbar as provided by pyplot
                         (e.g. 'jet','Greens','binary', etc.). Use function
                         nc2map.show_colormaps to visualize them.""")
  ticks         = props.cmapprop('ticks', """1D-array or integer (Default: None). Define the ticks of the
                     colorbar. In case of an integer i, every i-th value of the
                     default ticks will be used (might not work always).""")
  extend        = props.cmapprop('extend', """string  (‘neither’, ‘both’, ‘min’ or ‘max’) (Default:
                     'neither'). If not ‘neither’, make pointed end(s) for out-of-
                     range values. These are set for a given colormap using the 
                     colormap set_under and set_over methods.""")
  orientation   = props.cmapprop('orientation', """string ('horizontal' or 'vertical') (Default: 'horizontal').
                     Specifies the orientation of the colorbar""")
  tight         = props.default('tight', """Boolean (Default: False). Make tight_layout after plotting if True.""")
  
  # fontsizes and fontweights
  fontsize      = props.fontsize('fontsize', """string or float. Defines the default size of ticks, axis labels
                     and title. Strings might be ‘xx-small’, ‘x-small’, ‘small’,
                     ‘medium’, ‘large’, ‘x-large’, ‘xx-large’. Floats define the
                     absolute font size, e.g., 12""")
  ticksize      = props.ticksize('ticksize',"""string or float (Default: 'small'). Defines the size of the
                     ticks (see fontsize for possible values)""")
  figtitlesize      = props.figtitlesize('figtitlesize', """string or float (Default: 12). Defines the size of the
                     subtitle of the figure (see fontsize for possible values).
                     This is the title of this specific axes! For the title of the
                     figure see figtitlesize""")
  titlesize      = props.titlesize('titlesize', """string or float (Default: 'large'). Defines the size of the
                     title (see fontsize for possible values)""")
  labelsize      = props.labelsize('labelsize', """string or float (Default: 'medium'). Defines the size of x- 
                     and y-axis labels (see fontsize for possible values)""")
  fontweight     = props.fontweight('fontweight', """A numeric value in the range 0-1000 or string (Default: None).
                     Defines the fontweight of the ticks. Possible strings are one
                     of ‘ultralight’, ‘light’, ‘normal’, ‘regular’, ‘book’, ‘medium’,
                     ‘roman’, ‘semibold’, ‘demibold’, ‘demi’, ‘bold’, ‘heavy’,
                     ‘extra bold’, ‘black’.""")
  tickweight     = props.tickweight('tickweight', """Fontweight of ticks (Default: Defined by fontweight property).
                     See fontweight above for possible values.""")
  figtitleweight     = props.figtitleweight('figtitleweight', """Fontweight of the figure suptitle (Default: Defined by
                     fontweight property). See fontweight above for possible values.""")
  titleweight     = props.titleweight('titleweight', """Fontweight of the title (Default: Defined by fontweight proper-
                     ty). See fontweight above for possible values. This is the
                     title of this specific axes! For the title of the figure see
                     figtitleweight""")
  labelweight     = props.labelweight('labelweight', """Fontweight of axis labels (Default: Defined by fontweight pro-
                     perty). See fontweight above for possible values.""")
  
  
  # labels
  figtitle      = props.default('figtitle', """string (Default: None). Defines the figure suptitle of the
                     plot. Strings <<<var>>>, <<<longname>>>, <<<unit>>>, 
                     <<<time>>> and <<<level>>> will be replaced by the
                     corresponding short or longnames, units, times or levels
                     as stored in the netCDF file from all mapBase instances
                     in the figure separated by comma.""")
  title         = props.default('title', """string (Default: None). Defines the title of the plot. Strings
                     <<<var>>>, <<<longname>>>, <<<unit>>>, <<<time>>> and 
                     <<<level>>> will be replaced by the corresponding short
                     or longname, unit, time or level as stored in the netCDF
                     file.
                     This is the title of this specific axes! For the title of the
                     figure see figtitle""")
  clabel        = props.default('clabel', """string (Default: None). Defines the label of the colorbar
                     (if plotcbar is True). Strings <<<var>>>, <<<longname>>>,
                     <<<unit>>>, <<<time>>> and <<<level>>> will be replaced by
                     the corresponding short or longname, unit, time or level as
                     stored in the netCDF file.""")
  ticklabels    = props.default('ticklabels', """Array. Defines the ticklabels of the colorbar""")
  text          = props.text('text', """Tuple or list of tuple (x,y,s[,coord.-system][,options]]) (Default:
                     []). Each list object defines a text instance on the plot.
                     0<=x, y<=1 are the coordinates either in data coordinates
                     (default, 'data') or in axes coordinates ('ax') or figure
                     coordinate ('fig'). s (string) is the text. options might
                     be options to specify 'color', 'fontweight', 'fontsize',
                     etc.. To remove one single text from the plot, set
                     (x,y,'') for the text at position (x,y); to remove all set
                     text=[]. Strings <<<var>>>, <<<longname>>>, <<<unit>>>, 
                     <<<time>>> and <<<level>>> will be replaced by the
                     corresponding short or longnames, units, times or levels
                     as stored in the netCDF file from all mapBase instances
                     in the figure separated by comma.""")
  
  # basemap properties
  lonlatbox     = props.lonlatbox('lonlatbox', """1D-array [lon1,lon2,lat1,lat2] (Default: global, i.e.
                     [-180.0,180.0,-90.0,90.0] for proj=='cyl' and Norther Hemi-
                     sphere for 'northpole' and Southern for 'southpole'). Selects
                     the region for the plot""")
  proj          = props.proj('proj', """string ('cyl', 'northpole', 'southpole', Default: 'cyl').
                     Defines the options for the projection used for the plot.""")
  meridionals   = props.meridionals('meridionals',"""1D-array or integer (Default: 7). Defines the 
                     lines where to draw meridionals. Possible types 
                     are
                       - 1D-array: manually specify the location of
                         the meridionals
                       - integer: Gives the number of meridionals 
                         between maximal and minimal longitude (in-
                         cluding max- and minimum line)""")
  parallels     = props.parallels('parallels',"""1D-array or integer (Default: 5). Defines the lines
                     where to draw parallels. Possible types are
                       - 1D-array: manually specify the location of
                         the parallels
                       - integer: Gives the number of parallels 
                         between maximal and minimal lattitude (inclu-
                          ding max- and minimum line)""")
  merilabelpos  = props.merilabelpos('merilabelpos', """List of 4 values (default [0,0,0,0]) that control
                     whether meridians are labelled where they intersect
                     the left, right, top or bottom of the plot. For
                     example labels=[1,0,0,1] will cause meridians
                     to be labelled where they intersect the left and
                     and bottom of the plot, but not the right and top.""")
  paralabelpos  = props.paralabelpos('paralabelpos', """List of 4 values (default [0,0,0,0]) that control
                     whether parallels are labelled where they intersect
                     the left, right, top or bottom of the plot. For
                     example labels=[1,0,0,1] will cause parallels
                     to be labelled where they intersect the left and
                     and bottom of the plot, but not the right and top.""")
  lsm           = props.default('lsm', """Boolean (Default: True). If True, the continents will be plot-
                     tet.""")
  countries     = props.default('countries', """Boolean (Default: False). If True, draw country borders.""")
  land_color    = props.default('land_color', """color instance (Default: 'w'). Specify the color of the land.
                     Attention! Might reduce the performance a lot if many figures
                     are opened! To not kill everything. There is not update method
                     for land_color and ocean_color. You need to reset the maps
                     instance.""")
  ocean_color   = props.default('ocean_color', """color instance (Default: 'w'). Specify the color of the ocean.
                     Attention! Might reduce the performance a lot if many figures
                     are opened! To not kill everything. There is not update method
                     for land_color and ocean_color. You need to reset the maps
                     instance.""")
  
  # Colorcode properties
  bounds        = props.bounds('bounds',"""1D-array, tuple or string (Default:('rounded', 11): Defines the
                     bounds used for the colormap. Possible types are
                       - 1D-array: Defines the bounds directly by giving the values
                       - tuple (string, N): Compute the bounds automatically. N
                         gives the number of increments whereas <string> can be one
                         of the following strings
                         --'rounded': Rounds min and maxvalue of the data to the
                           next 0.5-value with respect to its exponent with base 10
                           (i.e. 1.3e-4 will be rounded to 1.5e-4) if Bool is True,
                           or to the next integer if Bool is false
                         --'roundedsym': Same as 'rounded' but symmetric around
                            zero using the maximum of the data maximum and (absolute
                            value of) data minimum.
                         -- 'minmax': Uses minimum and maximum of the data (without
                            rounding)
                         -- 'sym': Same as 'minmax' but symmetric around 0 (see
                         'rounded' and 'roundedsym').
                       - tuple (string, N, percentile): Same as (string, N) but
                         uses the percentiles defined in the 1D-list percentile as
                         maximum. percentile must have length 2 with
                         [minperc, maxperc]
                       - string: same as tuple with N automatically set to 11.
    """)
  
  def __init__(self, **kwargs):
    """initialization and setting of default values. Key word arguments may be
    any names of a property. Use show_fmtkeys for possible keywords and
    their documentation"""
    # dictionary for default values. The default values and keys will be set during the initilization (this is implemented in the property definitions)
    self._default          = {}
    
    # Option dictionaries
    self._projops          = {} # settings for projection
    self._meriops          = {} # settings for meridionals on map
    self._paraops          = {} # settings for parallels on map
    self._tickops          = {} # settings for tick labels
    self._labelops         = {} # setting for axis labels
    self._titleops         = {} # settings for title
    self._figtitleops      = {} # settings for figure suptitle
    self._bmprops          = [] # keys of the important properties for the basemap which force reinitialization of the mapBase object (currently: lonlatbox and proj)
    self._cmapprops        = [] # keys of colormap properties (important for the dialog between fieldplot and windplot instance.
    self._enablebounds     = True # if false, no changes can be made for cmap or bounds (see property definition)
    # self._general, the list containing the names of the baseclass keywords, is defined below
    
    
    # General properties
    self.plotcbar         = True
    self.cmap             = 'jet'
    self.ticks            = None
    self.ticklabels       = None
    self.extend           = 'neither'
    self.orientation      = 'horizontal'
    self.rasterized       = True
    self.tight            = False
    
    # Colorcode properties
    self.bounds           = ('rounded', 11)
    
    # Fontsize properties
    self.ticksize         = 'small'
    self.labelsize        = 'medium'
    self.titlesize        = 'large'
    self.figtitlesize     = 12
    self.fontsize         = None
    self.tickweight       = None
    self.labelweight      = None
    self.titleweight      = None
    self.figtitleweight   = None
    self.fontweight       = None
    
    # Label properties
    self.title            = None
    self.figtitle         = None
    self.clabel           = None
    self.text             = []
    
    # basemap properties
    self.glob             = [-180.,180.,-90.,90.]
    self.lonlatbox        = self.glob
    self.merilabelpos     = None
    self.paralabelpos     = None
    self.meridionals      = 7
    self.parallels        = 5
    self.proj             = 'cyl'
    self.lsm              = True
    self.countries        = False
    self.land_color       = 'w'
    self.ocean_color      = None
    
    self._general = sorted(self._default.keys()) # base class properties
    self.update(**kwargs)
  
  def update(self,**kwargs):
    """Update formatoptions property by the keywords defined in **kwargs.
    All key words of initialization are possible."""
    for key, val in kwargs.items():
      if key not in self._default:
        from difflib import get_close_matches
        similarkeys = get_close_matches(key, self._default.keys())
        if similarkeys == []: sys.exit('Unknown formatoption keyword ' + key + '! See function show_fmtkeys for possible formatopion keywords')
        else: sys.exit('Unknown formatoption keyword ' + key + '! Possible similiar frasings are ' + ', '.join(key for key in similarkeys) + '.')
      else: setattr(self,key,val)
  
  def asdict(self):
    """Returns the non-default fmtBase instance properties as a dictionary"""
    fmt = {key[1:]:val for key, val in self.__dict__.items() if key[1:] in self._default.keys() and val != self._default[key[1:]]}
    return fmt
  
  def get_fmtkeys(self, *args):
    """Function which returns a dictionary containing all possible
    formatoption settings as keys and their documentation as value"""
    if args == (): return {key:getattr(self.__class__,key).__doc__ for key in self._default.keys()}
    else: return {key:getattr(self.__class__,key).__doc__ for key in args}
  
  def show_fmtkeys(self, *args):
    """Function which prints the keys and documentations of all 
    formatoption keywords in a readable manner"""
    doc = self.get_fmtkeys(*args)
    print '\n'.join((key+':').ljust(20) + doc[key] for key in sorted(doc.keys()))
  
class fieldfmt(fmtBase):
  """Class to control the formatoptions of a fieldplot instance. See function 
  show_fmtkeys for formatoption keywords.
  """
  props = fmtproperties() # container containing methods for property definition
  
  # masking properties
  maskbelow     = props.maskprop('maskbelow', """Float (Default: None). Value under which the data shall be 
                     masked""")
  maskabove     = props.maskprop('maskabove', """Float (Default: None). Value under which the data shall be
                     masked""")
  maskbetween   = props.maskprop('maskbetween', """Tuple or list (Default: None). Pair (min, max) between which
                     the data shall be masked""")
  
  # wind plot property
  windplot      = props.default('windplot', """windfmt object. Defines the properties of the wind plot""")
  
  def __init__(self, **kwargs):
    """initialization and setting of default values. Key word arguments may be
    any names of a property. Use show_fmtkeys for possible keywords and
    their documentation"""
    super(fieldfmt, self).__init__()
    # first remove old keys
    kwargs = self._removeoldkeys(kwargs)
    
    self._maskprops = []
    # masking propert
    self.maskbelow   = None
    self.maskabove   = None
    self.maskbetween = None
    
    # set windfmt with options stored in 'windplot' and general options as defined in kwargs
    self.windplot        = windfmt()
    if 'windplot' in kwargs: windops = kwargs.pop('windplot')
    else: windops = {}
    windops.update({key:kwargs[key] for key in self._general if key in kwargs and key not in self._cmapprops})
    self.windplot.update(**windops)
    
    for key, val in kwargs.items():
      if key not in self._default:
        from difflib import get_close_matches
        similarkeys = get_close_matches(key, self._default.keys())
        if similarkeys == []: sys.exit('Unknown formatoption keyword ' + key + '! See function show_fmtkeys for possible formatopion keywords')
        else: sys.exit('Unknown formatoption keyword ' + key + '! Possible similiar frasings are ' + ', '.join(key for key in similarkeys) + '.')
      else: setattr(self,key,val)
  
  def update(self,updatewind=False,**kwargs):
    """Update formatoptions property by the keywords defined in **kwargs.
    All key words of initialization are possible."""
    # set windfmt with options stored in 'windplot' and general options as defined in kwargs
    if 'windplot' in kwargs: windops = kwargs.pop('windplot')
    else: windops = {}
    if updatewind:
      windops.update({key:kwargs[key] for key in self._general if key in kwargs})
      self.windplot.update(**windops)
    for key, val in kwargs.items():
      if key not in self._default:
        from difflib import get_close_matches
        similarkeys = get_close_matches(key, self._default.keys())
        if similarkeys == []: sys.exit('Unknown formatoption keyword ' + key + '! See function show_fmtkeys for possible formatopion keywords')
        else: sys.exit('Unknown formatoption keyword ' + key + '! Possible similiar frasings are ' + ', '.join(key for key in similarkeys) + '. For more possible keyword formatoptions see function show_fmtkeys')
      else: setattr(self,key,val)
  
  def asdict(self):
    """Returns the non-default fmtBase instance properties as a dictionary"""
    fmt = {key[1:]:val for key, val in self.__dict__.items() if key[1:] in self._default.keys() and val != self._default[key[1:]] and key[1:] != 'windplot'}
    fmt.update({'windplot':self.windplot.asdict(False)})
    if fmt['windplot'] == {}: fmt.pop('windplot')
    return fmt
    
  
  def _removeoldkeys(self, entries):
    if 'stream' in entries: entries['windplot'] = entries.pop('stream')
    return entries

class windfmt(fmtBase):
  """Class to control the formatoptions of a windplot instance. See function 
  show_fmtkeys for formatoption keywords.
  """
  props = fmtproperties() # container containing methods for property definition
  # ------------------ define properties here -----------------------
  # general properties
  enable        = props.default('enable', """Boolean (Default: True). Allows the windplot on the axes""")
  rasterized    = props.windplotops('rasterized', """Boolean (Default: True). Rasterize the pcolormesh (i.e. the
                     mapplot) or not.""")
  
  # Arrow properties
  arrowsize     = props.windplotops('arrowsize', """float (Default: 1). Defines the size of the arrows""")
  arrowstyle    = props.windplotops('arrowstyle', """str (Default: '-|>'). Defines the style of the arrows (See
                     :class:`~matplotlib.patches.FancyArrowPatch`)""")
  linewidth     = props.windplotops('linewidth', """float, string ('absolute', 'u' or 'v') or 2D-array (Default:
                     None). Defines the linewidth behaviour. Possible types are
                       - float: give the linewidth explicitly
                       - 2D-array (which has to match the shape of of u and v):
                         The values determine the linewidth according to the given
                         numbers
                       - 'absolute', 'u' or 'v': a normalized 2D-array is computed
                         and makes the colorcode corresponding to the absolute flow
                         of u or v. A further scaling can be done via the 'scale' key
                         (see above). Higher 'scale' corresponds to higher linewidth.""")
  density       = props.windplotops('density', """Float (Default: 1.0). Value scaling the density""")
  scale         = props.windplotops('scale', """Float (Default: 1.0). Scales the length of the arrows""")
  lengthscale      = props.default('lengthscale', """String (Default: 'lin'). If 'log' the length of the quiver
                     plot arrows are scaled logarithmically via
                     speed=sqrt(log(u)^2+log(v)^2)""")
  
  # colorcode properties
  bounds       = props.bounds('bounds', fmtBase.bounds.__doc__ + """
                     Note: Bounds do only have an effect for a quiver plot in a
                     windplot object but not for a streamplot, because the colormap
                     of the streamplot does not support normalization like
                     pcolormesh.""")
  color        = props.windplotops('color', """string ('absolute', 'u' or 'v'), matplotlib color code or
                     2D-array (Default: 'k' (i.e. black)). Defines the color behaviour.
                     Possible types are
                       - 2D-array (which has to match the shape of of u and v): The
                         values determine the colorcoding according to 'cmap'
                       - 'absolute', 'u' or 'v': a color coding 2D-array is computed
                         and make the colorcode corresponding to the absolute flow
                         or u or v.
                       - single letter ('b': blue, 'g': green, 'r': red, 'c': cyan,
                         'm': magenta, 'y': yellow, 'k': black, 'w': white): Color for
                         all arrows
                       - float between 0 and 1 (defines the greyness): Color for all
                         arrows
                       - html hex string (e.g. '#eeefff'): Color for all arrows""")
  cmap          = props.windplotops('cmap', fieldfmt.cmap.__doc__)
  
  # masking properties
  reduce           = props.default('reduce', """Float f with 0<f<=100 (Default: None, i.e. 100). Reduces resolution
                     to the given percentage of wind data using the weighted mean""")
  reduceabove    = props.default('reduceabove', """Tuple or list (perc, pctl) with floats between 0 and 100 (Default:
                     None). Reduces the resolution to 'perc' of the original resolution
                     if in the area defined by 'perc' average speed is higher than the
                     pctl-th percentile.""")
  
  # style and additional labeling properties
  streamplot    = props.streamplot('streamplot', """Boolean (Default: False). If True, a pyplot.streamplot() will be
                     used instead of a pyplot.quiver()""")
  legend           = props.default('legend', """Float or list of floats (Default: None). Draws quiverkeys over
                     the plot""")
  
  
  # initialization
  def __init__(self, **kwargs):
    """initialization and setting of default values. Key word arguments may be
    any names of a property. Use show_fmtkeys for possible keywords and
    their documentation"""
    super(windfmt, self).__init__()
    # Option dictionaries
    self._windplotops     = {}
    
    # General properties
    self.enable = True
    
    self.arrowsize       = 1.
    self.arrowstyle      = '-|>'
    self.scale           = 1.0
    self.density         = 1.0
    self.linewidth       = None
    self.color           = 'k'
    self.streamplot      = False
    self.reduce          = None
    self.reduceabove     = None
    self.lengthscale     = 'lin'
    self.legend          = None
    for key, val in kwargs.items():
      if key not in self._default:
        sys.exit('Unknown formatoption keyword ' + key + '!')
      else: setattr(self,key,val)
  
  def asdict(self, general = True):
    """Returns the non-default fmtBase instance properties as a dictionary"""
    if general: fmt = {key[1:]:val for key, val in self.__dict__.items() if key[1:] in self._default.keys() and val != self._default[key[1:]]}
    else: fmt = {key[1:]:val for key, val in self.__dict__.items() if key[1:] in self._default.keys() and val != self._default[key[1:]] and key[1:] not in self._general}
    return fmt
  
  def get_fmtkeys(self, *args):
    """Function which returns a dictionary containing all possible
    formatoption settings as keys and their documentation as value.
    Use as args 'windonly' if only those keywords specific to the
    windfmt object shall be returned."""
    args = tuple(arg for arg in args if arg != 'wind') # filter 'wind' out
    if args == (): return {key:getattr(self.__class__,key).__doc__ for key in self._default.keys()}
    elif args == tuple(['windonly']):
      return {key:getattr(self.__class__,key).__doc__ for key in self._default.keys() if key not in self._general}
    elif 'windonly' in args: return {key:getattr(self.__class__,key).__doc__ for key in args if key not in self._general + ['windonly', 'wind']}
    else: return {key:getattr(self.__class__,key).__doc__ for key in args}
  
  def show_fmtkeys(self, *args):
    super(windfmt, self).show_fmtkeys(*args)
  
  def _removeoldkeys(self, entries):
    return entries
  
  # ------------------ modify docstrings here --------------------------
  show_fmtkeys.__doc__ = '    ' + fmtBase.show_fmtkeys.__doc__ + "\n    Use as args 'windonly' if only those keywords specific to the\n    windfmt instance shall be shown."

class mapproperties():
  """class containg property definitions of class maps, mapBase, and subclasses"""
  def default(self,x,doc):
    """default property"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value): setattr(self, '_' + x, value)
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def mapdata(self, x,doc):
    def getx(self): return getattr(self,'_'+x)
    def setx(self,value):
      if value == {}: setattr(self, '_' + x, value)
      elif isinstance(value, mapBase):
        if self._initcompatible: ending = 'orig'
        else: ending = ''
        if not getattr(value,'name'+ending) in getattr(self,'_'+x).keys(): getattr(self,'_'+x).update({getattr(value,'name'+ending):{}})
        if not 't' + str(getattr(value,'time'+ending)) in getattr(self,'_'+x)[getattr(value,'name'+ending)].keys(): getattr(self,'_'+x)[getattr(value,'name'+ending)].update({'t' + str(getattr(value,'time'+ending)):{}})
        getattr(self,'_'+x)[getattr(value,'name'+ending)]['t' + str(getattr(value,'time'+ending))].update({'l'+str(getattr(value,'level'+ending)):value})
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def dim(self,x,doc):
    """Dimension property for longitude and latitude. Sets up the dimension by using self.nco and the given dimension name"""
    def getx(self): return getattr(self,'_'+x)
    def setx(self, names):
      try:
        for name in names:
          try: setattr(self, '_'+x,self.nco.variables[name][:]); exist = True; break
          except KeyError: exist = False
        if not exist: sys.exit('Unknown dimension name ' + name + '! Possible keys in the ncfile are ' + ','.join(key for key in self.nco.variables.keys()))
      except TypeError: setattr(self, '_'+x, names)
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def data(self,x,doc):
    def getx(self): return getattr(self,'_'+x)
    def setx(self, name):
      if isinstance(name, str):
        data = self._get_data(name, self.time, self.level)
        #shift data
        data = self._shift(data)
        # mask array if not global is needed
        data = self._mask_to_region(data)
        # mask data if whished
        if hasattr(self, '_mask_data'): data = self._mask_data(data)
        setattr(self, '_'+x,data)
      else: setattr(self,'_'+x,name)
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def nco(self,x,doc):
    """property of the netCDF4.Dataset instance"""
    def getx(self): 
      # return open nco
      return getattr(self,'_'+x)
    def setx(self,value): 
      if value is None:
        setattr(self, '_' + x, nc.MFDataset(self.fname))
      else:
        setattr(self, '_' + x, value)
    def delx(self):
      # close nco
      getattr(self,'_'+x).close()
      delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def subplots(self,x,doc):
    """subplot property"""
    def getx(self): return getattr(self, '_'+x)
    def setx(self, ax):
      # create subplots
      if isinstance(ax, tuple):
        if len(ax) == 3: maxplots = ax[2]
        else: maxplots = ax[0]*ax[1]
        subplots = np.array([])
        for i in xrange(0,len(self.vlst)*len(self.times)*len(self.levels),maxplots):
          fig, ax2 = plt.subplots(ax[0],ax[1], figsize=self.figsize)
          try:
            subplots = np.concatenate((subplots,ax2.ravel()[:maxplots]))
            for iax in xrange(maxplots, ax[0]*ax[1]): fig.delaxes(ax2.ravel()[iax])
          except AttributeError: subplots = np.concatenate((subplots,[ax2]))
          if i + maxplots > len(self.vlst)*len(self.times)*len(self.levels):
            for ax2 in subplots[len(self.vlst)*len(self.times)*len(self.levels):]:
              fig.delaxes(ax2)
          subplots = subplots[:len(self.vlst)*len(self.times)*len(self.levels)]
        self._subplot_nums = (range(1,maxplots+1)*int(np.ceil(len(self.vlst)*len(self.times)*len(self.levels)/float(maxplots))))[:len(self.vlst)*len(self.times)*len(self.levels)]
        self._subplot_shape = ax
      elif isinstance(ax, mpl.axes.SubplotBase):
        subplots = [ax]
        self._subplot_shape = None
        self._subplot_nums = [None]*len(subplots)
      else:
        subplots = np.ravel(ax)
        self._subplot_shape = None
        self._subplot_nums = [None]*len(subplots)
        
      setattr(self, '_' + x, subplots)
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def speed(self, x, doc):
    def getx(self):
      if hasattr(self, '_'+x): return getattr(self,'_'+x)
      # calculate speed
      if isinstance(self.u, np.ma.MaskedArray):
        speed = np.ma.power(self.u*self.u+self.v*self.v, 0.5)
        speed.mask = self.u.mask
        setattr(self, '_'+x, speed)
      else: setattr(self, '_'+x,np.power(self.u*self.u+self.v*self.v, 0.5))
      return getattr(self,'_'+x)
    def setx(self,value): setattr(self,'_'+x,value)
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def weights(self, x, doc):
    """property to calculate weights"""
    def getx(self):
      if hasattr(self, '_'+x): return getattr(self,'_'+x)
      try:
        from cdo import Cdo
        import glob
        cdo = Cdo()
        tmpfile='tmp_' + str(np.random.randint(0,100000)) + '.nc'
        cdo.gridweights(input=glob.glob(self.fname)[0], output=tmpfile)
        nco=nc.Dataset(tmpfile,mode='r')
        setattr(self, '_'+x, nco.variables['cell_weights'][:])
        nco.close()
        os.remove(tmpfile)
      except:
        setattr(self, '_'+x, None)
        print('Attention! Error in calculating weights via cdos. Ignoring weighted means causes error in reduced resolutions!')
      return getattr(self,'_'+x)
    def setx(self,value): setattr(self,'_'+x,value)
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
  def ncoprop(self,x,doc):
    def getx(self): return getattr(self,'_'+x)
    def setx(self, value):
      try: setattr(self, '_'+x, str(getattr(self.nco.variables[value],x)))
      except: setattr(self, '_'+x, str(value))
    def delx(self): delattr(self,'_'+x)
    return property(getx,setx,delx,doc)
    
class cbarmanager(object):
  """Class to control a single colorbar of multiple mapBase instances.
  It is not recommended to use the methods of this class but rather
  use the update_cbar method from the maps instance the cbarmanager
  instance belongs to.
  See function __init__ for a description of the properties
  """
  def __init__(self, maps, fig, cbar, fmt, mapsobj, _bounds = None, _cmap = None, _norm = None, wind = False):
    """initialize the class.
    Input:
     - maps: List of mapBase instances which are controlled by the
       cbarmanager
     - fig: list of figures in which the mapBase instances are
     - cbar: dictionary. Keys are positions of the colorbar ('b',
       'r','sh' and 'sv') and values are again dictionaries with 
       the figure of the colorbar instance as key and the colorbar
       instance as value.
     - fmt: a fmtBase instance to control the formatoptions
     - mapsobj: the maps instance the cbarmanager belongs to
     - _bounds are the bounds as calculated from the settings in fmt
     - _cmap is the colormap as defined from fmt
     - _norm is a normalization instance
     - wind is True, if the mapBase instances are windplot instances
       plotted over fieldplot instances
    """
    self.maps = maps
    self.fig = fig
    self.cbar = cbar
    self.fmt = fmt
    self.mapsobj = mapsobj
    self._bounds = _bounds
    self._cmap = _cmap
    self._norm = _norm
    self.wind = wind
  
  def _draw_colorbar(self, draw=True):
    """draws the colorbars specified in self.fmt.plotcbar"""
    names, times, levels, long_names, units = self.mapsobj.get_names(*self.maps)
    for cbarpos in self.fmt.plotcbar:
      if cbarpos in self.cbar:
        for fig in self.cbar[cbarpos]:
          plt.figure(fig.number)
          self.cbar[cbarpos][fig].set_cmap(self._cmap)
          self.cbar[cbarpos][fig].set_norm(self._norm)
          self.cbar[cbarpos][fig].draw_all()
          if fig not in self.fig: plt.draw()
      else:
        orientations = ['horizontal', 'vertical', 'horizontal', 'vertical']
        cbarlabels   = ['b','r','sh','sv']
        if cbarpos not in cbarlabels:
          try: sys.exit('Unknown position option ' + str(cbarpos) + '! Please use one of ' + ', '.join(label for label in cbarlabels) + '.')
          except KeyError: sys.exit('Unknown position option for the colorbar! Please use one of ' + ', '.join(label for label in cbarlabels) + '.')
        self.cbar[cbarpos] = {}
        orientation = orientations[cbarlabels.index(cbarpos)]
        if cbarpos in ['b','r']:
          for fig in self.fig:
            if cbarpos == 'b':
              fig.subplots_adjust(bottom=0.2)
              ax = fig.add_axes([0.125, 0.1, 0.775, 0.05])
            elif cbarpos == 'r':
              fig.subplots_adjust(right=0.8)
              ax = fig.add_axes([0.825, 0.25, 0.035, 0.6])
            self.cbar[cbarpos][fig] = mpl.colorbar.ColorbarBase(ax, cmap = self._cmap, norm = self._norm, orientation = orientation, extend = self.fmt.extend)
        elif cbarpos in ['sh', 'sv']:
          if cbarpos == 'sh':
            fig = plt.figure(figsize=(8,1))
            ax  = fig.add_axes([0.05, 0.5, 0.9, 0.3])
          elif cbarpos == 'sv':
            fig = plt.figure(figsize=(1,8))
            ax  = fig.add_axes([0.3, 0.05, 0.3, 0.9])
          fig.canvas.set_window_title('Colorbar, var ' + ','.join(name for name in names) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))
          self.cbar[cbarpos][fig] = mpl.colorbar.ColorbarBase(ax, cmap = self._cmap, norm = self._norm, orientation = orientation, extend = self.fmt.extend)
      for fig in self.cbar[cbarpos]:
        plt.figure(fig.number)
        if type(self.fmt.ticks) is int:
          if cbarpos in ['b', 'sh']: self.cbar[cbarpos][fig].set_ticks([float(text.get_text()) for text in self.cbar[cbarpos][fig].ax.get_xticklabels()[::self.fmt.ticks]])
          else: self.cbar[cbarpos][fig].set_ticks([float(text.get_text()) for text in self.cbar[cbarpos][fig].ax.get_yticklabels()[::self.fmt.ticks]])
        elif self.fmt.ticks is not None: self.cbar[cbarpos][fig].set_ticks(self.fmt.ticks)
        if self.fmt.ticklabels is not None: self.cbar[cbarpos][fig].set_ticklabels(self.fmt.ticklabels)
        self.cbar[cbarpos][fig].ax.tick_params(labelsize=self.fmt._tickops['fontsize'])
        if self.fmt.clabel is not None:
          self.cbar[cbarpos][fig].set_label(self.fmt.clabel.replace('<<<var>>>',', '.join([name for name in names])).replace('<<<time>>>',', '.join(str(time) for time in times)).replace('<<<level>>>',', '.join(str(level) for level in levels)).replace('<<<unit>>>', ', '.join(unit for unit in units)).replace('<<<longname>>>',', '.join(longname for longname in long_names)), **self.fmt._labelops)
        else:
          self.cbar[cbarpos][fig].set_label('')
        if draw: plt.draw()
  
  def _removecbar(self, positions = 'all'):
    """removes the colorbars with the defined positions (either 'b','r',
    'sh' or 'sv')"""
    if not hasattr(self,'cbar'): return
    if positions == 'all': positions = self.cbar.keys()
    for cbarpos in positions:
      if cbarpos in self.cbar:
        for fig in self.cbar[cbarpos]:
          if cbarpos in ['b','r'] and cbarpos in self.cbar:
            plt.figure(fig.number)
            fig.delaxes(self.cbar[cbarpos][fig].ax)
            if cbarpos == 'b': fig.subplots_adjust(bottom=0.1)
            else:              fig.subplots_adjust(right=0.9)
            plt.draw()
          elif cbarpos in ['sh', 'sv'] and cbarpos in self.cbar:
            plt.close(fig)
        del self.cbar[cbarpos]
    return

  def _moviedata(self, times, **kwargs):
    """Returns a generator for the movie with the formatoptions"""
    for time in times:
      yield {key:value[times.index(time)] for key, value in kwargs.items()}
    
  def _runmovie(self, fmt):
    """Run function for the movie suitable with generator _moviedata"""
    self.fmt.update(**{key:value for key,value in fmt.items() if key not in ['times','vlst','level']})
    self._draw_colorbar(draw=False)

class maps(object):
  """
  Creates an object containing mapBase instances for given variables, times and
  levels. For initialization keywords see initialization function __init__ below.
  To change attributes, please use the update function.
  Methods are:
    - get_maps:     returns a list of all mapBase instances contained in the maps instance
    - get_figs:     returns a dictionary with figs as keys and the corresponding mapBase
                    instance
    - get_winds:    if not windonly: Returns the windinstances of the fieldplot instances
    - get_cbars:    returns a list of cbars of the specified dimensions
    - get_names:    returns names, times, etc. of the given instances
    - output:       Saves the specified figures to a file
    - update:       updates the maps from a dictionary or given keywords
    - update_cbar:  updates the cbars if any cbars are drawn to handle multiple mapBase
                    instances
    - nextt:        updates all maps to their next timestep
    - previoust:    updates all maps to the previous timestep
    - reset:        reinitializes the maps instance and makes a new plot (probably your best
                    friend especially after playing around with colorbars)
    - show:         shows all figures
    - redo:         redo the changes made
    - undo:         undo the changes made
    - make_movie:   makes a movie of the specified figures and saves it
    - asdict:       returns the current formatoptions of the maps instance as dictionary
    - script:       creates a file for initialization of the maps instance
    - removecbars:  removes the specified colorbars if any cbars are drawn to handle multiple
                    mapBase instances and allows the manipulation of the bounds of the
                    separate mapBase instances
    - close:        closes the maps instance and all corresponding mapBase instances and 
                    figures
  See below for a more detailed description of each method.
  """
  
  props = mapproperties() # container containing methods for property definition
  # ------------------ define properties here -----------------------
  
  # mapdata dictionary property
  maps           = props.mapdata('maps', """dictionary containing mapobj objects of all plots sorted after variable, time and level (i.e. maps[var][time][level])""")
  fname          = props.default('fname', """Name of the nc-file""")
  vlst           = props.default('vlst', """List of variables""")
  times          = props.default('times', """List of time steps""")
  levels         = props.default('levels', """List of levels""")
  subplots       = props.subplots('subplots', """List of subplots""")
  nco            = props.nco('nco', """netCDF4.MFDataset instance of ncfile""")
  
  
  def __init__(self, ncfile, vlst = 'all',  times = 0, levels = 0, ax = (1,1), sort = 'vtl', formatoptions = None, timenames = ['time'], levelnames = ['level', 'lvl', 'lev'], lon=['lon', 'longitude', 'x'], lat=['lat', 'latitude', 'y'], windonly=False, onecbar = False, u=None, v=None, figsize = None):
    """
    Input:
      - ncfile: string or 1D-array of strings. Path to the netCDF-file containing the
        data for all variables. Filenames may contain wildcards (*, ?, etc.) as suitable
        with the Python glob module (the netCDF4.MFDataset is used to open the nc-file).
        You can even give the same netCDF file multiple times, e.g. for making one figure
        with plots for one variable, time and level but different regions.
      - vlst: string or 1D-array of strings. List containing all variables which
        shall be plotted or only one variable. The given strings names must correspond to
        the names in <ncfile>. If 'all', all variables which are not declared as dimensions
        are used. If windonly is True, this name will be used for the wind variable
      - u: string (Default: None). Name of the zonal wind variable if a windplot
        shall be visualized
      - v: string (Default: None). Name of the meridional wind variable if a windplot shall
        be visualized
      - times: integer or list of integers. Timesteps which shall be plotted
      - levels: integer or list of integers. Levels which shall be plotted
      - ax: matplotlib.axes.AxesSubplot instance or list of matplotlib.axes.AxesSubplot
        instances or tuple (x,y[,z]) (Default: (1,1)). If ax is an axes instance (e.g.
        plt.subplot()) or a list of axes instances, the data will be plotted into these axes.
        If ax is a tuple (x,y), figures will be created with x rows and y columns of subplots.
        If ax is (x,y,z), only the first z subplots of each figure will be used.
      - figsize: Tuple (x,y), (Default: None, i.e. (8,6)). Size of the figure (does not have
        an effect if ax is a subplot or a list of subplots).
      - windonly: Do not print an underlying field but only u and v on the maps.
      - sort: string, combination of 't', 'v' and 'l' (Default: 'vtl'). Gives the order of
        sorting the maps on the figures (i.e. sort = 'vtl' will first sort for variables, then
        time then level).
      - timenames: 1D-array of strings: Gives the name of the time-dimension for which will be
        searched in the netCDF file
      - levelnames: 1D-array of strings: Gives the name of the fourth dimension (e.g vertical levels)
        for which will be searched in the netCDF file
      - lon: 1D-array of strings: Gives the name of the longitude-dimension for which will be
        searched in the netCDF file
      - lat: 1D-array of strings: Gives the name of the latitude-dimension for which will be
        searched in the netCDF file
      - onecbar: boolean, dictionary or tuple of dictionaries (Default: False). If True, one
        colorbar will be drawn for each figure. If dictionary: the syntax is as follows:
        onecbar = {['vlst':...][, 'times':...][, 'levels':...][, 'formatoption keyword':...]}
        where [] indicate optional arguments. 'vlst', 'times' and 'levels' may be a list of the
        variables, times or levels respectively (If not set: Use all variables, times, etc. in the
        maps object) and 'formatoption keyword' may be any regular key word of the formatoptions
        controlling the colorbar ('cmap', 'bounds', 'clabel', 'plotcbar', etc.). To update those
        colorbars. Use the update_cbar function.
      - formatoptions: dictionary (Default: None). Dictionary controlling the format of the plots.
        Syntax is as follows:
        formatoptions = {['<<<var>>>':{
                                       ['t<<<time>>>':{
                                                       ['l<<<level>>>':{'keyword':..., ...}]
                                                       [, 'keyword':...,...]
                                                       }]
                                       [, 'l<<<level>>>::{'keyword':..., ...}]
                                       [, 'keyword':..., ...]
                                       }]
                         [, 't<<<time>>>':{
                                         ['l<<<level>>>':{'keyword':..., ...}]
                                         [, 'keyword':...,...]
                                         }]
                         [, 'l<<<level>>>::{'keyword':..., ...}]
                         [, 'keyword':..., ...]
                         }.
        Seems complicated, but in fact rather simple considering the following rules:
          -- Formatoptions are set via 'keyword':value (for possible keywords, see below).
          -- Time and level specific keywords are put into a dictionary indicated by the key
             't<<<time>>>' or 'l<<<level>>>' respectively (where <<<time>>> and <<<level>>>
             is the number of the time, and or level).
          -- To set default formatoptions for each map: set the keyword in the upper most hierarchical
             level of formatoptions (e.g. formatoptions = {'plotcbar':'r'}).
          -- To set default formatoptions for each variable, times or level separately set the keyword
             in the second hierarchical level of formatoptions (e.g. formatoptions = {'t4':{'plotcbar:'r'}}
             will only change the formatoptions of maps with time equal to 4,
             formatoptions = {'l4':{'plotcbar:'r'}} will only change formatoptions of maps with level
             equal to 4).
          -- To set default options for a specific variable and time, but all levels: put them in the 3rd
             hierarchical level of formatoptions (e.g. formatoptions = {<<<var>>>:{'t4':{'plotcbar':'r'}}}
             will only change the formatoptions of each level corresponding to variable <<<var>>> and
             time 4). Works the same for setting default options for specific variable and level, but all
             times.
          -- To set a specific key for one map, just set
             formatopions = {<<<var>>>:{'t<<<time>>>':{'l<<<level>>>':{'plotcbar:'r', ...}}}}.
        
        The formatoption keywords are:
      """
    # docstring is extended below
    global currentmap
    global openmaps
    currentmap = self
    openmaps = openmaps + [self]
    
    self.maps = {}
    self._cbars = []
    self.fname = ncfile
    self.nco = nc.MFDataset(ncfile)
    self.lonnames  = lon
    self.latnames  = lat
    self.timenames = timenames
    self.levelnames= levelnames
    self._dims      = {'lon':lon, 'lat':lat, 'time':timenames, 'level':levelnames}
    if vlst == 'all' and not windonly:
      self.vlst = [str(key) for key in self.nco.variables.keys() if key not in lon+lat+timenames+levelnames]
    else:
      if isinstance(vlst, str):   vlst = [vlst]
      self.vlst = vlst
    if isinstance(times, int):  times = [times]
    if isinstance(levels, int): levels = [levels]
    self.levels = levels
    self.times = times
    self.figsize = figsize
    self.subplots = ax
    self.sort = sort
    self.u  = u
    self.v  = v
    self.windonly = windonly
    self._initcompatible = False # do not add maps compatible to initialization but according to the times, vars and levels as specified in the formatoptions
    self._setupfigs(self._setupfmt(formatoptions))
    
    print("Setting up projections...")
    for mapo in self.get_maps(): mapo._setupproj()
    
    print("Making plots...")
    self.make_plot()
    
    for fig in self.get_figs():
      names, times, levels, long_names, units = self.get_names(fig)
      fig.canvas.set_window_title('Figure ' + str(fig.number) + ': Variable ' + ','.join(name for name in names) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))
       
    if onecbar is not False:
      if onecbar is True: self.update_cbar(*(dict(zip(['vlst','times','levels'], self.get_names(fig)[:3])) for fig in self.get_figs()), add = False, delete = False)
      elif isinstance(onecbar, dict): self.update_cbar(onecbar, add=False, delete = False)
      else: self.update_cbar(*onecbar, add = False, delete = False)
    
    # old fmts (for function undo)
    self._fmt = [self.asdict('maps','cbars',initcompatible=True, reduced=False)]
    
    # future fmts (for function redo)
    self._newfmt = []
    
  def make_plot(self):
    """makes the plot of all mapBase instances. Don't use this function but rather 
    the update function to make plots"""
    for mapo in self.get_maps(): mapo.make_plot()
  
  def get_maps(self, vlst='all', times='all', levels='all'):
    """Returns 1D-numpy array containing the mapBase instances stored in maps.
       Input:
         - vlst: string or 1D array of strings (Default: all). If not 'all', the
           strings need to be the name of a variable contained in the maps 
           instance
         - times: same as vlst but for times (as integers!)
         - levels: same as vlst but for levels (as integers!)
       Output:
         - list of mapBase instances
    """
    if vlst == 'all':   vlst = sorted(self.maps.keys())
    else:
      if isinstance(vlst, str):   vlst = [vlst]
    if times == 'all':  times = {var:sorted(self.maps[var].keys()) for var in vlst}
    else:               
      if isinstance(times, int):  times = [times]
      times = {var:['t' + str(time) for time in times] for var in vlst}
    if levels == 'all': levels = {var:{t:sorted(self.maps[var][t].keys()) for t in times[var]} for var in vlst}
    else:               
      if isinstance(levels, int): levels = [levels]
      levels = {var:{t:['l' + str(level) for level in levels] for t in times[var]} for var in vlst}
    maps = []
    for var in vlst:
      for t in times[var]:
        for l in levels[var][t]: maps.append(self.maps[var][t][l])
    return maps
  
  def get_winds(self, **kwargs):
    """Returns 1D-numpy array containing the windplot instances stored in maps
    if windonly is not True (in this case: use get_maps()).
    Keyword arguments are determined by function get_maps (i.e. vlst, times and levels)."""
    if not self.windonly: return [mapo.wind for mapo in self.get_maps(**kwargs) if mapo.wind is not None]
    else: return []
  
  def get_figs(self, *args, **kwargs):
    """Returns dictionary containing the figures used in the maps instance
    as keys and a list with the included mapBase instances as value.
    Without any kwargs and args, return all figures from the maps instance.
    Otherwise you can either give mapBase objects or figures as arguments
    or specify one or each of the following key words to return all figures
    related to the specified variable, time or level
     - vlst:   string or list of strings. Specify variables to return
     - times:  integer or list of integers. Specify times to return
     - levels: integer or list of integers. Specify levels to return
     Example: self.get_figs(vlst='temperature') will return a dictionary with
     all figures that have mapBase instances with the variable 'temperature' as
     subplot as keys and a list with the corresponding mapBase instances as values.
     If 'wind' in args: the mapBase instances will be the corresponding wind-
    plot instances to the figure.
    """
    if 'wind' in args: get_func = self.get_winds; args = tuple(arg for arg in args if arg != 'wind')
    else: get_func = self.get_maps
    if args == (): maps = get_func(**kwargs); figs = {}; append = True
    elif all(isinstance(arg, mapBase) for arg in args): maps = args; figs = {}; append = True
    elif all(isinstance(arg, mpl.figure.Figure) for arg in args): figs = {arg:[] for arg in args}; maps = get_func(); append = False
    else: print args, kwargs; sys.exit("Wrong type of obj! Object must either be 'maps' or 'winds'!")
    for mapo in maps:
      if mapo.ax.get_figure() not in figs and append: figs[mapo.ax.get_figure()] = []
      if mapo.ax.get_figure() in figs: figs[mapo.ax.get_figure()].append(mapo)
    return figs
  
  def output(self, output, *args, **kwargs):
    """Saves the figures.
    Just setting output = 'filename.pdf' will save all figures of the maps object to filename.pdf
    
    Further input options:
      - output: string or 1D-array of strings. If string: <<<var>>>,
        <<<time>>>, <<<level>>>, <<<long_name>>>, <<<unit>>> will be
        replaced by the attributes contained in the figures.
      
      Arguments:
      - Either figures or mapBase instances which shall be saved (in
        case of mapBase, the corresponding figure will be saved)
      - 'tight' making the bbox_inches of the plot tight, i.e. reduce
        the output to the plot margins
      
      Keyword arguments:
      
      - vlst: To save only the figures with variables specified in vlst
      - times: To save only the figures with times specified in times
      - levels: To save only the figures with levels specified in levels
      - any other keyword as specified in the pyplot.savefig function.
      These are:
    """
    # the docstring is extended by the plt.savefig docstring below
    from matplotlib.backends.backend_pdf import PdfPages
    saveops = {key:value for key, value in kwargs.items() if key not in ['vlst','times','level']}
    if 'tight' in args: saveops['bbox_inches'] = 'tight'; args = tuple([arg for arg in args if arg != 'tight'])
    kwargs = {key:value for key, value in kwargs.items() if key in ['vlst','times','level']}
    if args == ():
      figs = self.get_figs(**kwargs).keys()
    elif isinstance(args[0], mapBase):
      names, times, levels, long_names, units = self.get_names(*args)
      figs = self.get_figs(vlst=names, times=times, levels=levels)
    else:
      figs = args
    if isinstance(output, str):
      if output[-4:] in ['.pdf', '.PDF']:
        names, times, levels, long_names, units = self.get_names(*figs)
        output = output.replace('<<<var>>>','-'.join([name for name in names])).replace('<<<time>>>','-'.join(str(time) for time in times)).replace('<<<level>>>','-'.join(str(level) for level in levels)).replace('<<<unit>>>','-'.join(unit for unit in units)).replace('<<<longname>>>','-'.join(longname for longname in long_names))
        with PdfPages(output) as pdf:
          for fig in figs: pdf.savefig(fig, **saveops)
          print('Saving plot to ' + output)
        return
      else:
        strout = output
        output = []
        for fig in figs:
          names, times, levels, long_names, units = self.get_names(fig)
          output.append(strout.replace('<<<var>>>','-'.join([name for name in names])).replace('<<<time>>>','-'.join(str(time) for time in times)).replace('<<<level>>>','-'.join(str(level) for level in levels)).replace('<<<unit>>>','-'.join(unit for unit in units)).replace('<<<longname>>>','-'.join(longname for longname in long_names)))
    else: pass
    # test output
    try:
      if len(np.shape(output)) > 1: sys.exit('Output array must be a 1D-array!')
      if len(figs) != len(output): sys.exit('Length of output names (' + str(len(output)) + ') does not fit to the number of figures (' + str(len(figs)) + ').')
    except TypeError:
      sys.exit('Output names must be either a string or an 1D-array of strings!')
    for fig in figs:
      fig.savefig(output[figs.index(fig)], **saveops)
      print('Plot saved to ' + output[figs.index(fig)])
    return
  
  def update(self, fmt={}, add = True, delete = True, todefault = False, **kwargs):
    """Function to update the mapBase objects.
    Input:
      - fmt: dictionary (the same shape and options linke in the initialization function
      __init__).
      - add: Boolean (Default: True). Adds the new formatoptions to old formatoptions
        allowing a undoing via the undo function
      - delete: Boolean (Default: True). Deletes the newer formatoptions if created by
        function undo for the redo function.
      - todefault: Boolean (Default: False). Sets all formatoptions which are not speci-
        fied by fmt or kwargs to default.
      Additional keyword arguments may be any valid formatoption keyword.
    """
    from copy import deepcopy
    fmt = deepcopy(fmt) # if not deepcopied, the update in the next line will use previous fmts given to the update function
    fmt.update({key:value for key, value in kwargs.items() if key not in ['vlst', 'times','levels']})
    fmt = self._setupfmt(fmt)
    maps = self.get_maps(**{key: value for key, value in kwargs.items() if key in ['vlst', 'times','levels']})
    # handle initcompatible state
    if self._initcompatible: ending = 'orig'
    else: ending = ''
    # update maps
    for mapo in maps: mapo.update(todefault=todefault, **fmt[getattr(mapo, 'name'+ending)]['t'+str(getattr(mapo, 'time'+ending))]['l'+str(getattr(mapo, 'level'+ending))])
    # update figure window title and draw
    for cbar in self.get_cbars(*maps): cbar._draw_colorbar()
    for fig in self.get_figs(*maps):
      plt.figure(fig.number)
      names, times, levels, long_names, units = self.get_names(fig)
      fig.canvas.set_window_title('Figure ' + str(fig.number) + ': Variable ' + ','.join(name for name in names) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))
      plt.draw() # if it is part of a cbar, it has already been drawn above
    # reorder
    self._reorder()
    # add to old fmts
    if add: self._fmt.append(self.asdict('maps','cbars',initcompatible=True, reduced=False))
    # delete new fmts
    if delete: self._newfmt = []
    del fmt
  
  def nextt(self,*args,**kwargs):
    """takes the next time step for maps specified by args and kwargs
    (same syntax as get_maps. Use 'wind' as an argument if only winds
    shall be updated."""
    if 'wind' in args: maps = self.get_winds(*(arg for arg in args if arg != 'wind'), **kwargs)
    else: maps = self.get_maps(*args, **kwargs)
    for mapo in maps: mapo.update(time = mapo.time + 1)
    self._reorder()
    for fig in self.get_figs(*maps): plt.figure(fig.number); plt.draw()
  
  def prevt(self,*args,**kwargs):
    """takes the previous time step for maps specified by args and kwargs
    (same syntax as get_maps. Use 'wind' as an argument if only winds shall
    be updated"""
    if 'wind' in args: maps = self.get_winds(*(arg for arg in args if arg != 'wind'), **kwargs)
    else: maps = self.get_maps(*args, **kwargs)
    for mapo in maps: mapo.update(time = mapo.time - 1)
    self._reorder()
    for fig in self.get_figs(*maps): plt.figure(fig.number); plt.draw()
  
  def _reorder(self):
    """reorder self.maps after changes in variables, times or levels"""
    mapos = self.get_maps()
    self.maps = {}
    for mapo in mapos: self.maps = mapo
  
  def reset(self, num=0, fromscratch = False, ax=None, sort=None, figsize=None):
    """Reinitializes the maps object with the specified settings.
    Works even if undo function fails.
    Input:
      - num: Number of formatoptions (like undo function). 0 is cur-
        rent, -1 the one before (often the last one working), etc.
      - fromscratch: Boolean. If False, only figures will be closed
        and recreated (if ax is not None) or the axes will be reset 
        if ax is None. If True the whole maps instance will be closed
        and reopend by loading the data from the nc-file (use this
        option if you accidently closed the maps instance or for example
        if you accidently set the wrong variables, times or levels and
        undo function failed.)
      - ax, sort, figsize: Like in initialization function __init__:
        Specify the subplot, sort and figsize setting (if None: the
        current settings will be used and if ax is None, no new figures
        will be created).
      """
    if self._fmt == []: sys.exit('Impossible option')
    if num > 0 and num >= len(self._fmt)-1: sys.exit('Too high number! Maximal number is ' + str(len(self._fmt)-1))
    elif num < 0 and num < -len(self._fmt): sys.exit('Too small number! Minimal number is ' + str(-len(self._fmt)+1))
    self._initcompatible = True
    self._reorder()
    if sort is not None: self.sort = sort
    if figsize is not None: self.figsize = figsize
    # reset cbars
    self.removecbars()
    self._cbars = []
    # close the maps instance
    if fromscratch:
      if ax is None: ax = self._subplot_shape
      if ax is None: ax = (1,1)
      try:
        self.close(vlst=self.vlst)
      except AttributeError: print("Could not close the figures but anyway will draw new figures")
      except KeyError: print("Could not close the figures but anyway will draw new figures")
      if not hasattr(self, 'nco'): self.nco = None
    elif ax is not None:
      try:
        self.close('figure')
      except AttributeError: print("Could not close the figures but anyway will draw new figures")
      except KeyError: print("Could not close the figures but anyway will draw new figures")
    # set new subplots
    if ax is not None: self.subplots = ax
    else:
      for ax in self.subplots: ax.clear
    # set new figures
    self._setupfigs(self._fmt[num-1][0], fromscratch=fromscratch)
    
    print("Setting up projections...")
    for mapo in self.get_maps(): mapo._setupproj()
    
    print("Making plots...")
    self.make_plot()
    
    for fig in self.get_figs():
      plt.figure(fig.number)
      names, times, levels, long_names, units = self.get_names(fig)
      fig.canvas.set_window_title('Figure ' + str(fig.number) + ': Variable ' + ','.join(name for name in names) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))
      plt.draw()
    
    self._initcompatible = False
    self._reorder()
    if self._fmt[num-1][1] != []: self.update_cbar(*self._fmt[num-1][1], add = False, delete = False)
    # shift to new fmt
    if num != 0:
      self._newfmt = self._fmt[num:] + self._newfmt
      if num < 0: self._fmt.__delslice__(len(self._fmt)+num, len(self._fmt))
      else: self._fmt.__delslice__(num,len(self._fmt))
    
  def undo(self, num=-1):
    """Undo the changes made. num gives the number of changes to go back."""
    if self._fmt == [] or len(self._fmt) == 1: sys.exit('Impossible option')
    if num > 0 and num >= len(self._fmt)-1: sys.exit('Too high number! Maximal number is ' + str(len(self._fmt)-1))
    elif num < 0 and num < -len(self._fmt): sys.exit('Too small number! Minimal number is ' + str(-len(self._fmt)+1))
    self._initcompatible = True
    self._reorder()
    if self._fmt[num-1][1] == []: self.removecbars()
    self.update(self._fmt[num-1][0], add=False, delete=False, todefault = True)
    self._initcompatible = False
    self._reorder()
    if self._fmt[num-1][1] != []: self.update_cbar(*self._fmt[num-1][1], add=False, delete=False, todefault = True)
    # shift to new fmt
    self._newfmt = self._fmt[num:] + self._newfmt
    if num <= 0: self._fmt.__delslice__(len(self._fmt)+num, len(self._fmt))
    else: self._fmt.__delslice__(num,len(self._fmt))
    
  def redo(self, num=1):
    """Redo the changes made. num gives the number of changes to use."""
    if self._newfmt == []: sys.exit('Impossible option')
    if num > 0 and num > len(self._newfmt): sys.exit('Too high number! Maximal number is ' + str(len(self._newfmt)))
    elif num < 0 and num < -len(self._newfmt): sys.exit('Too small number! Minimal number is ' + str(-len(self._newfmt)-1))
    self._initcompatible = True
    self._reorder()
    if self._newfmt[num][1] == []: self.removecbars()
    self.update(self._newfmt[num][0], add=False, delete=False, todefault = True)
    self._initcompatible = False
    self._reorder()
    if self._newfmt[num][1] != []: self.update_cbar(*self._newfmt[num][1], add=False, delete=False, todefault = True)
    # shift to old fmt
    self._fmt = self._fmt + self._newfmt[:num]
    if num > 0: self._newfmt.__delslice__(0,num)
    else: self._newfmt.__delslice__(0,len(self._newfmt)+num)
  
  def show(self):
    """shows all open figures (without blocking)"""
    plt.show(block=False)
  
  def close(self,*args,**kwargs):
    """Without any args and kwargs, close all open figure from the maps object,
    delete all mapBase objects and the close the netCDF4.MFDataset is closed.
    Otherwise you can give the following arguments:
      - 'data': delete all mapBase instances and (without any additional keywords)
         close the netCDF4.MFDataset
      - 'figure': Close the figures specified by kwargs
    You can further specify, which mapBase instances to close. Possible keywords are
     - vlst:   string or list of strings. Specify variables to close
     - times:  integer or list of integers. Specify times to close
     - levels: integer or list of integers. Specify levels to close
    """
    if any(arg not in ['data','figure'] for arg in args):
      sys.exit('Unknown argument ' + ', '.join(arg for arg in args if arg not in ['data','figure']) + ". Possibilities are 'data' and 'figure'.")
    if self.maps == {}: return
    if 'data' in args or args is ():
      for mapo in self.get_maps(**kwargs):
        mapo.close('data')
    if 'figure' in args or args is ():
      for mapo in self.get_maps(**kwargs):
        mapo._removecbar(['sh','sv'])
        if isinstance(mapo, fieldplot) and mapo.wind is not None: mapo.wind._removecbar(['sh','sv'])
      for fig in self.get_figs(**kwargs).keys(): plt.close(fig)
      for cbar in self._cbars:
        for cbarpos in cbar.cbar:
          for fig in cbar.cbar[cbarpos]:
            plt.close(fig)
    if args == ():
      if self._initcompatible: ending = 'orig'
      else: ending = ''
      for mapo in self.get_maps(**kwargs):
        self.maps[getattr(mapo,'name'+ending)]['t'+str(getattr(mapo,'time'+ending))].pop('l'+str(getattr(mapo,'level'+ending)))
        if self.maps[getattr(mapo,'name'+ending)]['t'+str(getattr(mapo,'time'+ending))] == {}: self.maps[getattr(mapo,'name'+ending)].pop('t'+str(getattr(mapo,'time'+ending)))
        if self.maps[getattr(mapo,'name'+ending)] == {}: self.maps.pop(getattr(mapo,'name'+ending))
    if kwargs == {} and ('data' in args or args == ()): del self.nco
    
  def update_cbar(self,*args, **kwargs):
    """Update or create a cbar.
    Arguments are dictionaries
        onecbar = {['vlst':...][, 'times':...][, 'levels':...][, 'formatoption keyword':...]}
    where [] indicate optional arguments. 'vlst', 'times' and 'levels' may be a list of the
    variables, times or levels respectively (If not set: Use all variables, times, etc. in the
    maps object) and 'formatoption keyword' may be any regular key word of the formatoptions
    controlling the colorbar ('cmap', 'bounds', 'clabel', 'plotcbar', etc.).
    Keyword arguments (kwargs) may also be formatoption keywords or out of vlst, times and levels.
    They will then be treated like a single dictionary (this is just to avoid nasty typing of :,
    {}, etc.). Further keyword arguments may be
      - add: Boolean (Default: True). Adds the new formatoptions to old formatoptions
        allowing a undoing via the undo function
      - delete: Boolean (Default: True). Deletes the newer formatoptions if created by
        function undo for the redo function.
      - todefault: Boolean (Default: False). Sets all formatoptions which are not speci-
        fied by fmt or kwargs to default.
    If no colorbar with any of the specified dimensions is found, a new colorbar manager object is
    created. For security reasons: There is no possibility to add new dimensions or variables to an
    existing colorbar. To do so, remove the colorbars with the removecbar function and make a new one with this
    function.
    """
    add       = kwargs.get('add', True)
    delete    = kwargs.get('delete', True)
    todefault = kwargs.get('todefault', False)
    kwargs = {key:value for key,value in kwargs.items() if key not in ['add','delete','todefault']}
    if kwargs != {}: args = list(args) + [kwargs]
    
    # first set colorbars
    for cbarops in args:
      if 'windplot' in cbarops: args = tuple(['wind']); cbarops.update(cbarops.pop('windplot')); wind=True; get_func = self.get_winds
      else: args = (); wind = False; get_func=self.get_maps
      dims = {key:cbarops.get(key, 'all') for key in ['vlst','levels','times']}
      # if no colorbars are set up to now and no specific var, time and level options are set, make colorbars for each figure
      if all(value == 'all' for key, value in dims.items()) and self._cbars == []:
        figs = self.get_figs(*args)
        for fig in figs:
          self._cbars.append(cbarmanager(maps=figs[fig], fig=[fig], cbar={}, fmt=fmtBase(**{key:value for key,value in cbarops.items() if key not in ['times','vlst','levels']}), mapsobj = self, wind=wind))
      # now update colorbar objects or create them if they are not existent
      cbars = self.get_cbars(*args,**dims)
      if cbars == []:
        self._cbars.append(cbarmanager(maps=get_func(**dims),fig = self.get_figs(*args, **dims).keys(), cbar={}, fmt=fmtBase(**{key:value for key,value in cbarops.items() if key not in dims.keys()}), mapsobj = self, wind=wind))
        cbars = [self._cbars[-1]]
      
      # now draw and update colorbars    
      for cbar in cbars:
        # delete colorbars
        if not todefault: cbarops = {key:value for key,value in cbarops.items() if key not in ['times','vlst','levels']}
        else:
          cbarops = {key:cbarops.get(key, value) for key, value in cbar.fmt._default.items() if (key not in cbarops and np.all(getattr(cbar.fmt, key) != cbar.fmt._default[key])) or (key in cbarops and np.all(cbarops[key] != getattr(cbar.fmt,key)))}
        if 'plotcbar' in cbarops:
          if cbarops['plotcbar'] in [False, None]: cbarops['plotcbar'] = ''
          if cbarops['plotcbar'] == True: cbarops['plotcbar'] = 'b'
          cbar._removecbar([cbarpos for cbarpos in cbar.fmt.plotcbar if cbarpos not in cbarops['plotcbar']])
        cbar.fmt.update(**cbarops)
        if cbar.fmt.bounds[0] in ['rounded', 'sym', 'minmax', 'roundedsym'] and len(cbar.fmt.bounds) == 2: cbar._bounds = returnbounds(map(lambda x: (np.min(x), np.max(x)), (mapo.data for mapo in cbar.maps)), cbar.fmt.bounds)
        elif cbar.fmt.bounds[0] in ['rounded', 'sym', 'minmax', 'roundedsym'] and len(cbar.fmt.bounds) == 3: cbar._bounds = returnbounds(np.ma.concatenate(tuple(mapo.data for mapo in cbar.maps)), cbar.fmt.bounds)
        else: cbar._bounds = cbar.fmt.bounds
        cbar._cmap   = get_cmap(cbar.fmt.cmap, N=len(cbar._bounds)-1)
        cbar._norm   = mpl.colors.BoundaryNorm(cbar._bounds, cbar._cmap.N)
        for mapo in cbar.maps: mapo.fmt._enablebounds = False; mapo.fmt._bounds = cbar._bounds; mapo.fmt._cmap = cbar._cmap; mapo.make_plot()
        cbar._draw_colorbar()
        for fig in cbar.fig: plt.figure(fig.number); plt.draw()
        
    if add: self._fmt.append(self.asdict('maps','cbars',initcompatible=True, reduced=False))
    if delete: self._newfmt = []
  
  def get_cbars(self,*args,**kwargs):
    """Function to return the cbarmanager related to the given input
    Input:
      - Arguments (args) may be instances of mapBase, figures or
        cbarmanagers
      - Keyword arguments may be the one defined by get_maps
        (vlst, times, levels).
    Output:
      list of cbarmanager instances"""
    maps = []
    args = list(args)
    cbars = [cbar for cbar in args if isinstance(cbar, cbarmanager)]
    if args == []: maps = self.get_maps(**kwargs)
    elif args == ['wind']: maps = self.get_winds(**kwargs)
    #elif all(isinstance(arg, mpl.figure.Figure) for arg in args if arg != 'wind'): maps = [mapo for fig, mapo in self.get_figs(*args).items()]
    else:
      figs = self.get_figs(*(arg for arg in args if arg == 'wind'))
      for fig in figs:
        if fig in args: maps = maps + figs[fig]; args.remove(fig)
      maps = maps + list(arg for arg in args if not isinstance(arg,cbarmanager))
    cbars = cbars + [cbar for cbar in self._cbars if any(mapo in cbar.maps for mapo in maps)]
    return cbars
  
  def removecbars(self,*args, **kwargs):
    """Function to remove cbarmanager instances from the plot.
    args and kwargs are determined by the get_cbars function, i.e.
    Input:
      - Arguments (args) may be instances of mapBase, figures or
        cbarmanagers
      - Keyword arguments may be the one defined by get_maps
        (vlst, times, levels).
    """
    cbars = self.get_cbars(*args,**kwargs)
    for cbar in cbars:
      for mapo in cbar.maps: mapo.fmt._enablebounds = True
      cbar._removecbar()
      self._cbars.pop(self._cbars.index(cbar))
  
  def get_names(self,*args):
    """Function to return the descriptions of the specific input
    Input:
      - Arguments (args) may be instances of mapBase, figures or
        cbarmanagers and
    Output: names, times, levels, long_names, units
      - names:      list of variables contained in the input (without duplicates)
      - times:      list of times contained in the input (without duplicates)
      - levels:     list of levels contained in the input (without duplicates)
      - long_names: list of long_names of the variables in names (without duplicates)
      - units:      list of units of the variables in names (without duplicates)
    return names, times, levels, long_names and units of the given mapobjects or the mapobjects in the given figures without duplicates"""
    if args == (): return None
    else:
      args = list(args)
      maps = []
      figs = self.get_figs()
      for fig in figs:
        if fig in args: maps = maps + figs[fig]; args.remove(fig)
      for cbar in self.get_cbars('wind') + self.get_cbars():
        if cbar in args: maps = maps + cbar.maps; args.remove(cbar)
      maps = maps + args
      attrs = ['name','time','level', 'long_name', 'units']
      tmpout = [[getattr(mapo, attr) for mapo in maps] for attr in attrs]
      out = [[] for count in xrange(len(tmpout))]
      for iattr in xrange(len(tmpout)):
        for attr in tmpout[iattr]:
          if attr not in out[iattr]: out[iattr].append(attr)
      return out
  
  def _setupsortlist(self):
    sortlist = list(self.sort)
    sortlist[sortlist.index('v')] = [var for var in self.vlst]
    sortlist[sortlist.index('t')] = [time for time in self.times]
    sortlist[sortlist.index('l')] = [level for level in self.levels]
    return sortlist
  
  def _setupfigs(self, fmt, fromscratch=True):
    """set up the figures and map objects of the maps object"""
    vlst     = self.vlst
    times    = self.times
    levels   = self.levels
    windonly = self.windonly
    u        = self.u
    v        = self.v
    
    if windonly: mapo = windplot
    else: mapo = fieldplot
    # set up sorting informations
    sortlist = self._setupsortlist()
    if len(self.subplots) != len(vlst)*len(times)*len(levels): sys.exit('Length of given axis (' + str(len(self.subplots)) + ') does not fit to number of variables, times and lengths (' + str(len(vlst)*len(times)*len(levels)) + ')!')
    # setup axes
    isubplot = 0
    for val0 in sortlist[0]:
      for val1 in sortlist[1]:
        for val2 in sortlist[2]:
          var = eval('val'+str(self.sort.index('v')))
          time = eval('val'+str(self.sort.index('t')))
          level = eval('val'+str(self.sort.index('l')))
          if fromscratch: self.maps = mapo(self.fname, var=str(var), time=time, level=level, ax=self.subplots[isubplot], fmt=fmt[var]['t'+str(time)]['l'+str(level)], nco = self.nco, timenames = self.timenames, levelnames = self.levelnames, lon=self.lonnames, lat=self.latnames, ax_shapes=self._subplot_shape, ax_num=self._subplot_nums[isubplot], mapsobj = self, u = u, v = v)
          else:
            if not self._initcompatible: sys.exit('Initcompatible must be set to True!')
            mapo = self.maps[var]['t'+str(time)]['l'+str(level)]
            mapo.ax = self.subplots[isubplot]
            if hasattr(mapo, 'cbar'): del mapo.cbar
            if hasattr(mapo,'wind') and mapo.wind is not None: mapo.wind._removeplot()
            mapo._subplot_shape=self._subplot_shape
            mapo._ax_num=self._subplot_nums[isubplot]
            mapo.update(plot=False, todefault = True, **fmt[var]['t'+str(time)]['l'+str(level)])
          isubplot+=1
    return  
  
  def make_movie(self, output, fmt={}, cbarfmt = {}, steps = 'all', *args, **kwargs):
    """Function to create a movie with the current settings.
    Input:
      - output: string or 1D-array of strings. If string: <<<var>>>,
        <<<time>>>, <<<level>>>, <<<long_name>>>, <<<unit>>> will be
        replaced by the attributes contained in the figures. If 1D-
        array: The length of the array must fit to the specified figures
      - fmt: Dictionary (Default: {}). Formatoptions (same hierarchical
        order as in the initialization function) where the values of the
        formatoption keywords need to be 1D-arrays with the same length
        as the number of steps of the movie (e.g. to modify the title of
        variable 't2m' with three time steps:
        fmt = {'t2m':{'title':['title1','title2','title3']}}).
      - cbarfmt: Dictionary or list of dictionaries (Default: {}). Same
        settings as for update_cbar function but (like fmt) with values
        of formatoption keywords being 1D-arrays with same length as number
        of steps
      - steps: List of integers or 'all'. If 'all', all timesteps in the
        nc-file are used for the movie. Otherwise set the timesteps as a list
      - Additional arguments (*args) and keyword arguments may be figures,
        mapBase instances, vlst=[...], etc. as used in the get_figs func-
        tion to specify the figures to make movies of.
      - Furthermore any valid keyword of the FuncAnimation save function
        can be set. Default value for writer is 'imagemagick', and extra_args
        are ['-vcodec', 'libx264']. Please note, if filename is in the addi-
        tional keywords, it will replace the output variable.
        The additional keywords inherited from FuncAnimation.save function are
        
    """
    # docstring will be extended below
    # default options for kwargs if not 'vlst', 'times', etc.
    defaults = {'dpi':None, 'fps':3, 'writer':'imagemagick', 'extra_args':['-vcodec', 'libx264']}
    # options as set in kwargs
    movieops = {key:value for key, value in kwargs.items() if key not in ['vlst','times','levels', 'wind']}
    for key, value in defaults.items(): movieops.setdefault(key, value)
    # delete options from kwargs
    kwargs = {key:value for key, value in kwargs.items() if key in ['vlst','times','levels', 'wind']}
    # reset output to 'filename' in movieops if given
    if 'filename' in movieops: output = movieops.pop('filename')
    
    fmt = self._setupfmt(fmt)
    
    figs = self.get_figs(*args, **kwargs)
    
    for fig in figs:
      if isinstance(output, str):
        names, times, levels, long_names, units = self.get_names(fig)
        out = output.replace('<<<var>>>','-'.join([name for name in names])).replace('<<<time>>>','-'.join(str(time) for time in times)).replace('<<<level>>>','-'.join(str(level) for level in levels)).replace('<<<unit>>>','-'.join(unit for unit in units)).replace('<<<longname>>>','-'.join(longname for longname in long_names))
      else:
        out = output[i]
      maps = figs[fig]
      cbars = self.get_cbars(*maps)
      
      if steps == 'all':
        for timename in self.timenames:
          try:
            steps = xrange(self.nco.variables[maps[0].name]._shape()[self.nco.variables[maps[0].name].dimensions.index(timename)])
            break
          except ValueError: pass
      
      # save bound options
      bounds = [getattr(mapo.fmt, 'bounds') for mapo in maps]
      windbounds = [getattr(mapo.wind.fmt, 'bounds') for mapo in maps if hasattr(mapo, 'wind') and mapo.wind is not None]
      
      # modify bounds
      print("Calculate bounds")
      
      # handle the mapobject coordinated by one single cbar
      for cbar in cbars:
        if isinstance(cbar.maps[0], fieldplot):
          if cbar.fmt.bounds[0] in ['rounded', 'sym', 'minmax', 'roundedsym']:
            cbar.fmt.bounds = returnbounds(map(lambda x: (np.min(x), np.max(x)), chain(*((data[1] for data in mapo._moviedata(steps, nowind=True)) for mapo in [mapo for mapo in maps if mapo in cbar.maps]))), cbar.fmt.bounds)
      # now handle the rest of the mapobjects
      for mapo in (mapo for mapo in maps if all(mapo not in cbar.maps for cbar in cbars)):
        if isinstance(mapo.fmt.bounds, tuple) and isinstance(mapo.fmt.bounds[0], str):
          if isinstance(mapo, fieldplot):
            if mapo.fmt.bounds[0] in ['rounded', 'sym', 'minmax', 'roundedsym']:
              mapo.fmt.bounds = returnbounds(map(lambda x: (np.min(x), np.max(x)), (data[1] for data in mapo._moviedata(steps, nowind=True))), mapo.fmt.bounds)
          if isinstance(mapo, windplot) or (hasattr(mapo, 'wind') and mapo.wind is not None and mapo.wind._bounds is not None):
            if isinstance(mapo, windplot): wind = mapo
            else: wind = mapo.wind
            if wind.fmt.bounds[0] in ['rounded', 'sym', 'minmax', 'roundedsym']: wind.fmt.bounds = returnbounds(map(lambda x: (np.min(x), np.max(x)), (np.power(data[1]*data[1]+data[2]*data[2], 0.5) for data in wind._moviedata(steps))), wind.fmt.bounds)
      
      # izip has no __len__ method which is required by the animation function. Therefore we define a subclass and use it for the data generator
      class myizip(izip):
        def __len__(self): return len(steps)
      
      # data generator
      if cbars != []: data_gen = myizip(myizip(*(mapo._moviedata(steps, **fmt[mapo.name]['t'+str(mapo.time)]['l'+str(mapo.level)]) for mapo in maps)), myizip(*(cbar._moviedata(steps, **cbarfmt) for cbar in cbars)))
      else: data_gen = myizip(*(mapo._moviedata(steps, **fmt[mapo.name]['t'+str(mapo.time)]['l'+str(mapo.level)]) for mapo in maps))
      # run function
      if cbars != []: runmovie = lambda args: [mapo._runmovie(args[0][maps.index(mapo)]) for mapo in maps] + [cbar._runmovie(args[1][cbars.index(cbar)]) for cbar in cbars]
      else: runmovie = lambda args: [mapo._runmovie(args[maps.index(mapo)]) for mapo in maps]
      
      # movie initialization function
      def init_func():
        self.update({}, add = False, delete = False)
        if self._cbars != []: self.update_cbar({}, add = False, delete = False)
      
      print("Make movie")
      ani = FuncAnimation(fig, runmovie, frames=data_gen, repeat=True, init_func=init_func)
      
      if out == 'show': plt.show()
      else: ani.save(out, **movieops)
      if not out == 'show': print('Saved movie to ' + out)
      
    # restore initial settings
    self._initcompatible = True
    self._reorder()
    self.update(self._fmt[-1][0], add=False, delete=False, todefault = True)
    self._initcompatible = False
    self._reorder()
    if self._fmt[-1][1] == []:
      for cbar in self._cbars: cbar._removecbar()
    else: self.update_cbar(*self._fmt[-1][1], add=False, delete=False, todefault = True)
        

  def _setupfmt(self, formatoptions):
    # set up the formatoptions for each variable
    if self.maps == {}:
      vlst   = self.vlst
      times  = {var:['t' + str(time) for time in self.times] for var in vlst}
      levels = {var:{t:['l'+str(level) for level in self.levels] for t in times[var]} for var in vlst}
    else:
      vlst = sorted(self.maps.keys())
      times = {var:sorted(self.maps[var].keys()) for var in vlst}
      levels = {var:{t:sorted(self.maps[var][t].keys()) for t in times[var]} for var in vlst}
    strtimes = []
    for var in vlst: strtimes = strtimes + times[var]
    strlevels = []
    for var in vlst:
      for time in levels[var]: strlevels = strlevels + levels[var][time]
    """set up the formatoptions for the map objects"""
    if formatoptions is None: fmt = {var:{time:{level:{} for level in strlevels} for time in strtimes} for var in vlst}
    elif isinstance(formatoptions,fmtBase): fmt = {var:{time:{level:formatoptions for level in levels[var][time]} for time in levels[var]} for var in levels}
    elif isinstance(formatoptions,dict):
      fmt = {}
      for var in vlst:
        fmt.update({var:{}})
        for time in times[var]:
          fmt[var].update({time:{}})
          for level in levels[var][time]:
            fmt[var][time][level] = {key:value for key, value in formatoptions.items() if key not in vlst+strtimes+strlevels}
            if time in formatoptions: fmt[var][time][level].update({key:value for key, value in formatoptions[time].items() if key not in vlst+strtimes+strlevels})
            if level in formatoptions: fmt[var][time][level].update({key:value for key, value in formatoptions[level].items() if key not in vlst+strtimes+strlevels})
            if var in formatoptions:
              fmt[var][time][level].update({key:value for key, value in formatoptions[var].items() if key not in vlst+strtimes+strlevels})
              if level in formatoptions[var]: fmt[var][time][level].update({key:value for key, value in formatoptions[var][level].items() if key not in vlst+strtimes+strlevels})
              if time in formatoptions[var]:
                fmt[var][time][level].update({key:value for key, value in formatoptions[var][time].items() if key not in vlst+strtimes+strlevels})
                if level in formatoptions[var][time]: fmt[var][time][level].update({key:value for key, value in formatoptions[var][time][level].items() if key not in vlst+strtimes+strlevels})
    return fmt
  
  def script(self, output, reduced = True):
    """Function to create a script named output with the current formatoptions.
    To get all the formatoptions for every mapBase object, set reduced = False.
    Experimental function! Please take care of bounds and colormaps in the output
    script."""
    import datetime as dt
    with open(output,'w') as f:
      f.write("# -*- coding: utf-8 -*-\n# script for the generation of nc2map.maps object. Time created: """ + dt.datetime.now().strftime("%d/%m/%y %H:%M") + '\n' + "import nc2map\nncfile = '" + str(self.fname) + "'\nvlst = " + str(self.vlst) + "\ntimes = " + str(self.times) + "\nlevels = " + str(self.levels) + "\ntimenames = " + str(self.timenames) + "\nlevelnames = " + str(self.levelnames) + "\nlon = " + str(self.lonnames) + "\nlat = " + str(self.latnames) + "\nsort = '" + str(self.sort) + "'\n\nfmt = " + str(self.asdict(initcompatible=True, reduced = reduced)) + "\n")
      
      openstring = "mymaps = nc2map.maps(ncfile = ncfile, vlst = vlst, times = times, levels = levels, formatoptions = fmt, lon = lon, lat = lat, timenames = timenames, levelnames = levelnames, sort = sort"
      
      if self._cbars != []:
        f.write("onecbar = " + str(self.asdict('cbars')) + "\n")
        openstring = openstring + ", onecbar = onecbar"
        
      if self._subplot_shape is not None:
        f.write("ax = " + str(self._subplot_shape) + "\n")
        openstring = openstring + ", ax=ax"
      
      if self.figsize is not None:
        f.write("figsize = " + str(self.figsize))
        openstring = openstring + ", figsize=figsize"
      
      f.write("\n" + openstring + ")")
        
  def asdict(self, *args, **kwargs):
    """returns the current formatoptions of all mapBase objects and
    cbarmangers as dictionary.
    Arguments may be
      - 'maps' to return only the dictionary controlling the mapBase
        instances (see formatoptions in the initialization) (Default)
      - 'cbars' to return only the dictionary controlling the
        cbarmanager instances (see onecbar in the initialization)
     Keyword arguments (kwargs) may be
      - initcompatible: Boolean (Default: False). Returns the diction-
        ary in such a way that the maps object can be reiniatlized
        (i.e. with the original variables, times and levels as keys).
      - reduced: Boolean (Default: True). Reduces the formatoptions
        such that if formatoption keywords are multiply set for more
        than one instances (e.g. for all variables), they will be
        put together. As an example:
        {<<<var1>>>:{<<<t1>>>>:{<<<l1>>>:{<<<keyword>>>:<<<value>>>}}},
        <<<var2>>>:{<<<t2>>>>:{<<<l2>>>:{<<<keyword>>>:<<<value>>>}}}}
        will be reduced to {<<<keyword>>>:<<<value>>>} (as it is suitable
        for update and initialization function but shorter).
    """
    initcompatible = kwargs.get('initcompatible', False)
    reduced        = kwargs.get('reduced', True)
    fmt = {}
    cbars = []
    if args == () or 'maps' in args:
      returnfmt = True
      if initcompatible: ending='orig'
      else: ending = ''
      for mapo in self.get_maps():
        if getattr(mapo,'name'+ending) not in fmt: fmt[getattr(mapo,'name'+ending)] = {}
        if 't'+str(getattr(mapo,'time'+ending)) not in fmt[getattr(mapo,'name'+ending)]: fmt[getattr(mapo,'name'+ending)]['t'+str(getattr(mapo,'time'+ending))] = {}
        fmt[getattr(mapo,'name'+ending)]['t'+str(getattr(mapo,'time'+ending))]['l'+str(getattr(mapo,'level'+ending))] = mapo.asdict()
      if reduced: fmt = self._reducefmt(fmt)
    else: returnfmt = False
    if 'cbars' in args:
      returncbars = True
      for cbar in self._cbars:
        names, times, levels, long_names, units = self.get_names(*cbar.maps)
        cbars.append({key:value for key,value in cbar.fmt.asdict().items()})
        if names  != self.vlst:   cbars[-1]['vlst']   = names
        if times  != self.times:  cbars[-1]['times']  = times
        if levels != self.levels: cbars[-1]['levels'] = levels
    else: returncbars = False
    if returnfmt and returncbars: return (fmt, cbars)
    if returnfmt: return (fmt)
    if returncbars: return (cbars)
  
  def _reducefmt(self, formatoptions):
    """reduce the given formatoptions"""
    dims = ['time','var','level','u','v']
    # set up vlst
    mainvlst = sorted(formatoptions.keys())
    # set up times and list including all times (without duplicates)
    times = {var:sorted(formatoptions[var].keys()) for var in mainvlst}
    maintimelist = []
    for var in mainvlst:
      for time in times[var]:
        if time not in maintimelist: maintimelist.append(time)
    maintimelist.sort()
    # set up levels and list including all levels (without duplicates)
    levels = {var:{t:sorted(formatoptions[var][t].keys()) for t in times[var]} for var in mainvlst}
    mainlevellist = []
    for var in mainvlst:
      for t in levels[var]:
        for level in levels[var][t]:
          if level not in mainlevellist: mainlevellist.append(level)
    mainlevellist.sort()
    # set up options and list including all times (without duplicates)
    options = {}
    for var in mainvlst:
      for time in times[var]:
        for level in levels[var][time]:
          for option in formatoptions[var][time][level]:
            if option not in options: options[option] = {'list':[]}
            options[option]['list'].append((var,time,level,formatoptions[var][time][level][option]))
    optionlist = options.keys()
    mask = np.zeros((len(mainvlst),len(maintimelist),len(mainlevellist)), dtype=int)
    for var in mainvlst:
      for time in [t for t in maintimelist if t not in times[var]]: mask[mainvlst.index(var),maintimelist.index(time),:] = 1
      for time in levels[var]:
        for level in [l for l in mainlevellist if l not in levels[var][time]]: mask[mainvlst.index(var),maintimelist.index(time),mainlevellist.index(level)] = 1
    mask = np.ma.make_mask(mask, shrink = False)
    fill_value = 'XXXXXXXX'
    fmt = {}
    for option in options:
      vlst      = mainvlst[:]
      timelist  = maintimelist[:]
      levellist = mainlevellist[:]
      array = np.zeros((len(vlst),len(timelist),len(levellist)), dtype=object)
      array[:,:,:] = fill_value
      #if option in dims: array = np.ma.array(array, mask = np.ma.make_mask(np.zeros((len(vlst),len(timelist),len(levellist)), dtype=object), shrink=False), shrink=False)
      array = np.ma.array(array, mask = mask, shrink=False)
      for dimtuple in options[option]['list']: array[vlst.index(dimtuple[0]), timelist.index(dimtuple[1]), levellist.index(dimtuple[2])] = dimtuple[3]
      if np.all(array == array.compressed()[0]): fmt[option] = array.compressed()[0]
      else:
        for var in mainvlst:
          if len(array[vlst.index(var),:].compressed()) > 0 and np.all(array[vlst.index(var),:] == array[vlst.index(var),:].compressed()[0]) and array[vlst.index(var),:].compressed()[0] != fill_value:
            if not var in fmt: fmt[var]={}
            fmt[var][option] = array[vlst.index(var),:].compressed()[0]
            array= np.delete(array, vlst.index(var), 0)
            vlst.pop(vlst.index(var))
        for time in maintimelist:
          if len(array[:,timelist.index(time),:].compressed()) > 0 and np.all(array[:,timelist.index(time),:] == array[:,timelist.index(time),:].compressed()[0]) and array[:,timelist.index(time),:].compressed()[0] != fill_value:
            if time not in fmt: fmt[time] = {}
            fmt[time][option] = array[:,timelist.index(time),:].compressed()[0]
            array = np.delete(array, timelist.index(time), 1)
            timelist.pop(timelist.index(time))
        for level in mainlevellist:
          if len(array[:,:,levellist.index(level)].compressed()) > 0 and np.all(array[:,:,levellist.index(level)] == array[:,:,levellist.index(level)].compressed()[0]) and array[:,:,levellist.index(level)].compressed()[0] != fill_value:
            if level not in fmt: fmt[level] = {}
            fmt[level][option] = array[:,:,levellist.index(level)].compressed()[0]
            array = np.delete(array, levellist.index(level), 2)
            levellist.pop(levellist.index(level))
        for var in vlst:
          for level in levellist:
            if len(array[vlst.index(var),:,levellist.index(level)].compressed()) > 0 and np.all(array[vlst.index(var),:,levellist.index(level)] == array[vlst.index(var),:,levellist.index(level)].compressed()[0]) and array[vlst.index(var),:,levellist.index(level)].compressed()[0] != fill_value:
              if not var in fmt: fmt[var]= {}
              if not time in fmt[var]: fmt[var][level] = {}
              fmt[var][level][option] = array[vlst.index(var),:,levellist.index(level)].compressed()[0]
          for time in [t for t in timelist if t in times[var]]:
            if len(array[vlst.index(var),timelist.index(time),:].compressed()) > 0 and np.all(array[vlst.index(var),timelist.index(time),:] == array[vlst.index(var),timelist.index(time),:].compressed()[0]) and array[vlst.index(var),timelist.index(time),:].compressed()[0] != fill_value:
              if not var in fmt: fmt[var]= {}
              if not time in fmt[var]: fmt[var][time] = {}
              fmt[var][time][option] = array[vlst.index(var),timelist.index(time),:].compressed()[0]
            else:
              for level in levels[var][time]:
                if level in levellist and not (var in fmt and level in fmt[var] and option in fmt[var][level]) and array[vlst.index(var),timelist.index(time),levellist.index(level)] != fill_value:
                  if not var in fmt: fmt[var]= {}
                  if not time in fmt[var]: fmt[var][time] = {}
                  if not level in fmt[var][time]: fmt[var][time][level] = {}
                  fmt[var][time][level][option] = array[vlst.index(var),timelist.index(time),levellist.index(level)]
    return fmt
  
  def get_fmtkeys(self,*args):
    """Function which returns a dictionary containing all possible
    formatoption settings as keys and their documentation as value.
    Arguments (*args) may be any keyword of the formatoptions plus
    wind (to plot the 'wind' formatoption keywords) and 'windonly'
    (to plot the wind only formatoption keywords (i.e. not
    projection keywords, etc.))"""
    if not 'wind' in args: return self.get_maps()[0].get_fmtkeys(*args)
    else: return self.get_winds()[0].get_fmtkeys(*args)
  
  def show_fmtkeys(self,*args):
    """Function which prints the keys and documentations in a readable
    manner.
    Arguments (*args) may be any keyword of the formatoptions
    (without: Print all), plus wind (to plot the 'wind' formatoption
    keywords) and 'windonly' (to plot the wind only formatoption key-
    words (i.e. not projection keywords, etc.))"""
    if not 'wind' in args: return self.get_maps()[0].get_fmtkeys(*args)
    else: return self.get_winds()[0].get_fmtkeys(*args)
  
  # ------------------ modify docstrings here --------------------------
  __init__.__doc__   = __init__.__doc__ + '\n' + '\n'.join((key+':').ljust(20) + get_docs()[key] for key in sorted(get_docs().keys())) \
                       + '\n\nAnd the windplot specific options are\n\n' + '\n'.join((key+':').ljust(20) + get_docs('wind', 'windonly')[key] \
                       for key in sorted(get_docs('wind', 'windonly').keys()))
  output.__doc__     = output.__doc__ + plt.savefig.__doc__[plt.savefig.__doc__.find('Keyword arguments:') + len('Keyword arguments:\n'):]
  make_movie.__doc__ = make_movie.__doc__ + '    ' + FuncAnimation.save.__doc__[FuncAnimation.save.__doc__.find('*'):]
    
class mapBase(object):
  """Class controlling a the plot of a single variable, timestep and level of a netCDF
  file. Properties are below. It is not recommended to use one single mapBase instance
  but to use the maps class. Nevertheless: for initialization see function __init__
  Methods are:
    - asdict: Returns the current formatoptions and variables, etc. as a dictionary
    - get_fmtkeys: Shortcut to the fmtBase instance get_fmtkeys function
    - show_fmtkeys: Shortcut to the fmtBase instance show_fmtkeys function
  Further Methods are defined in the subclasses fieldplot and windplot. For manual
  initilization use the _setupproj function after initilization before making the plot.
  """
  props = mapproperties() # container containing methods for property definition
  # ------------------ define properties here -----------------------
  # General properties
  nameorig       = props.default('nameorig',"""String. Original variable name from the initialization""")
  name           = props.default('name',"""String. Variable name""")
  timeorig       = props.default('timeorig',"""Original time from the initialization""")
  time           = props.default('time',"""Integer (Default: 0). timestep in nc-file""")
  levelorig      = props.default('levelorig',"""Original level from the initialization""")
  level          = props.default('level',"""Integer (Default: 0). Level in the nc-file""")
  units          = props.ncoprop('units', """Unit of the variable as saved in the netCDF file""")
  long_name      = props.ncoprop('long_name', """Long name of the variable as stored in the netCDF file""")
  fname          = props.default('fname', """Name of the nc-file""")
  timenames      = props.default('timenames', """List of timenames to look for in the netCDF file""")
  levelnames     = props.default('levelnames', """List of level names to look for in the netCDF file""")
  lonnames       = props.default('lonnames', """List of longitude names to look for in the netCDF file""")
  latnames       = props.default('latnames', """List of latitude names to look for in the netCDF file""")
  nco            = props.nco('nco', """netCDF4.MFDataset instance of ncfile""")
  ax            = props.default('ax',"""axes instance the mapBase instance plots on.""")
  
  # Data properties
  lonorig        = props.dim('lonorig', """numpy array. Original (i.e. non-shifted) longitude data""")
  lon            = props.dim('lon', """numpy array. (Possibly shifted) Longitude data.""")
  lon2d          = props.default('lon2d', """numpy.ndarray. Two dimensional longitude as used for the plot""")
  lat            = props.dim('lat', """numpy array. Original latitude data""")
  lat2d          = props.default('lat2d', """numpy.ndarray. Two dimensional latitude as used for the plot""")
  

  
  def __init__(self, ncfile, time = 0, level = 0, ax = None, fmt = {}, nco=None, timenames = ['time'], levelnames = ['level', 'lvl', 'lev'], lon=['lon', 'longitude', 'x'], lat=['lat', 'latitude', 'y'], ax_shapes=None, ax_num=None, mapsobj = None):
    """
    Input:
     - ncfile:    string or 1D-array of strings. Path to the netCDF-file containing the
        data for all variables. Filenames may contain wildcards (*, ?, etc.) as suitable
        with the Python glob module (the netCDF4.MFDataset is used to open the nc-file).
     - time: integer. Timestep which shall be plotted
     - level: integer. level which shall be plotted
     - ax:  matplotlib.axes.AxesSubplot instance matplotlib.axes.AxesSubplot
       where the data can be plotted on
     - ax_shapes: Tuple (x,y). Gives the original shape of the axes instances
       (e.g if ax was initialized by plt.subplot(x,y,z)). Probably necessary to remove
       and update colorbars.
     - ax_num: integer. Original number of the axes instance (e.g if ax was initialized
       by plt.subplot(x,y,z), it is z). Probably necessary to remove       and update colorbars.
     - mapsobj: The maps instance this mapBase instance belongs to.
     - nco: the netCDF4.MFDataset instance which is used here (if None: it will
       be opened by the use of ncfile above)
     - timenames: 1D-array of strings: Gives the name of the time-dimension for which will be
        searched in the netCDF file
     - levelnames: 1D-array of strings: Gives the name of the fourth dimension (e.g vertical levels)
       for which will be searched in the netCDF file
     - lon: 1D-array of strings: Gives the name of the longitude-dimension for which will be
       searched in the netCDF file
     - lat: 1D-array of strings: Gives the name of the latitude-dimension for which will be
       searched in the netCDF file
    """
    # original time and level from initialization
    self.timeorig  = time
    self.levelorig = level
    
    if isinstance(fmt, dict):
      time = fmt.get('time',time)
      level = fmt.get('level', level)
    
    # time as used for the plot
    self.time      = time
    self.level     = level
    self.ax        = ax
    self.fname     = ncfile
    self._subplot_shape = ax_shapes
    self._ax_num   = ax_num
    self._maps     = mapsobj # parent maps object
    
    # open netCDF4.Dataset
    self.nco       = nco
    # set up dimension settings
    self.lonorig   = lon
    self.figtitle  = None
    self.texts     = {'axes':[], 'fig':[], 'data':[]}
    self.lon       = lon
    self.lat       = lat
    self.lonnames  = lon
    self.latnames  = lat
    self.timenames = timenames
    self.levelnames= levelnames
    self._dims      = {'lon':lon, 'lat':lat, 'time':timenames, 'level':levelnames}
    if hasattr(self, '_furtherinit'): self._furtherinit(fmt=fmt,**kwargs)
    
  def _reinit(self, time=None, level=None, ncfile=None, ax=None, **kwargs):
    if time is not None:     self.time      = time
    if level is not None:    self.level     = level
    if ax is not None:       self.ax        = ax
    plt.sca(self.ax)

    if ncfile is not None:   self.fname     = ncfile
    # set up dimension settings
    self.lonorig   = self.lonnames
    self.lon       = self.lonnames
    self.lat       = self.latnames
  
  def _setupproj(self):
    plt.sca(self.ax)
    plt.cla()
    if hasattr(self, 'plot'):
      del self.plot
    self._removecbar(['b','r'])
    if hasattr(self,'wind'):
      if self.wind is not None:
        self.wind._removeplot()
        self.wind._removecbar(['b','r'])
    basemapops = {}
    # create basemap
    mapproj = bm.Basemap(**self.fmt._projops)
    # draw coastlines
    if self.fmt.lsm: basemapops['lsm']=mapproj.drawcoastlines()
    # color the ocean
    if self.fmt.ocean_color is not None: basemapops['lsmask']=mapproj.drawlsmask(land_color=self.fmt.land_color, ocean_color=self.fmt.ocean_color)
    # draw parallels
    basemapops['parallels'] = mapproj.drawparallels(self.fmt._paraops['parallels'], **{key:value for key, value in self.fmt._paraops.items() if key != 'parallels'})
    # draw meridians
    basemapops['meridionals'] = mapproj.drawmeridians(self.fmt._meriops['meridionals'], **{key:value for key, value in self.fmt._meriops.items() if key != 'meridionals'})
    # draw countries
    if self.fmt.countries: basemapops['countries']=mapproj.drawcountries()
    # configure longitude and latitude
    self.lon2d, self.lat2d = mapproj(self.lon2d, self.lat2d)
    # configure title
    self._configuretitles()
    # save mapproj to attribute
    setattr(self, 'mapproj', mapproj)
    setattr(self,'_basemapops', basemapops)
  
  def _get_data(self, name, time, level):
    """Function to get the data of variable var from the netCDF file for the time
    specified by time and level specified by level. Dimension names for time and
    level will be taken from the mapBase object.
    """
    if name not in self.nco.variables.keys(): sys.exit('Unknown variable name ' + name + '! Possible keys in the ncfile are ' + ','.join(key for key in self.nco.variables.keys()))
    # set up dimension order to read from the netCDF
    ncodims = self.nco.variables[name].dimensions
    ncodims = [dim for dim in ncodims] # convert tuple to list
    for i in xrange(len(ncodims)):
      if ncodims[i] in self.timenames:    ncodims[i] = time
      elif ncodims[i] in self.levelnames: ncodims[i] = level
      elif ncodims[i] in self.lonnames or ncodims[i] in self.latnames:   ncodims[i] = slice(None)
      else: sys.exit("Unknown dimension '" + ncodims[i] + "' in netCDF file!")
    # read data
    data = self.nco.variables[name].__getitem__(ncodims)
    if type(data) is not np.ma.MaskedArray: data = np.ma.masked_array(data, mask=np.ma.make_mask(np.zeros(shape=np.shape(data)),shrink=False), copy=True)
    return data
  
  def _mask_to_region(self, data):
    """masks the data if not global is needed using the 2-dimensional longitude and
    latitude as specified in self.lon2d and self.lat2d"""
    if self.fmt._box != self.fmt._defaultrange:
      lonlatminmax = self.fmt._box[:]
      for i in xrange(len(self.fmt._defaultrange)):
        if lonlatminmax[i] != self.fmt._defaultrange[i]:
          if   i == 0 and self.lon[self.lon<self.fmt._box[i]] != []: lonlatminmax[i] = self.lon[self.lon<self.fmt._box[i]].max()
          elif i == 1 and self.lon[self.lon>self.fmt._box[i]] != []: lonlatminmax[i] = self.lon[self.lon>self.fmt._box[i]].min()
          elif i == 2 and self.lat[self.lat<self.fmt._box[i]] != []: lonlatminmax[i] = self.lat[self.lat<self.fmt._box[i]].max()
          elif i == 3 and self.lat[self.lat>self.fmt._box[i]] != []: lonlatminmax[i] = self.lat[self.lat>self.fmt._box[i]].min()
      maskedlon=np.ma.masked_outside(self.lon2d,lonlatminmax[0], lonlatminmax[1])
      maskedlat=np.ma.masked_outside(self.lat2d,lonlatminmax[2],lonlatminmax[3])
      data=np.ma.array(data, mask=maskedlon.mask, copy=True)
      data=np.ma.array(data, mask=maskedlat.mask, copy=True)
      del maskedlon, maskedlat
    return data
  
  def _shift(self, data):
    """shifts the data using mpl_toolkits.basemap.shiftgrid function to position the
    center of the map at 0 longitude"""
    if len(np.shape(self.lon)) == 1:
      if self.lon[0]  == 0:
        data, shiftedlon = bm.shiftgrid(180.,data,self.lonorig,start=False)
        if np.all(shiftedlon != self.lon): self.lon = shiftedlon
    self._meshgrid()
    return data
  
  def _meshgrid(self):
    """converts longitude and latitude data to 2-dimensional vectors using numpys
    meshgrid function"""
    if len(np.shape(self.lon)) == 1: [self.lon2d, self.lat2d] = np.meshgrid(self.lon, self.lat)
    else: [self.lon2d, self.lat2d] = [self.lon, self.lat]
    
  def _configuretitles(self):
    plt.sca(self.ax)
    # first remove old texts which are not in the formatoptions
    for trans in self.texts:
      
      for oldtext in (oldtext for oldtext in self.texts[trans] if all(any(val != fmttext[index] for index, val in enumerate(oldtext.get_position())) for fmttext in self.fmt.text if fmttext[3] == trans)): oldtext.remove(); self.texts[trans].remove(oldtext)
    # check if an update is needed
    for text in self.fmt._textstoupdate:
      if text[3] == 'axes':
        trans = self.ax.transAxes
      elif text[3] == 'fig':
        trans = self.ax.get_figure().transFigure
      else:
        trans = self.ax.transData
      for oldtext in self.texts[text[3]]:
        if oldtext.get_position() == (text[0],text[1]): oldtext.remove(); self.texts[text[3]].remove(oldtext)
      if len(text) == 4: self.texts[text[3]].append(plt.text(text[0], text[1], text[2], transform = trans))
      elif len(text) == 5 : self.texts[text[3]].append(plt.text(text[0], text[1], text[2], transform = trans, fontdict = text[4]))
    
    if self._maps is not None: names, times, levels, long_names, units = self._maps.get_names(self.ax.get_figure())
    else: names, times, levels, long_names, units = [[self.name], [self.time], [self.level], [self.long_name], [self.units]]
    if self.fmt.figtitle is not None and self.figtitle is not None:
      self.figtitle.set_text(self.fmt.figtitle.replace('<<<var>>>',', '.join([name for name in names])).replace('<<<time>>>',', '.join(str(time) for time in times)).replace('<<<level>>>',', '.join(str(level) for level in levels)).replace('<<<unit>>>', ', '.join(unit for unit in units)).replace('<<<longname>>>',', '.join(longname for longname in long_names)))
      self.figtitle.set_size(self.fmt._figtitleops['fontsize'])
      self.figtitle.set_weight(self.fmt._figtitleops['fontweight'])
    elif self.fmt.figtitle is not None:
        fig = self.ax.get_figure()
        self.figtitle = fig.suptitle(self.fmt.figtitle.replace('<<<var>>>',', '.join([name for name in names])).replace('<<<time>>>',', '.join(str(time) for time in times)).replace('<<<level>>>',', '.join(str(level) for level in levels)).replace('<<<unit>>>', ', '.join(unit for unit in units)).replace('<<<longname>>>',', '.join(longname for longname in long_names)), **self.fmt._figtitleops)
        if len(fig.texts) > 1:
          for text in fig.texts[1:]:
            if text.get_position() == self.figtitle.get_position(): del fig.texts[fig.texts.index(text)]
    elif self.figtitle is not None: self.figtitle.set_text('')
    if self.fmt.title is not None: plt.title(self.fmt.title.replace('<<<var>>>',self.name).replace('<<<time>>>',str(self.time)).replace('<<<level>>>',str(self.level)).replace('<<<unit>>>', self.units).replace('<<<longname>>>',self.long_name), **self.fmt._titleops)
    else: plt.title('')
    if hasattr(self,'cbar') and self.fmt.clabel is not None:
      for cbarpos in self.cbar:
        self.cbar[cbarpos].set_label(self.fmt.clabel.replace('<<<var>>>',self.name).replace('<<<time>>>',str(self.time)).replace('<<<level>>>',str(self.level)).replace('<<<unit>>>', self.units).replace('<<<longname>>>',self.long_name), **self.fmt._labelops)
    elif hasattr(self,'cbar'):
      for cbarpos in self.cbar:
        self.cbar[cbarpos].set_label('')
    
  def _draw_colorbar(self):
    if not hasattr(self, 'cbar'): self.cbar = {}
    for cbarpos in self.fmt.plotcbar:
      if cbarpos in self.cbar:
        if cbarpos in ['b','r']:
          self.cbar[cbarpos].update_bruteforce(self.plot)
        else:
          plt.figure(self.cbar[cbarpos].ax.get_figure().number)
          self.cbar[cbarpos].set_cmap(self._cmap)
          self.cbar[cbarpos].set_norm(self._norm)
          self.cbar[cbarpos].draw_all()
          plt.draw()
      else:
        orientations = ['horizontal', 'vertical', 'horizontal', 'vertical']
        cbarlabels   = ['b','r','sh','sv']
        if cbarpos not in cbarlabels: sys.exit('Unknown position option ' + str(cbarpos) + '! Please use one of ' + ', '.join(label for label in cbarlabels) + '.')
        orientation = orientations[cbarlabels.index(cbarpos)]
        if cbarpos in ['b','r']:
          self.cbar[cbarpos] = plt.colorbar(self.plot, orientation = orientation, extend = self.fmt.extend, use_gridspec = True)
        elif cbarpos in ['sh', 'sv']:
          if cbarpos == 'sh':
            fig = plt.figure(figsize=(8,1))
            fig.canvas.set_window_title('Colorbar, var ' + self.name + ', time ' + str(self.time) + ', level ' + str(self.level))
            ax  = fig.add_axes([0.05, 0.5, 0.9, 0.3])
          elif cbarpos == 'sv':
            fig = plt.figure(figsize=(1,8))
            fig.canvas.set_window_title('Colorbar, var ' + self.name + ', time ' + str(self.time) + ', level ' + str(self.level))
            ax  = fig.add_axes([0.3, 0.05, 0.3, 0.9])
          self.cbar[cbarpos] = mpl.colorbar.ColorbarBase(ax, cmap = self._cmap, norm = self._norm, orientation = orientation, extend = self.fmt.extend)
          plt.sca(self.ax)
      if type(self.fmt.ticks) is int:
        if cbarpos in ['b', 'sh']: self.cbar[cbarpos].set_ticks([float(text.get_text()) for text in self.cbar[cbarpos].ax.get_xticklabels()[::self.fmt.ticks]])
        else: self.cbar[cbarpos].set_ticks([float(text.get_text()) for text in self.cbar[cbarpos].ax.get_yticklabels()[::self.fmt.ticks]])
      if self.fmt.ticklabels is not None: self.cbar[cbarpos].set_ticklabels(self.fmt.ticklabels)
      self.cbar[cbarpos].ax.tick_params(labelsize=self.fmt._tickops['fontsize'])
      if self.fmt.clabel is not None: self.cbar[cbarpos].set_label(self.fmt.clabel.replace('<<<var>>>',self.name).replace('<<<time>>>',str(self.time)).replace('<<<level>>>',str(self.level)).replace('<<<unit>>>', self.units).replace('<<<longname>>>',self.long_name), **self.fmt._labelops)
  
  def _removecbar(self, positions = 'all'):
    if not hasattr(self, 'cbar'): return
    if positions == 'all': positions = self.cbar.keys()
    for cbarpos in positions:
      if cbarpos in ['b','r'] and cbarpos in self.cbar:
        if self._subplot_shape is None: self._subplot_shape = input('Note on ' + self.name + ', time ' + str(self.time) + ', level ' + str(self.level) + ':\nOriginal position of the subplot in the subplot grid is unknown. Please enter it as a tuple (x,y)\n')
        if self._ax_num is None: self._ax_num = input('Note on ' + self.name + ', time ' + str(self.time) + ', level ' + str(self.level) + ':\nOriginal number of the subplot in the grid is unknown. Please enter it as an integer\n')
        self.ax.get_figure().delaxes(self.cbar[cbarpos].ax) # delete colorbar axes
        self.ax.change_geometry(self._subplot_shape[0],self._subplot_shape[1], self._ax_num) # reset geometry
        del self.cbar[cbarpos]
      elif cbarpos in ['sh', 'sv'] and cbarpos in self.cbar:
        plt.close(self.cbar[cbarpos].ax.get_figure())
        del self.cbar[cbarpos]
    return
    
  def asdict(self):
    fmt = self.fmt.asdict()
    if self.time != self.timeorig:       fmt.update({'time':self.time})
    if self.level != self.levelorig:     fmt.update({'level':self.level})
    return fmt
  
  def get_fmtkeys(self,*args):
    """Function which returns a dictionary containing all possible formatoption settings
    as keys and their documentation as value (shortcut to self.fmt.get_fmtkeys()"""
    return self.fmt.get_fmtkeys(*args)
  
  def show_fmtkeys(self,*args):
    """Function which prints the keys and documentations in a readable manner (shortcut
    to self.fmt.show_fmtkeys())"""
    self.fmt.show_fmtkeys(*args)
  
class fieldplot(mapBase):
  props = mapproperties()
  # container containing methods for property definition
  # ------------------ define properties here -----------------------
  # Data properties
  data           = props.data('data', """numpy.ma.array. Data of the variable""")
  
  
  def __init__(self, ncfile, var, u = None, v = None, fmt = {}, **kwargs):
    """
    Input:
     - var: string. Name of the variable which shall be plotted and red from the netCDF file
     - u: string (Default: None). Name of the zonal wind variable if a windplot
        shall be visualized
        (set u and v to also plot the wind above the data of var)
      - v: string (Default: None). Name of the meridional wind variable if a windplot shall
        be visualized
      - fmt: dictionary containing the formatoption keywords as keys and settings as value.
        Possible keywords are:
    """
    
    super(fieldplot, self).__init__(ncfile=ncfile, fmt=fmt, **kwargs)
    
    dims = ['time','var','level','u','v']
    
    self.nameorig  = var
    
    # define formatoptions
    if isinstance(fmt, dict):
      var = fmt.get('var',var)
      u   = fmt.get('u', u)
      v   = fmt.get('v', v)
      self.fmt = fieldfmt(**{key:val for key, val in fmt.items() if key not in dims})
    elif isinstance(fmt, fmtBase): self.fmt = fmt
    elif fmt is None:              self.fmt = fieldfmt()
    else: sys.exit('Wrong type ' + str(type(fmt)) + ' for formatoptions')
    
    # initialize data
    self.name      = var
    self.data      = var
    self.units     = var
    self.long_name = var
    if u is not None and v is not None and self.fmt.windplot.enable:
      self.wind = windplot(self.fname, time = self.time, level = self.level, ax = self.ax, fmt = self.fmt.windplot, timenames = self.timenames, levelnames = self.levelnames, lon=self.lonnames, lat=self.latnames, u=u,v=v,nco=self.nco, ax_shapes=self._subplot_shape, ax_num=self._ax_num, mapsobj = self._maps)
    else: self.wind = None
  
  def _reinit(self, var = None, data=None, **kwargs):
    """Function to reinitialize the data"""
    super(fieldplot, self)._reinit(**kwargs)
    if var is None: var = self.name
    if data is None: self.data = var
    else: self.data = data
    if self.name != var: self.name = var; self.units = var; self.long_name = var
  
  def update(self,todefault = False, plot = True, **kwargs):
    """Update the mapBase instance, formatoptions, variable, time and level.
    Possible key words (kwargs) are
      - all keywords as set by formatoptions (see initialization function __init__)
      - time: integer. Sets the time of the mapBase instance and reloads the data
      - level: integer. Sets the level of the mapBase instance and reloads the data
      - var: string. Sets the variable of the mapBase instance and reloads the data
    Additional keys in the windplot dictionary are:
      - u: string. Sets the variable of the windplot instance and reloads the data
      - v: Sets the variable of the windplot instance and reloads the data
      - time, level and var as above
    """
    # set current axis
    plt.sca(self.ax)
    
    # check the keywords
    dims = {'time':self.time,'var':self.name,'level':self.level,'fname':self.fname, 'data':None}
    dimsorig = {'time':self.timeorig,'var':self.nameorig,'level':self.levelorig}
    if np.any([key not in self.fmt._default.keys() + dims.keys() for key in kwargs]): sys.exit('Unknown keywords ' + ', '.join(key for key in kwargs if key not in self.fmt._default.keys() + dims.keys()))
    
    # delete formatoptions which are already at the wished state
    if not todefault: kwargs = {key:value for key, value in kwargs.items() if np.all(value != self.fmt.asdict().get(key, self.fmt._default.get(key, dims.get(key,None))))}
    else:
      oldkwargs = kwargs.copy()
      kwargs = {key:kwargs.get(key, value) for key, value in self.fmt._default.items() if key != 'windplot' and ((key not in kwargs and np.all(value != getattr(self.fmt,key))) or (key in kwargs and np.all(kwargs[key] != getattr(self.fmt,key))))}
      if not self.fmt._enablebounds: kwargs = {key:value for key, value in kwargs.items() if key not in ['cmap','bounds']}
      kwargs.update({key:oldkwargs.get(key, value) for key, value in dimsorig.items() if dims[key] != dimsorig[key] or (key in oldkwargs and oldkwargs[key] != dims[key])})
      if 'windplot' in oldkwargs: kwargs['windplot'] = oldkwargs['windplot']
      else: kwargs['windplot'] = {}
    # update plotting of cbar properties
    if 'plotcbar' in kwargs:
      if kwargs['plotcbar'] in [False, None]: kwargs['plotcbar'] = ''
      if kwargs['plotcbar'] == True: kwargs['plotcbar'] = 'b'
      cbar2close = [cbar for cbar in self.fmt.plotcbar if cbar not in kwargs['plotcbar']]
      self._removecbar(cbar2close)
    
    # update masking options
    maskprops = {key:value for key, value in kwargs.items() if key in self.fmt._maskprops}
    if maskprops != {}: self.fmt.update(**maskprops)
    
    # update basemap properties
    bmprops = {key:value for key, value in kwargs.items() if key in self.fmt._bmprops}
    if bmprops != {}: self.fmt.update(**bmprops)
    # update mapobject dimensions and reinitialize
    newdims = {key:value for key, value in kwargs.items() if key in dims.keys()}
    if newdims != {} or bmprops != {} or maskprops != {}:
      self._reinit(**newdims)
      if self.wind is not None:
        self.wind._reinit()
    if (bmprops != {} or 'tight' in kwargs) and plot: self._setupproj()
    
    # update rest
    self.fmt.update(**{key:value for key, value in kwargs.items() if key not in self.fmt._bmprops + dims.keys()})
    if 'meridionals' in kwargs or 'merilabelpos' in kwargs:
      keys = self.basemapops['meridionals'].keys()
      for key in keys: del self.basemapops['meridionals'][key]
      self.fmt.meridionals = self.fmt.meridionals
      if plot: self.basemapops['meridionals'] = self.mapproj.drawmeridians(self.fmt._meriops['meridionals'], **{key:value for key, value in self.fmt._meriops.items() if key != 'meridionals'})
    if 'parallels' in kwargs or 'paralabelpos' in kwargs:
      keys = self.basemapops['parallels'].keys()
      for key in keys: del self.basemapops['parallels'][key]
      self.fmt.parallels = self.fmt.parallels
      if plot: self.basemapops['parallels'] = self.mapproj.drawparallels(self.fmt._paraops['parallels'], **{key:value for key, value in self.fmt._paraops.items() if key != 'parallels'})
    if 'lsm' in kwargs:
      if 'lsm' in self._basemapops: self._basemapops['lsm'].remove(); del self._basemapops['lsm']
      if kwargs['lsm']: self._basemapops=self.mapproj.drawcoastlines()
    if 'countries' in kwargs:
      if 'countries' in self._basemapops: self._basemapops['countries'].remove(); del self._basemapops['countries']
      if kwargs['countries']: self._basemapops['countries'] = self.mapproj.drawcountries()
    # update wind
    if self.wind is not None or 'windplot' in kwargs:
      if 'windplot' in kwargs:
        windops = kwargs['windplot']
        if windops.get('u', None) is not None and windops.get('v', None) is not None and self.fmt.windplot.enable and self.wind is None:
          self.wind = windplot(self.fname, time = self.time, level = self.level, ax = self.ax, fmt = self.fmt.windplot, timenames = self.timenames, levelnames = self.levelnames, lon=self.lonnames, lat=self.latnames, u=windops['u'],v=windops['v'],nco=self.nco)
        for key in ['time','level']:
          if key in kwargs: windops.setdefault(key, kwargs[key])
      else: windops = {}
      if self.wind is not None: self.wind.update(plot=False, todefault=todefault, **windops)
    if plot: self._configuretitles()
    if plot: self.make_plot()
  
  def _mask_data(self,data):
    """mask the data if maskbelow, maskabove or maskbetween is not None"""
    if self.fmt.maskbelow is not None:   data = np.ma.masked_less(data, self.fmt.maskbelow, copy = True)
    if self.fmt.maskabove is not None:   data = np.ma.masked_greater(data, self.fmt.maskabove, copy = True)
    if self.fmt.maskbetween is not None: data = np.ma.masked_inside(data, self.fmt.maskbetween[0], self.fmt.maskbetween[1], copy = True)
    return data
  
  def _moviedata(self, times, nowind=False, **kwargs):
    """generator to get the data for the movie"""
    for time in times:
      if self.wind is None or not self.wind.fmt.enable or nowind:
        yield (time,
               self._mask_data(self._mask_to_region(self._shift(self._get_data(self.name, time, self.level)))), # data
               {key:value[times.index(time)] for key, value in kwargs.items() if key != 'windplot'}) # formatoptions
      else:
        yield (time, # timestep
               self._mask_to_region(self._shift(self._get_data(self.name, time, self.level))), # data
               self._mask_to_region(self._shift(self._get_data(self.wind.uname, time, self.level))), # udata
               self._mask_to_region(self._shift(self._get_data( self.wind.vname, time, self.level))), # vdata
               {key:value[times.index(time)] for key, value in kwargs.items()}) # formatoptions
  
  def _runmovie(self, args):
    """Function to update the movie with args from self._moviedata"""
    if len(args) ==5:
      if 'windplot' not in args[-1]: args[-1]['windplot'] = {}
      args[-1]['windplot'].update({'udata':args[2], 'vdata':args[3]})
    self.update(time=args[0], data=args[1], **args[-1])
    return
  
  def make_plot(self):
    """Make the plot with the current settings and the windplot. Use it after reinitialization
    and _setupproj. Don't use this function to update the plot but rather the update function!"""
    plt.sca(self.ax)
    if self.fmt.bounds[0] in ['rounded', 'sym', 'minmax', 'roundedsym']:
      self._bounds = returnbounds(self.data, self.fmt.bounds)
    else: self._bounds = self.fmt.bounds
    self._cmap   = get_cmap(self.fmt.cmap, N=len(self._bounds)-1)
    self._norm   = mpl.colors.BoundaryNorm(self._bounds, self._cmap.N)
    if hasattr(self, 'plot'):
      self.plot.set_cmap(self._cmap)
      self.plot.set_norm(self._norm)
      self.plot.set_array(self.data[:-1,:-1].ravel())
    else:
      # make plot
      self.plot = self.mapproj.pcolormesh(self.lon, self.lat, self.data, cmap = self._cmap, norm = self._norm, rasterized = self.fmt.rasterized)
    if not (self.fmt.plotcbar == '' or self.fmt.plotcbar == () or self.fmt.plotcbar is None or self.fmt.plotcbar is False):
      self._draw_colorbar()
    if self.wind is not None:
      self.wind.lon2d = self.lon2d
      self.wind.lat2d = self.lat2d
      if self.wind.fmt.enable: self.wind.make_plot()
    if self.fmt.tight: plt.tight_layout()
  
  def close(self,*args):
    """Close the mapBase instance.
    Arguments may be
     - 'data': To delete all data (but not close the netCDF4.MFDataset
     instance)
     - 'figure': To close the figure and the figures of the colorbars.
    Without any arguments, everything will be closed (including the
    netCDF4.MFDataset instance) and deleted.
    """
    if 'data' in args or args is ():
      del self.data
      del self.lon2d
      del self.lat2d
      del self.lon
      del self.lat
      if self.wind is not None: self.wind.close('data')
    if 'figure' in args or args is ():
      plt.close(self.ax.get_figure())
      self._removecbar(['sh','sv'])
    if args == (): del self.nco
  
  def asdict(self):
    """Returns a dictionary containing the current formatoptions and (if the time,
    level or name changed compared to the original initialization) the time, level
    or name"""
    fmt = self.fmt.asdict()
    if self.time != self.timeorig:       fmt.update({'time':self.time})
    if self.level != self.levelorig:     fmt.update({'level':self.level})
    if self.name != self.nameorig:       fmt.update({'var':self.name})
    return fmt
  
  # ------------------ modify docstrings here --------------------------
  __init__.__doc__   = __init__.__doc__ + '\n' + '\n'.join((key+':').ljust(20) + get_docs()[key] for key in sorted(get_docs().keys())) \
                       + '\n\nAnd the windplot specific options are\n\n' + '\n'.join((key+':').ljust(20) + get_docs('wind', 'windonly')[key] \
                       for key in sorted(get_docs('wind', 'windonly').keys())) + '\nFurther keyword arguments (kwargs) inherited by mapBase object are:\n' + mapBase.__init__.__doc__
  
class windplot(mapBase):
  props = mapproperties()
  # container containing methods for property definition
  # ------------------ define properties here -----------------------
  # Data properties
  unameorig      = props.default('unameorig', """numpy.ma.array. Original name of the zonal windfield variable as used in
                   the initialization""")
  uname          = props.default('uname', """numpy.ma.array. Name of the zonal windfield variable.""")
  vnameorig      = props.default('vnameorig', """numpy.ma.array. Original name of the meridional windfield variable as used in
                   the initialization""")
  vname          = props.default('vname', """numpy.ma.array. Name of the meridional windfield variable.""")
  u              = props.data('u', """numpy.ma.array. Data of the zonal windfield""")
  v              = props.data('v', """numpy.ma.array. Data of the meridional wind field""")
  speed          = props.speed('speed', """numpy.ma.array. Speed as calculated from u and v""")
  weights        = props.weights('weights', """numpy array. Grid cell weights as calculated via cdos""")
  
  
  def __init__(self, ncfile, u, v, var = 'wind', fmt = {}, **kwargs):
    """
    Input:
     - u: string. Name of the zonal wind variable
     - v: string. Name of the meridional wind variable 
     - var: string (Default: 'wind'). Name of the variable.
     - fmt: dictionary containing the formatoption keywords as keys and settings as value.
        Possible keywords are:
    """
    super(windplot, self).__init__(ncfile=ncfile, fmt=fmt, **kwargs)
    
    dims = ['time','var','level','u','v']
    
    self.nameorig  = var
    self.unameorig = u
    self.vnameorig = v
    
    # define formatoptions
    if isinstance(fmt, dict):
      var = fmt.get('var',var)
      u   = fmt.get('u', u)
      v   = fmt.get('v', v)
      self.fmt = windfmt(**{key:val for key, val in fmt.items() if key not in dims})
    elif isinstance(fmt, fmtBase): self.fmt = fmt
    elif fmt is None:              self.fmt = windfmt()
    else: sys.exit('Wrong type ' + str(type(fmt)) + ' for formatoptions')
    
    self.name      = var
    self.uname     = u
    self.vname     = v
    self.u         = u
    self.v         = v
    self.units     = u
    self.long_name = 'Wind speed'
    if self.fmt.reduce and not self.fmt.streamplot:      self._reduceuv(perc=self.fmt.reduce)
    if self.fmt.reduceabove and not self.fmt.streamplot: self._reduceuv(perc=self.fmt.reduceabove[0], pctl=self.fmt.reduceabove[1])
  
  def _reinit(self, u=None, v=None, var = None, udata=None, vdata=None, **kwargs):
    """Function to reinitialize the data"""
    super(windplot, self)._reinit(**kwargs)
    if hasattr(self, 'speed'): del self.speed
    if var is not None:  self.name = var
    if u is None:        u = self.uname
    if v is None:        v = self.vname
    if udata is None: self.u = self.uname
    else:             self.u = udata
    if vdata is None: self.v = self.vname
    else:             self.v = vdata
    if u != self.uname: self.uname = u; self.units = u; self.long_name = u
    if v != self.vname: self.vname = v
    if self.fmt.reduce is not None and not self.fmt.streamplot:      self._reduceuv(perc=self.fmt.reduce)
    if self.fmt.reduceabove is not None and not self.fmt.streamplot: self._reduceuv(perc=self.fmt.reduceabove[0], pctl=self.fmt.reduceabove[1])
    
  def make_plot(self):
    """Make the plot with the current settings and the windplot. Use it after reinitialization
    and _setupproj. Don't use this function to update the plot but rather the update function!"""
    plt.sca(self.ax)
    
    # configure windplot options
    plotops = self.fmt._windplotops
    if plotops.get('color', None) is not None:
      if isinstance(plotops['color'], str):
        if plotops['color'] == 'absolute': plotops['color'] = self.speed
        elif plotops['color'] == 'u':      plotops['color'] = self.u
        elif plotops['color'] == 'v':      plotops['color'] = self.v
    try: # configure colormap options if possible (i.e. if self._bounds is an array)
      self._bounds = returnbounds(plotops['color'], self.fmt.bounds)
      if plotops['cmap'] is None: plotops['cmap'] = plt.cm.jet
      plotops['cmap'] = get_cmap(plotops['cmap'], N=len(self._bounds)-1)
      self._cmap = plotops['cmap']
      if not self.fmt.streamplot:
        plotops['norm'] = mpl.colors.BoundaryNorm(self._bounds, plotops['cmap'].N)
        self._norm = plotops['norm']
        self.data = plotops.pop('color')
        args = (self.lon2d, self.lat2d, self.u, self.v, self.data)
      else: args = (self.lon2d, self.lat2d, self.u, self.v)
    except TypeError:
      self._bounds = None
      self._cmap   = None
      self._norm   = None
      args = (self.lon2d, self.lat2d, self.u, self.v)
      if 'cmap' in plotops: cmap = plotops.pop('cmap')
    if self.fmt.tight: plt.tight_layout()
      
    if plotops.get('linewidth', None) is not None:
      if isinstance(plotops['linewidth'], str):
        if plotops['linewidth'] == 'absolute': plotops['linewidth'] = self.speed/np.max(self.speed)*self.fmt.scale
        elif plotops['linewidth'] == 'u':      plotops['linewidth'] = self.u/np.max(np.abs(self.u))*self.fmt.scale
        elif plotops['linewidth'] == 'v':      plotops['linewidth'] = self.v/np.max(np.abs(self.v))*self.fmt.scale
        if not self.fmt.streamplot: plotops['linewidth'] = np.ravel(plotops['linewidth'])
    
    if self.fmt.streamplot:
      self._removeplot()
      self.plot = plt.streamplot(*args, **plotops)
    else:
      if hasattr(self, 'plot'):
        self.plot.set_UVC(*args[2:])
        self.plot.set_linewidth(plotops['linewidth'])
        self.plot.set_rasterized(plotops['rasterized'])
        if self._cmap is not None: self.plot.set_cmap(self._cmap)
        if self._norm is not None: self.plot.set_norm(self._norm)
        if 'color' in plotops: self.plot.set_color(plotops['color'])
      else:
        self.plot = plt.quiver(*args, **plotops)
      try: plotops.update({'color':self.data})
      except AttributeError: pass
      try: plotops.update({'cmap':cmap})
      except NameError: pass
    if self.fmt.plotcbar != '' or self.fmt.plotcbar != () or self.fmt.plotcbar is not None or self.fmt.plotcbar is not False:
      if self._bounds is not None: self._draw_colorbar()
  
  def update(self,todefault = False, plot = True, **kwargs):
    """Update the mapBase instance, formatoptions, variable, time and level.
    Possible key words (kwargs) are
      - all keywords as set by formatoptions (see initialization function __init__)
      - time: integer. Sets the time of the mapBase instance and reloads the data
      - level: integer. Sets the level of the mapBase instance and reloads the data
      - var: string. Sets the variable name of the mapBase instance
      - u: string. Sets the variable of the windplot instance and reloads the data
      - v: Sets the variable of the windplot instance and reloads the data
      - time, level and var as above
    """
    # set current axis
    plt.sca(self.ax)
    
    # check the keywords
    dims = {'time':self.time,'var':self.name,'level':self.level,'fname':self.fname,'u':None,'v':None, 'udata':None, 'vdata':None}
    if np.any([key not in self.fmt._default.keys() + dims.keys() for key in kwargs]): sys.exit('Unknown keywords ' + ', '.join(key for key in kwargs if key not in self.fmt._default.keys() + dims.keys()))
    
    
    # delete formatoptions which are already at the wished state
    if not todefault: kwargs = {key:value for key, value in kwargs.items() if value != self.fmt.asdict().get(key, self.fmt._default.get(key, dims.get(key,None)))}
    else:
      oldkwargs = kwargs.copy()
      kwargs = {key:kwargs.get(key, value) for key, value in self.fmt._default.items() if key != 'windplot' and getattr(self.fmt, key) != self.fmt._default[key]}
      if 'windplot' in oldkwargs: kwargs['windplot'] = oldkwargs['windplot']
      else: kwargs['windplot'] = {}
    # update plotting of cbar properties
    if 'plotcbar' in kwargs:
      if kwargs['plotcbar'] in [False, None]: kwargs['plotcbar'] = ''
      if kwargs['plotcbar'] == True: kwargs['plotcbar'] = 'b'
      cbar2close = [cbar for cbar in self.fmt.plotcbar if cbar not in kwargs['plotcbar']]
      self._removecbar(cbar2close)
        
    if 'scale' in kwargs: self._removeplot()
    
    # update basemap properties
    bmprops = {key:value for key, value in kwargs.items() if key in self.fmt._bmprops}
    if bmprops != {}: self.fmt.update(**bmprops)
    reduceprops = {key:value for key, value in kwargs.items() if key in ['reduce', 'reduceabove']}
    if reduceprops != {}: self.fmt.update(**reduceprops)
    # update mapobject dimensions and reinitialize
    newdims = {key:value for key, value in kwargs.items() if key in dims.keys() and key != 'var'}
    if newdims != {} or bmprops != {} or reduceprops != {}: self._reinit(**newdims)
    if bmprops != {} and plot: self._setupproj()
    
    # handle streamplot changing
    if 'streamplot' in kwargs or 'enable' in kwargs: self._removeplot()
    
    # update rest
    self.fmt.update(**{key:value for key, value in kwargs.items() if key not in self.fmt._bmprops + dims.keys()})
    if 'meridionals' in kwargs or 'merilabelpos' in kwargs:
      keys = self.basemapops['meridionals'].keys()
      for key in keys: del self.basemapops['meridionals'][key]
      if plot: self.basemapops['meridionals'] = self.mapproj.drawmeridians(self.fmt._meriops['meridionals'], **{key:value for key, value in self.fmt._meriops.items() if key != 'meridionals'})
    if 'parallels' in kwargs or 'paralabelpos' in kwargs:
      keys = self.basemapops['parallels'].keys()
      for key in keys: del self.basemapops['parallels'][key]
      self.fmt.parallels = self.fmt.parallels
      if plot: self.basemapops['parallels'] = self.mapproj.drawparallels(self.fmt._paraops['parallels'], **{key:value for key, value in self.fmt._paraops.items() if key != 'parallels'})
    
    
    if plot: self._configuretitles()
    if plot and self.fmt.enable: self.make_plot()
    
  
  def _moviedata(self, times, **kwargs):
    """generator to get the data for the movie"""
    for time in times:
      yield (time,
             self._mask_to_region(self._shift(self._get_data(self.uname, time, self.level))), # udata
             self._mask_to_region(self._shift(self._get_data(self.vname, time, self.level))), # vdata
             {key:value[times.index(time)] for key, value in kwargs.items()}) # formatoptions
  
  def _runmovie(self, args):
    """Function to update the movie with args from self._moviedata"""
    self.update(time=args[0], udata=args[1], vdata=args[2],**args[-1])
  
  def _removeplot(self):
    """Removes the plot from the axes and deletes the plot property from the instance"""
    if hasattr(self,'plot'):
      if self.fmt.streamplot:
        # remove lines
        self.plot.lines.remove()
        # remove arrows
        keep = lambda x: not isinstance(x, mpl.patches.FancyArrowPatch)
        self.ax.patches = [patch for patch in self.ax.patches if keep(patch)]
      else:
        self.plot.remove()
      del self.plot
  
  def _reduceuv(self, perc=50, pctl=0):
    """reduces resolution of u and v to perc of original resolution if the mean is larger
    than the value of the given percentile pctl"""
    # reset speed
    try:        len(perc)
    except TypeError:     perc=[perc]*len(np.shape(self.speed))
    step=np.ceil(np.array(np.shape(self.speed))/(np.array(np.shape(self.speed))*perc/100.0)).astype(int)
    halfstep0=np.ceil(step/2.0).astype(int)
    if pctl == 0: pctl = np.min(self.speed)
    else:         pctl = np.percentile(self.speed,pctl)
    for i in xrange(0,len(self.speed),step[0]):
      for j in xrange(0,len(self.speed[i]), step[1]):
        halfstep=[0,0]
        if i+step[0] >= np.shape(self.speed)[0]: stepx = np.shape(self.speed)[0]-i
        else: stepx=step[0]; halfstep[0]=halfstep0[0]
        if j+step[1] >= np.shape(self.speed)[1]: stepy = np.shape(self.speed)[1]-j
        else: stepy=step[1]; halfstep[1]=halfstep0[1]
        if self.weights is not None: w=self.weights[i:i+stepx, j:j+stepy]
        else: w=None
        
        compute=False
        for c in np.ravel(self.speed.mask[i:i+stepx, j:j+stepy]):
          if not c: compute=True
        if compute:
          if np.average(self.speed[i:i+stepx, j:j+stepy], weights=w) >= pctl:
            self.u[i+halfstep[0], j+halfstep[1]] = np.average(self.u[i:i+stepx, j:j+stepy], weights=w)
            mask=np.ma.make_mask(np.ones(shape=(stepx,stepy))); mask[halfstep[0],halfstep[1]]=False
            self.u.mask[i:i+stepx, j:j+stepy]=mask
            self.v[i+halfstep[0], j+halfstep[1]] = np.average(self.v[i:i+stepx, j:j+stepy], weights=w)
            mask=np.ma.make_mask(np.ones(shape=(stepx,stepy))); mask[halfstep[0],halfstep[1]]=False
            self.v.mask[i:i+stepx, j:j+stepy]=mask
  
  def close(self,*args):
    """Close the mapBase instance.
    Arguments may be
     - 'data': To delete all data (but not close the netCDF4.MFDataset
     instance)
     - 'figure': To close the figure and the figures of the colorbars.
    Without any arguments, everything will be closed (including the
    netCDF4.MFDataset instance) and deleted.
    """
    if 'data' in args or args is ():
      del self.u
      del self.v
      del self.lon2d
      del self.lat2d
      del self.lon
      del self.lat
    if 'figure' in args or args is ():
      plt.close(self.ax.get_figure())
      self._removecbar(['sh','sv'])
    if args == (): del self.nco
      
  def asdict(self):
    """Returns a dictionary containing the current formatoptions and (if the time,
    level or name changed compared to the original initialization) the time, level
    or name"""
    fmt = self.fmt.asdict()
    if self.time != self.timeorig:       fmt.update({'time':self.time})
    if self.level != self.levelorig:     fmt.update({'level':self.level})
    if self.name != self.nameorig:       fmt.update({'var':self.name})
    if self.uname != self.unameorig:     fmt.update({'u':self.uname})
    if self.vname != self.vnameorig:     fmt.update({'v':self.vname})
    return fmt
  # ------------------ modify docstrings here --------------------------
  __init__.__doc__   = __init__.__doc__ + '\n' + '\n'.join((key+':').ljust(20) + get_docs('wind')[key] \
                       for key in sorted(get_docs('wind').keys())) + '\nFurther keyword arguments (kwargs) inherited by mapBase object are:\n' + mapBase.__init__.__doc__
# ------------------ modify docstrings on module level here --------------------------
update.__doc__ = update.__doc__ + '\n' + maps.update.__doc__
get_fmtkeys.__doc__ = maps.get_fmtkeys.__doc__
show_fmtkeys.__doc__ = maps.show_fmtkeys.__doc__
