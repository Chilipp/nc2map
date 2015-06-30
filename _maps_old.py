# -*- coding: utf-8 -*-
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import izip, chain, permutations, product
from collections import OrderedDict
from mapos import mapBase, fieldplot, windplot, returnbounds, round_to_05
from _ncos import ncos
from _mapsproperties import mapsproperties
import formatoptions
from formatoptions import fmtBase
from _cmap_ops import get_cmap, CbarManager
from evaluators import ViolinEval

_props = mapsproperties()

evaluatorsdict = {  # dictionary with evaluator classes
    'violin': ViolinEval
    }

currentmap = None
openmaps = []


def gcm():
  """Return the current maps instance"""
  return currentmap


def scm(mymaps):
  """Sets the current maps instance"""
  global currentmap
  currentmap = mymaps


def close():
  """close all open maps instances"""
  global openmaps
  for mymaps in openmaps: mymaps.close()
  openmaps = []


def update(fmt={}, add = True, delete = True, todefault = False, **kwargs):
  """Shortcut to the current maps instance update function"""
  currentmap.update(fmt,add,delete,todefault, **kwargs)


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
    - get_labels:    returns names, times, etc. of the given instances
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
  # ------------------ define properties here -----------------------

  # mapdata dictionary property
  maps           = _props.default('maps', """list containing mapBase instances from the initialization""")
  evaluators     = _props.default('evaluators', """List containing the evaluator instances of the maps instance""")
  fname          = _props.default('fname', """Name of the nc-file""")
  names          = _props.names('names', """List of tuples (name, var, time, level) for each mapBase instance""")
  vlst           = _props.vlst('vlst', """List of variables""")
  times          = _props.times('times', """List of time steps""")
  levels         = _props.levels('levels', """List of levels""")
  subplots       = _props.subplots('subplots', """List of subplots""")
  nco            = _props.nco('nco', """netCDF4.MFDataset instance of ncfile""")


  def __init__(self, ncfile, names=None, vlst = None,  times = 0, levels = 0, ax = (1,1), sort = 'vtl', fmt = None, timenames = ['time'], levelnames = ['level', 'lvl', 'lev'], lon=['lon', 'longitude', 'x'], lat=['lat', 'latitude', 'y'], windonly=False, onecbar = False, u=None, v=None, figsize = None, plot=True):
    """
    Input:
      - ncfile: string or 1D-array of strings. Path to the netCDF-file containing the
        data for all variables. Filenames may contain wildcards (*, ?, etc.) as suitable
        with the Python glob module (the netCDF4.MFDataset is used to open the nc-file).
        You can even give the same netCDF file multiple times, e.g. for making one figure
        with plots for one variable, time and level but different regions.
      - names: string, tuple or list of those. This sets up the unique identifier for each
          mapBase instance. If None, they will be called like mapo0, mapo1, ... and variables
          will be chosen as defined by vlst, times and levels (in other words
          N=len(vlst)*len(times)*len(levels) mapBases will be created with the names
          mapo0, ..., mapoN. If names is not None, it can be either a list of tuples or strings:
            -- string: Names of the mapBase instances (or only one string if only one mapBase
                 instance)
            -- tuple: (<name>, <var>, <time>, <level>), where <name> is the name of the mapBase
                 instance (as string), <var> the variable name (as string), <time> the timestep
                 (as integer) and <level> the level (as integer (will not be used if no level
                 dimension found in netCDF file)). This tuple defines directly which variables at
                 which timestep and level to plot and how to name the mapBase instance
      - vlst: string or 1D-array of strings. List containing all variables which
        shall be plotted or only one variable. The given strings names must correspond to
        the names in <ncfile>. If None, all variables which are not declared as dimensions
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
      - fmt: dictionary (Default: None). Dictionary controlling the format of the plots.
        Syntax is as follows:
        fmt = {['<<<var>>>':{
                                       ['t<<<time>>>': {
                                                       ['l<<<level>>>': {'keyword': ..., ...}]
                                                       [, 'keyword': ...,...]
                                                       }]
                                       [, 'l<<<level>>>: {'keyword': ..., ...}]
                                       [, 'keyword': ..., ...]
                                       }]
                         [, 't<<<time>>>': {
                                         ['l<<<level>>>': {'keyword': ..., ...}]
                                         [, 'keyword': ...,...]
                                         }]
                         [, 'l<<<level>>>: {'keyword':..., ...}]
               [, <<<name>>>: {'keyword':..., ...}]
               [, 'keyword':..., ...]
                         }.
        Seems complicated, but in fact rather simple considering the following rules:
          -- Formatoptions are set via 'keyword':value (for possible keywords, see below).
          -- Time and level specific keywords are put into a dictionary indicated by the key
             't<<<time>>>' or 'l<<<level>>>' respectively (where <<<time>>> and <<<level>>>
             is the number of the time, and or level).
          -- To set default formatoptions for each map: set the keyword in the upper most hierarchical
             level of formatoptions (e.g. fmt = {'plotcbar':'r'}).
          -- To set default formatoptions for each variable, times or level separately set the keyword
             in the second hierarchical level of formatoptions (e.g. fmt = {'t4':{'plotcbar:'r'}}
             will only change the formatoptions of maps with time equal to 4,
             fmt = {'l4':{'plotcbar:'r'}} will only change formatoptions of maps with level
             equal to 4).
          -- To set default options for a specific variable and time, but all levels: put them in the 3rd
             hierarchical level of formatoptions (e.g. fmt = {<<<var>>>:{'t4':{'plotcbar':'r'}}}
             will only change the formatoptions of each level corresponding to variable <<<var>>> and
             time 4). Works the same for setting default options for specific variable and level, but all
             times.
          -- To set a specific key for one map, just set
             fmt = {<<<var>>>: {'t<<<time>>>': {'l<<<level>>>': {'plotcbar: 'r', ...}}}}
             or directly with the name of the mapBase instance (see names keyword)
             fmt = {<<<name>>>: {'plotcbar': 'r', ...}}
             .

        The formatoption keywords are:
      """
    # docstring is extended below
    global currentmap
    global openmaps
    currentmap = self
    openmaps = openmaps + [self]

    self.maps = []
    self.evaluators = []
    self._cbars = []
    self._ncos = []
    try:
      self.fname = glob.glob(ncfile)
    except TypeError:
      self.fname = ncfile
    self.nco = self.fname
    self.lonnames  = lon
    self.latnames  = lat
    self.plot = plot
    self.timenames = timenames
    self.levelnames= levelnames
    self._dims      = {'lon':lon, 'lat':lat, 'time':timenames, 'level':levelnames}
    if vlst is None and not windonly:
      self.vlst = [str(key) for key in self.nco.variables.keys() if key not in lon+lat+timenames+levelnames]
    else:
      if isinstance(vlst, str):   vlst = [vlst]
      self.vlst = vlst
    if isinstance(times, int):  times = [times]
    if isinstance(levels, int): levels = [levels]
    self.levels = levels
    self.times = times
    self.figsize = figsize
    self.sort = sort
    self.u  = u
    self.v  = v
    self.windonly = windonly
    self.names = self._setupnames(names, self.vlst, self.times, self.levels, self.sort)
    self.subplots = (ax, len(self.names))
    self._setupfigs(self.names, self._setupfmt(fmt), self.subplots, self.nco)

    if plot:
      print("Setting up projections...")
      for mapo in self.get_maps(): mapo._setupproj()

      print("Making plots...")
      self.make_plot()

      for fig in self.get_figs():
        names, vlst, times, levels, long_names, units = self.get_labels(fig)
        fig.canvas.set_window_title('Figure ' + str(fig.number) + ': Variable ' + ','.join(var for var in vlst) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))

      if onecbar is not False:
        if onecbar is True: self.update_cbar(*(dict(zip(['names', 'vlst','times','levels'], self.get_labels(fig)[:4])) for fig in self.get_figs()), add = False, delete = False)
        elif isinstance(onecbar, dict): self.update_cbar(onecbar, add=False, delete = False)
        else: self.update_cbar(*onecbar, add = False, delete = False)

    self._namesfrominit = [nametuple[0] for nametuple in self.names]
    # old fmts (for function undo)
    self._fmt = [self.asdict('maps','cbars')]

    # future fmts (for function redo)
    self._newfmt = []

  def evaluate(self, evalname, *args, **kwargs):
    """Perform and evaluation on mapBase instances. kwargs depend on the chosen
    evaluator. See method eval_doc for documentation of each evaluator.
    Possible evaluators are
     - """
    # docstring is extended below
    self.evaluators.append(evaluatorsdict[evalname](*args, mapsin=self, **kwargs))
    return self.evaluators[-1]

  def eval_doc(self, evalname):
    """Shows the documentation of the evaluator. Possible evaluator names are
     - """
    # docstring is extended below
    help(evaluatorsdict[evalname])

  def make_plot(self, *args, **kwargs):
    """makes the plot of mapBase instances. Don't use this function but rather
    the update function to make plots"""
    for mapo in self.get_maps(*args, **kwargs): mapo.make_plot()

  def get_maps(self, names=None, vlst=None, times=None, levels=None):
    """Returns 1D-numpy array containing the mapBase instances stored in maps.
       Input:
         - names:   string of 1D array of strings (Default: None). If not None,
                    but string or list of strings. Only the specified mapBase
                    instances are returned.
         - vlst:    string or 1D array of strings (Default: None). If not None,
                    the strings need to be the name of a variable contained in
                    the maps instance
         - times:   same as vlst but for times (as integers!)
         - levels:  same as vlst but for levels (as integers!)
       Output:
         - list of mapBase instances
    """
    if names is None: names = np.unique([mapo.name for mapo in self.maps]).tolist()
    elif isinstance(names, str): names = [names]
    if vlst is None:   vlst = np.unique([mapo.var for mapo in self.maps]).tolist()
    elif isinstance(vlst, str): vlst = [vlst]
    if times is None: times = np.unique([mapo.time for mapo in self.maps]).tolist()
    elif isinstance(times, int): times = [times]
    if levels is None: levels= np.unique([mapo.level for mapo in self.maps]).tolist()
    elif isinstance(levels, int): levels = [levels]

    return [mapo for mapo in self.maps if mapo.name in names and mapo.var in vlst and mapo.time in times and mapo.level in levels]

  def get_winds(self, *args, **kwargs):
    """Returns 1D-numpy array containing the windplot instances stored in maps
    if windonly is not True (in this case: use get_maps()).
    Keyword arguments are determined by function get_maps (i.e. names, vlst, times and levels)."""
    if not self.windonly: return [mapo.wind for mapo in self.get_maps(*args, **kwargs) if mapo.wind is not None]
    else: return []

  def get_figs(self, *args, **kwargs):
    """Returns dictionary containing the figures used in the maps instance
    as keys and a list with the included mapBase instances as value.
    Without any kwargs and args, return all figures from the maps instance.
    Otherwise you can either give mapBase objects or figures as arguments
    or specify one or each of the following key words to return all figures
    related to the specified variable, time or level
     - names:  string or list of strings. Specify names of mapBase instances
               to return
     - vlst:   string or list of strings. Specify variables to return
     - times:  integer or list of integers. Specify times to return
     - levels: integer or list of integers. Specify levels to return
     Example: self.get_figs(vlst='temperature') will return a dictionary with
     all figures that have mapBase instances with the variable 'temperature' as
     subplot as keys and a list with the corresponding mapBase instances as values.
     If 'wind' in args: the mapBase instances will be the corresponding wind-
    plot instances to the figure.
    """
    if 'wind' in args:
      get_func = self.get_winds
      args = tuple(arg for arg in args if arg != 'wind')
    else:
      get_func = self.get_maps
    if args == ():
      maps = get_func(**kwargs)
      figs = OrderedDict()
      append = True
    elif all(isinstance(arg, mapBase) for arg in args):
      maps = args
      figs = OrderedDict()
      append = True
    elif all(isinstance(arg, mpl.figure.Figure) for arg in args):
      figs = OrderedDict([(arg, []) for arg in args])
      maps = get_func()
      append = False
    else: raise TypeError("Wrong type of obj! Object must either be 'maps' or 'winds'!")
    for mapo in maps:
      if mapo.ax.get_figure() not in figs and append: figs[mapo.ax.get_figure()] = []
      if mapo.ax.get_figure() in figs: figs[mapo.ax.get_figure()].append(mapo)
    return figs

  def _replace(self, txt, fig, delimiter='-'):
    """Function to replace strings by objects from fig

    Input:
      - txt: string where <<<var>>>, <<<time>>>, <<<name>>>, <<<level>>>,
             <<<longname>>>, <<<units>>> shall be replaced by the corresponding
             attributes of the mapBase object plotted in fig.
      - fig: figure or list of figures
      - delimiter: string which shall be used for separating values, if more
             than one mapBase instance is inside the figure
    Returns:
      - string without <<<var>>>, <<<time>>> and so on
    """
    if isinstance(fig, mpl.figure.Figure):
      fig = [fig]
    values = self.get_labels(*fig)
    values = [map(str, value) for value in values]
    wildcards = ['<<<name>>>', '<<<var>>>', '<<<time>>>', '<<<level>>>', '<<<longname>>>', '<<<unit>>>']
    for wildcard, value in izip(wildcards, values):
      txt = txt.replace(wildcard, delimiter.join(value))
    return txt

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
      - names: To save only the figures with the mapBase instances
               specified in names
      - vlst: To save only the figures with variables specified in vlst
      - times: To save only the figures with times specified in times
      - levels: To save only the figures with levels specified in levels
      - any other keyword as specified in the pyplot.savefig function.
      These are:
    """
    # the docstring is extended by the plt.savefig docstring below
    from matplotlib.backends.backend_pdf import PdfPages
    saveops = {key:value for key, value in kwargs.items() if key not in ['names', 'vlst','times','level']}
    if 'tight' in args: saveops['bbox_inches'] = 'tight'; args = tuple([arg for arg in args if arg != 'tight'])
    kwargs = {key:value for key, value in kwargs.items() if key in ['names', 'vlst','times','level']}
    if args == ():
      figs = self.get_figs(**kwargs).keys()
    elif isinstance(args[0], mapBase):
      names, vlst, times, levels, long_names, units = self.get_labels(*args)
      figs = self.get_figs(vlst=names, times=times, levels=levels)
    else:
      figs = args
    if isinstance(output, str):
      if output[-4:] in ['.pdf', '.PDF']:
        output = self._replace(output, figs)
        with PdfPages(output) as pdf:
          for fig in figs: pdf.savefig(fig, **saveops)
          print('Saving plot to ' + output)
        return
      else:
        strout = output
        output = []
        for fig in figs:
          names, vlst, times, levels, long_names, units = self.get_labels(fig)
          output = self._replace(output, fig)
    else: pass
    # test output
    try:
      if len(np.shape(output)) > 1: raise ValueError('Output array must be a 1D-array!')
      if len(figs) != len(output): raise ValueError('Length of output names (' + str(len(output)) + ') does not fit to the number of figures (' + str(len(figs)) + ').')
    except TypeError:
      raise TypeError('Output names must be either a string or an 1D-array of strings!')
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
    fmt.update({key:value for key, value in kwargs.items() if key not in ['names', 'vlst', 'times','levels']})
    fmt = self._setupfmt(fmt)
    maps = self.get_maps(**{key: value for key, value in kwargs.items() if key in ['names', 'vlst', 'times', 'levels']})
    # update maps
    for mapo in maps: mapo.update(todefault=todefault, **fmt[mapo.name])
    # update figure window title and draw
    for cbar in self.get_cbars(*maps): cbar._draw_colorbar()
    for fig in self.get_figs(*maps):
      plt.figure(fig.number)
      names, vlst, times, levels, long_names, units = self.get_labels(fig)
      fig.canvas.set_window_title('Figure ' + str(fig.number) + ': Variable ' + ','.join(var for var in vlst) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))
      plt.draw() # if it is part of a cbar, it has already been drawn above
    # add to old fmts
    if add: self._fmt.append(self.asdict('maps','cbars'))
    # delete new fmts
    if delete: self._newfmt = []
    del fmt

  def get_names(self, *args, **kwargs):
    """return a list of tuples (name, var, time, level) for each map object
    *args and **kwargs are determined by get_maps method
    """
    names = []
    for mapo in self.get_maps(*args, **kwargs):
      names.append((mapo.name, mapo.var, mapo.time, mapo.level))
    return names

  def nextt(self,*args,**kwargs):
    """takes the next time step for maps specified by args and kwargs
    (same syntax as get_maps. Use 'wind' as an argument if only winds
    shall be updated."""
    if not self.plot: plot=False
    else: plot=True
    if 'wind' in args: maps = self.get_winds(*(arg for arg in args if arg != 'wind'), **kwargs)
    else: maps = self.get_maps(*args, **kwargs)
    for mapo in maps: mapo.update(time = mapo.time + 1, plot=plot)
    if self.plot:
        for fig in self.get_figs(*maps): plt.figure(fig.number); plt.draw()

  def prevt(self,*args,**kwargs):
    """takes the previous time step for maps specified by args and kwargs
    (same syntax as get_maps. Use 'wind' as an argument if only winds shall
    be updated"""
    if not self.plot: plot=False
    else: plot=True
    if 'wind' in args: maps = self.get_winds(*(arg for arg in args if arg != 'wind'), **kwargs)
    else: maps = self.get_maps(*args, **kwargs)
    for mapo in maps: mapo.update(time = mapo.time - 1, plot=plot)
    if self.plot:
        for fig in self.get_figs(*maps): plt.figure(fig.number); plt.draw()


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
    if self._fmt == []: raise ValueError('Impossible option')
    if num > 0 and num >= len(self._fmt)-1: raise ValueError('Too high number! Maximal number is ' + str(len(self._fmt)-1))
    elif num < 0 and num < -len(self._fmt): raise ValueError('Too small number! Minimal number is ' + str(-len(self._fmt)+1))
    if figsize is not None: self.figsize = figsize
    # try to save ncos
    nametuples = self.names
    enhancednametuples = [list(nametuple) for nametuple in self.names]
    for nametuple in enhancednametuples:
      nametuple.append(self.get_maps(*nametuple)[0].nco)
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
    if ax is not None:
      del self.subplots
      self.subplots = (ax, len(enhancednametuples))
    else:
      for ax in self.subplots: ax.clear
    # set new figures
    # change names sorting
    if sort is not None:
      self.sort = sort
      nametuples = self._setupnames(nametuples, self.vlst, self.times, self.levels, self.sort)
    # set up figures
    for nametuple in enhancednametuples:
      subplot = self.subplots[nametuples.index(tuple(nametuple[:-1]))]
      self._setupfigs([tuple(nametuple[:-1])], fmt=self._fmt[num-1][0], subplots=[subplot], nco=nametuple[-1], fromscratch=fromscratch)

    if self.plot:
      print("Setting up projections...")
      for mapo in self.get_maps(): mapo._setupproj()

      print("Making plots...")
      self.make_plot()

      for fig in self.get_figs():
        plt.figure(fig.number)
        names, vlst, times, levels, long_names, units = self.get_labels(fig)
        fig.canvas.set_window_title('Figure ' + str(fig.number) + ': Variable ' + ','.join(var for var in vlst) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))
        plt.draw()


    if self._fmt[num-1][1] != []: self.update_cbar(*self._fmt[num-1][1], add = False, delete = False)
    # shift to new fmt
    if num != 0:
      self._newfmt = self._fmt[num:] + self._newfmt
      if num < 0: self._fmt.__delslice__(len(self._fmt)+num, len(self._fmt))
      else: self._fmt.__delslice__(num,len(self._fmt))

  def undo(self, num=-1):
    """Undo the changes made. num gives the number of changes to go back."""
    if self._fmt == [] or len(self._fmt) == 1: raise ValueError('Impossible option')
    if num > 0 and num >= len(self._fmt)-1: raise ValueError('Too high number! Maximal number is ' + str(len(self._fmt)-1))
    elif num < 0 and num < -len(self._fmt): raise ValueError('Too small number! Minimal number is ' + str(-len(self._fmt)+1))
    if self._fmt[num-1][1] == []: self.removecbars()
    self.update(self._fmt[num-1][0], add=False, delete=False, todefault = True)
    if self._fmt[num-1][1] != []: self.update_cbar(*self._fmt[num-1][1], add=False, delete=False, todefault = True)
    # shift to new fmt
    self._newfmt = self._fmt[num:] + self._newfmt
    if num <= 0: self._fmt.__delslice__(len(self._fmt)+num, len(self._fmt))
    else: self._fmt.__delslice__(num,len(self._fmt))

  def redo(self, num=1):
    """Redo the changes made. num gives the number of changes to use."""
    if self._newfmt == []: raise ValueError('Impossible option')
    if num > 0 and num > len(self._newfmt): raise ValueError('Too high number! Maximal number is ' + str(len(self._newfmt)))
    elif num < 0 and num < -len(self._newfmt): raise ValueError('Too small number! Minimal number is ' + str(-len(self._newfmt)-1))
    if self._newfmt[num-1][1] == []: self.removecbars()
    self.update(self._newfmt[num-1][0], add=False, delete=False, todefault = True)
    if self._newfmt[num-1][1] != []: self.update_cbar(*self._newfmt[num-1][1], add=False, delete=False, todefault = True)
    # shift to old fmt
    self._fmt = self._fmt + self._newfmt[:num]
    if num > 0: self._newfmt.__delslice__(0,num)
    else: self._newfmt.__delslice__(0,len(self._newfmt)+num)

  def show(self):
    """shows all open figures (without blocking)"""
    plt.show(block=False)

  def close(self,*args,**kwargs):
    """Without any args and kwargs, close all open figure from the maps object,
    delete all mapBase objects and close the netCDF4.MFDataset.
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
      raise KeyError('Unknown argument ' + ', '.join(arg for arg in args if arg not in ['data','figure']) + ". Possibilities are 'data' and 'figure'.")
    if self.maps == []: return
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
      for mapo in self.get_maps(**kwargs):
        self.maps.remove(mapo)
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
    if kwargs != {}: newops = list(args) + [kwargs]
    else: newops = list(args)

    # first set colorbars
    for cbarops in newops:
      if 'windplot' in cbarops: args = tuple(['wind']); cbarops.update(cbarops.pop('windplot')); wind=True; get_func = self.get_winds
      else: args = (); wind = False; get_func=self.get_maps
      dims = {key:cbarops.get(key, None) for key in ['names', 'vlst','levels','times']}
      # if no colorbars are set up to now and no specific var, time and level options are set, make colorbars for each figure
      if all(value is None for key, value in dims.items()) and self._cbars == []:
        figs = self.get_figs(*args)
        for fig in figs:
          self._cbars.append(CbarManager(maps=figs[fig], fig=[fig], cbar={}, fmt=fmtBase(**{key:value for key,value in cbarops.items() if key not in ['names', 'times','vlst','levels']}), mapsobj = self, wind=wind))
      # now update colorbar objects or create them if they are not existent
      cbars = self.get_cbars(*args,**dims)
      if cbars == []:
        self._cbars.append(CbarManager(maps=get_func(**dims),fig = self.get_figs(*args, **dims).keys(), cbar={}, fmt=fmtBase(**{key:value for key,value in cbarops.items() if key not in dims.keys()}), mapsobj = self, wind=wind))
        cbars = [self._cbars[-1]]

      # now draw and update colorbars
      for cbar in cbars:
        # delete colorbars
        if not todefault: cbarops = {key:value for key,value in cbarops.items() if key not in ['names', 'times', 'vlst', 'levels']}
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

    if add: self._fmt.append(self.asdict('maps','cbars'))
    if delete: self._newfmt = []

  def get_cbars(self,*args,**kwargs):
    """Function to return the CbarManager related to the given input
    Input:
      - Arguments (args) may be instances of mapBase, figures or
        CbarManagers
      - Keyword arguments may be the one defined by get_maps
        (names, vlst, times, levels).
    Output:
      list of CbarManager instances"""
    maps = []
    args = list(args)
    cbars = [cbar for cbar in args if isinstance(cbar, CbarManager)]
    if args == []: maps = self.get_maps(**kwargs)
    elif args == ['wind']: maps = self.get_winds(**kwargs)
    #elif all(isinstance(arg, mpl.figure.Figure) for arg in args if arg != 'wind'): maps = [mapo for fig, mapo in self.get_figs(*args).items()]
    else:
      figs = self.get_figs(*(arg for arg in args if arg == 'wind'))
      for fig in figs:
        if fig in args: maps = maps + figs[fig]; args.remove(fig)
      maps = maps + list(arg for arg in args if not isinstance(arg,CbarManager))
    cbars = cbars + [cbar for cbar in self._cbars if any(mapo in cbar.maps for mapo in maps)]
    return cbars

  def removecbars(self,*args, **kwargs):
    """Function to remove CbarManager instances from the plot.
    args and kwargs are determined by the get_cbars function, i.e.
    Input:
      - Arguments (args) may be instances of mapBase, figures or
        CbarManagers
      - Keyword arguments may be the one defined by get_maps
        (vlst, times, levels).
    """
    cbars = self.get_cbars(*args,**kwargs)
    for cbar in cbars:
      for mapo in cbar.maps: mapo.fmt._enablebounds = True
      cbar._removecbar()
      self._cbars.pop(self._cbars.index(cbar))

  def get_labels(self,*args):
    """Function to return the descriptions of the specific input
    Input:
      - Arguments (args) may be instances of mapBase, figures or
        CbarManagers and
    Output: names, vlst, times, levels, long_names, units
      - names:      list of mapBase instance names of the input (without duplicates)
      - vlst:       list of variables contained in the input (without duplicates)
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
      attrs = ['name', 'var','time','level', 'long_name', 'units']
      tmpout = [[getattr(mapo, attr) for mapo in maps] for attr in attrs]
      out = [[] for count in xrange(len(tmpout))]
      for iattr in xrange(len(tmpout)):
        for attr in tmpout[iattr]:
          if attr not in out[iattr]: out[iattr].append(attr)
      return out

  def _setupnames(self, names, vlst, times, levels, sort):
    # initialize names
    newnames = []
    nnames = len(vlst)*len(times)*len(levels)
    # if names is none, initialize unique own names
    if names is None:
      if self.maps != []:
        existingnames = [nametuple[0] for nametuple in self.names]
      else:
        existingnames = []
      names = []
      icounter = 0
      i = 0
      while i < nnames:
        if 'mapo%i' % icounter not in existingnames:
          names.append('mapo%i' % icounter)
          i += 1
          icounter += 1
        else:
          icounter += 1
      #names = ['mapo%i' % i for i in xrange(nnames)]
    # if tuple, means only one mapo
    if isinstance(names, tuple):
      if len(names) != 4:
        raise ValueError("Either wrong length (has to be 4) of tuple or wrong type (must not be a tuple) of names!")
      newnames = [names]
      vlst = [names[1]]
      times = [names[2]]
      levels = [names[3]]
    else:
      if isinstance(names, str):
        names = [names]
      if isinstance(names[0], tuple):
        if any(len(name) != 4 for name in names):
          raise ValueError("Wrong length of name tuples (has to be 4: (name, var, time, level))!")
        newnames = names
        vlst = np.unique([name[1] for name in names]).tolist()
        times = np.unique([name[2] for name in names]).tolist()
        levels = np.unique([name[3] for name in names]).tolist()
      else:
        if len(names) != len(vlst)*len(times)*len(levels):
          raise ValueError("Names has the wrong length (%i)! Expected %i!" % (
            len(names), len(vlst)*len(times)*len(levels)))
        for name, vtl in izip(names, product(vlst, times, levels)):
          newnames.append((name, vtl[0], vtl[1], vtl[2]))
    # --- resort names ---
    if sort is not None:
      if tuple(sort) in permutations(['t','v','l']):
        sortlist = list(sort)
        sortlist[sortlist.index('v')] = [var for var in vlst]
        sortlist[sortlist.index('t')] = [time for time in times]
        sortlist[sortlist.index('l')] = [level for level in levels]
        sortlist = list(product(*sortlist))
        names = newnames[:]
        newnames = []
        for sortitem in sortlist:
          var = sortitem[sort.index('v')]
          time = sortitem[sort.index('t')]
          level = sortitem[sort.index('l')]
          for nametuple in names:
            if nametuple[1:] == (var, time, level):
              newnames.append(nametuple)
      # if not in the style of 'tvl', we assume that the names of mapBases are used for sorting
      else:
        names = newnames[:]
        newnames = []
        for name in sort:
          for nametuple in names:
            if nametuple[0] == name:
              newnames.append(nametuple)
    return newnames

  def _setupfigs(self, names, fmt, subplots, nco=None, fromscratch=True):
    """set up the figures and map objects of the maps object"""
    windonly = self.windonly
    u        = self.u
    v        = self.v

    if windonly: mapo = windplot
    else: mapo = fieldplot
    if len(subplots) != len(names): raise ValueError('Number of given axes (' + str(len(subplots)) + ') does not fit to number of mapBase instances (' + str(len(names)) + ')!')
    # setup axes
    isubplot = 0
    for name, var, time, level in names:
      if fromscratch:
        self.maps.append(mapo(self.fname, name=name, var=str(var), time=time, level=level, ax=subplots[isubplot], fmt=fmt[name], nco=nco, timenames = self.timenames, levelnames = self.levelnames, lon=self.lonnames, lat=self.latnames, ax_shapes=self._subplot_shape, ax_num=self._subplot_nums[isubplot], mapsin = self, u = u, v = v))
      else:
        mapo = self.get_maps(names=name, vlst=var, times=time, levels=level)[0]
        if hasattr(mapo, 'cbar'): mapo._removecbar(); del mapo.cbar
        if hasattr(mapo,'wind') and mapo.wind is not None: mapo.wind._removeplot()
        mapo.ax = subplots[isubplot]
        mapo._subplot_shape=self._subplot_shape
        mapo._ax_num=self._subplot_nums[isubplot]
        mapo.update(plot=False, todefault = True, **fmt[mapo.name])
      isubplot+=1
    return

  def make_movie(self, output, fmt={}, onecbar = {}, steps = None, *args, **kwargs):
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
      - onecbar: Dictionary or list of dictionaries (Default: {}). Same
        settings as for update_cbar function but (like fmt) with values
        of formatoption keywords being 1D-arrays with same length as number
        of steps
      - steps: List of integers or None. If None, all timesteps in the
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
        names, vlst, times, levels, long_names, units = self.get_labels(fig)
        out = self._replace(output, fig)
      else:
        out = output[i]
      maps = figs[fig]
      cbars = self.get_cbars(*maps)

      if steps is None:
        for timename in self.timenames:
          try:
            steps = range(self.nco.variables[maps[0].var]._shape()[self.nco.variables[maps[0].var].dimensions.index(timename)])
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
      if cbars != []: data_gen = myizip(myizip(*(mapo._moviedata(steps, **fmt[mapo.name]) for mapo in maps)), myizip(*(cbar._moviedata(steps, **onecbar) for cbar in cbars)))
      else: data_gen = myizip(*(mapo._moviedata(steps, **fmt[mapo.name]) for mapo in maps))
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
    self.update(self._fmt[-1][0], add=False, delete=False, todefault = True)
    if self._fmt[-1][1] == []:
      for cbar in self._cbars: cbar._removecbar()
    else: self.update_cbar(*self._fmt[-1][1], add=False, delete=False, todefault = True)


  def _setupfmt(self, oldfmt, names=None):
    """set up the fmt for the mapBase instances
    if names is None: use self.names"""
    # set up the dictionary for each variable
    def removedims(fmt):
      dims = list(chain(*map(lambda (name, var, t, l): (name, var, "t%i" % t, "l%i" % l), self.names + self.get_names())))
      return {key: val for key, val in fmt.items() if key not in dims}
    if names is None:
      names = self.names[:]
    if oldfmt is None:
      return {name[0]: None for name in names}
    fmt = {}
    for name, var, time, level in names:
      fmt[name] = removedims(oldfmt)
      try:
        fmt[name].update(removedims(oldfmt["l%i" % level]))
      except KeyError:
        pass
      try:
        fmt[name].update(removedims(oldfmt["t%i" % time]))
        try:
          fmt[name].update(removedims(oldfmt["t%i" % time]["l%i" % level]))
        except KeyError:
          pass
      except KeyError:
        pass
      try:
        fmt[name].update(removedims(oldfmt[var]))
        try:
          fmt[name].update(removedims(oldfmt[var]["t%i" % time]))
          try:
            fmt[name].update(removedims(oldfmt[var]["t%i" % time]["l%i" % level]))
          except KeyError:
            pass
        except KeyError:
          pass
      except KeyError:
        pass
      fmt[name].update(removedims(oldfmt.get(name, {})))
    return fmt

  def script(self, output):
    """Function to create a script named output with the current formatoptions.
    Experimental function! Please take care of bounds and colormaps in the output
    script."""
    import datetime as dt
    import pickle
    with open(output,'w') as f:
      f.write("# -*- coding: utf-8 -*-\n# script for the generation of nc2map.maps object. Time created: """ + dt.datetime.now().strftime("%d/%m/%y %H:%M") + '\n' + "import nc2map\nimport pickle\nncfile = " + str(self.fname) + "\nnames = " + str(self.names) + "\ntimenames = " + str(self.timenames) + "\nlevelnames = " + str(self.levelnames) + "\nlon = " + str(self.lonnames) + "\nlat = " + str(self.latnames) + "\nsort = '" + str(self.sort) + "'\n")

      # save dictionary to pickle object
      with open(output.replace('.py', '.pkl'), 'w') as fmtf:
        pickle.dump(self.asdict(), fmtf)
      f.write("with open('%s') as f:\n    fmt = pickle.load(f)\n" % output.replace('.py', '.pkl'))
      openstring = "mymaps = nc2map.maps(ncfile=ncfile, names=names, fmt=fmt, lon=lon, lat=lat, timenames=timenames, levelnames=levelnames, sort=sort"

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

  def addmap(self, ncfile, names=None, vlst = None,  times = 0, levels = 0, ax = (1,1), sort = 'vtl', fmt = None, onecbar = False, u=None, v=None):
    """add a mapBase instance to maps instance
    Input:
      - ncfile: Either string (or list of strings) or mapBase instance or list
          of mapBase instances. If mapBase instance or list of mapBase instances,
          all of the other keywords are obsolete. If string or list of strings:
          Path to the netCDF-file containing the data for all variables.
          Filenames may contain wildcards (*, ?, etc.) as suitable with the
          Python glob module (the netCDF4.MFDataset is used to open the
          nc-file). You can even give the same netCDF file multiple times, e.g.
          for making one figure with plots for one variable, time and level but
          different regions.

    All the rest of keywords are the same as for the init function:
    """
    # if single mapBase instance, just add it
    if isinstance(ncfile, mapBase):
      self.maps.append(ncfile)
      return
    # if many mapBase instances, just add them
    if isinstance(ncfile[0], mapBase):
      if any(not isinstance(mapo, mapBase) for mapo in ncfile):
        raise ValueError("Found mixture of objects in Input. Please use only mapBase instances or strings!")
      self.maps += np.ravel(ncfile).tolist()
      return
    # else, initialize them
    try:
      self.fname = glob.glob(ncfile)
    except TypeError:
      self.fname = ncfile
    self.nco = self.fname
    if vlst is None and not self.windonly:
      vlst = [
        str(key) for key in self.nco.variables.keys() if key not in self.lonnames+self.latnames+self.timenames+self.levelnames]
    else:
      if isinstance(vlst, str):   vlst = [vlst]
    if isinstance(times, int):  times = [times]
    if isinstance(levels, int): levels = [levels]
    nametuples = self._setupnames(names=names, vlst=vlst, times=times, levels=levels, sort=sort)
    names = [nametuple[0] for nametuple in nametuples]
    nsub0 = len(self.subplots)
    self.subplots = (ax, len(names))
    self.names.append(nametuples)
    self._setupfigs(nametuples, self._setupfmt(fmt, nametuples), self.subplots[nsub0:], self.nco)
    print("Setting up projections...")
    for mapo in self.get_maps(names=names): mapo._setupproj()
    print("Making plots...")
    self.make_plot(names=names)

    if onecbar is not False:
      if onecbar is True: self.update_cbar(*(dict(zip(['names', 'vlst','times','levels'], self.get_labels(fig)[:4])) for fig in self.get_figs()), add = False, delete = False)
      elif isinstance(onecbar, dict): self.update_cbar(onecbar, add=False, delete = False)
      else: self.update_cbar(*onecbar, add = False, delete = False)

    for fig in self.get_figs(names=names):
      names, vlst, times, levels, long_names, units = self.get_labels(fig)
      fig.canvas.set_window_title('Figure ' + str(fig.number) + ': Variable ' + ','.join(var for var in vlst) + ', time ' + ', '.join(str(time) for time in times) + ', level ' + ', '.join(str(level) for level in levels))

    # reset old fmts (for function undo)
    self._fmt = [self.asdict('maps','cbars')]

    # reset future fmts (for function redo)
    self._newfmt = []

  def asdict(self, *args, **kwargs):
    """returns the current formatoptions of all mapBase objects and
    cbarmangers as dictionary.
    Arguments may be
      - 'maps' to return only the dictionary controlling the mapBase
        instances (see formatoptions in the initialization) (Default)
      - 'frominit' to return only the dictionary of the mapBase instances
          which where created during intialization
      - 'cbars' to return only the dictionary controlling the
        CbarManager instances (see onecbar in the initialization)
    Keyword argument may be any of get_maps, i.e.
      - names
      - vlst
      - times
      - levels
    """
    # not used keyword:
    #- reduced: Boolean (Default: True). Reduces the formatoptions
    #  such that if formatoption keywords are multiply set for more
    #  than one instances (e.g. for all variables), they will be
    #  put together. As an example:
    #  {<<<var1>>>:{<<<t1>>>>:{<<<l1>>>:{<<<keyword>>>:<<<value>>>}}},
    #  <<<var2>>>:{<<<t2>>>>:{<<<l2>>>:{<<<keyword>>>:<<<value>>>}}}}
    #  will be reduced to {<<<keyword>>>:<<<value>>>} (as it is suitable
    #  for update and initialization function but shorter).
    #- initcompatible: Boolean (Default: False). Returns the diction-
    #    ary in such a way that the maps object can be reiniatlized
    #    (i.e. with the original variables, times and levels as keys).
    fmt = {}
    cbars = []
    if args == () or 'maps' in args or 'frominit' in args:
      returnfmt = True
      if 'frominit' in args:
        kwargs['names'] = self._namesfrominit
      for mapo in self.get_maps(**kwargs):
        fmt[mapo.name] = mapo.asdict()
    else: returnfmt = False
    if 'cbars' in args:
      returncbars = True
      for cbar in self.get_cbars(**kwargs):
        names, vlst, times, levels, long_names, units = self.get_labels(*cbar.maps)
        nametuples = self.get_names(names=names)
        cbars.append({key:value for key,value in cbar.fmt.asdict().items()})
        if nametuples != self.names:
          cbars[-1]['names'] = nametuples
    else: returncbars = False
    if returnfmt and returncbars: return (fmt, cbars)
    if returnfmt: return (fmt)
    if returncbars: return (cbars)

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
    if not 'wind' in args: self.get_maps()[0].show_fmtkeys(*args)
    else: self.get_winds()[0].show_fmtkeys(*args)

  # ------------------ modify docstrings here --------------------------
  __init__.__doc__   += "\n%s\n\nAnd the windplot specific options are\n\n%s" % (
      # formatoptions keywords
      '\n'.join((key+':').ljust(20) + val
                for key, val in sorted(formatoptions.get_fmtkeys().items())),
      # wind options keywords
      '\n'.join((key+':').ljust(20) + val
                  for key, val in sorted(formatoptions.get_fmtkeys('wind', 'windonly').items())))
  addmap.__doc__ += __init__.__doc__[__init__.__doc__.find('- names'):]
  output.__doc__     = output.__doc__ + plt.savefig.__doc__[plt.savefig.__doc__.find('Keyword arguments:') + len('Keyword arguments:\n'):]
  make_movie.__doc__ = make_movie.__doc__ + '    ' + FuncAnimation.save.__doc__[FuncAnimation.save.__doc__.find('*'):]
  evaluate.__doc__ += '\n - '.join(evaluatorsdict.keys())
  eval_doc.__doc__ += '\n - '.join(evaluatorsdict.keys())

# ------------------ modify docstrings on modular level here --------------------------
update.__doc__ = update.__doc__ + '\n' + maps.update.__doc__
