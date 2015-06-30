# -*- coding: utf-8 -*-
"""Module containing the BasePlot class

This class is mainly responsible for general formatting of the axes.
It does not contain any plotting features."""
import logging
import matplotlib.pyplot as plt
from itertools import izip
from collections import OrderedDict
from _mapproperties import MapProperties
from .._axes_wrapper import wrap_subplot

_props = MapProperties()


class BasePlot(object):
    """Base class to control the formatting of an axes (texts, labels, etc.)"""
    ax = _props.ax(
        'ax', """
        Axes instance of the plot""")

    def __init__(self, name, mapsin=None, ax=None, figsize=None, meta={}):
        self.name = name
        self._mapsin = mapsin
        self.figtitle  = None
        self.figsize = figsize
        self._reinitialize = 1
        self.ax = ax
        self._meta = OrderedDict(meta)
        self.texts     = {'axes':[], 'fig':[], 'data':[]}

    def _replace(self, txt, fig=False, delimiter='-'):
        """Replaces the text from self.meta and self.data.time

        Input:
          - txt: string where '%(key)s' will be replaced by the value of the
              'key' information in the MapBase.meta attribute (e.g.
              '%(long_name)s' will be replaced by the long_name of the
              variable file in the NetCDF file.
          - fig: True/False. If True and this MapBase instance belongs to a
              nc2map.MapsManager instance, the _replace method of this
              instance is used to get the meta information
          - delimiter: string. Delimter that shall be used for meta
              informations if fig is True.
        Returns:
          - string with inserted meta information
        """
        if self._mapsin is not None and fig:
            return self._mapsin._replace(txt, self.ax.get_figure(),
                                         delimiter=delimiter)
        try:
            meta = self.meta
            for key, val in meta.items():
                meta[key] = str(val)
        except AttributeError:
            return txt
        return txt % meta

    def _configureaxes(self):
        plt.sca(self.ax)
        # set color of axis
        if self.fmt.axiscolor is not None:
            for pos, col in self.fmt.axiscolor.items():
                self.ax.spines[pos].set_color(col)
        # first remove old texts which are not in the formatoptions
        for trans in self.texts:
            for oldtext in (
                    oldtext for oldtext in self.texts[trans] if all(
                        any(val != fmttext[index] for index, val
                            in enumerate(oldtext.get_position()))
                        for fmttext in self.fmt.text if fmttext[3] == trans)):
                oldtext.remove(); self.texts[trans].remove(oldtext)
        # check if an update is needed
        for text in self.fmt._textstoupdate:
            if text[3] == 'axes':
                trans = self.ax.transAxes
            elif text[3] == 'fig':
                trans = self.ax.get_figure().transFigure
            else:
                trans = self.ax.transData
            for oldtext in self.texts[text[3]]:
                if oldtext.get_position() == (text[0],text[1]):
                    oldtext.remove()
                    self.texts[text[3]].remove(oldtext)
            if len(text) == 4:
                self.texts[text[3]].append(plt.text(
                    text[0], text[1], self._replace(text[2]), transform=trans))
            elif len(text) == 5 :
                self.texts[text[3]].append(plt.text(
                    text[0], text[1], self._replace(text[2]), transform=trans,
                                                    fontdict = text[4]))

        if self.fmt.figtitle is not None and self.figtitle is not None:
            self.figtitle.set_text(self._replace(self.fmt.figtitle, fig=True,
                                                 delimiter=', '))
            self.figtitle.set_size(self.fmt._figtitleops['fontsize'])
            self.figtitle.set_weight(self.fmt._figtitleops['fontweight'])
        elif self.fmt.figtitle is not None:
            fig = self.ax.get_figure()
            self.figtitle = fig.suptitle(
                self._replace(self.fmt.figtitle, fig=True,
                              delimiter=', '), **self.fmt._figtitleops)
            if len(fig.texts) > 1:
                for text in fig.texts[1:]:
                    if text.get_position() == self.figtitle.get_position():
                        del fig.texts[fig.texts.index(text)]
        elif self.figtitle is not None:
            self.figtitle.set_text('')

        if self.fmt.title is not None:
            plt.title(self._replace(self.fmt.title), **self.fmt._titleops)
        else:
            plt.title('')

    def set_logger(self, name=None, force=False):
        """This function sets the logging.Logger instance in the MapsManager
        instance.
        Input:
          - name: name of the Logger (if None: it will be named like
             <module name>.<class name>)
          - force: True/False (Default: False). If False, do not set it if the
              instance has already a logger attribute."""
        if name is None:
            try:
                name = '%s.%s.%s' % (self.__module__, self.__class__.__name__,
                                    self.name)
            except AttributeError:
                name = '%s.%s' % (self.__module__, self.__class__.__name__)
        if not hasattr(self, 'logger') or force:
            self.logger = logging.getLogger(name)
            self.logger.debug('Initializing...')

    def get_fmtkeys(self,*args):
        """Function to return formatoptions as dictionary

        This function gives formatoption keys and their documentation as
        dictionary (shortcut to self.fmt.get_fmtkeys()). *args may be any
        formatoption keyword.
        """
        return self.fmt.get_fmtkeys(*args)

    def show_fmtkeys(self,*args):
        """Function to print formatoptions keywords and documentation

        This function prints out formatoption keys and their documentation as
        dictionary (shortcut to self.fmt.show_fmtkeys()). *args may be any
        formatoption keyword.
        """
        self.fmt.show_fmtkeys(*args)

    def _get_axes_shape(self, mode='ask'):
        """Returns the axes shape from the ax._AxesWrapper__shape attribute"""
        if not hasattr(self._ax, '_AxesWrapper__shape'):
            if mode == 'ask':
                shape = tuple(map(int, raw_input((
                    'Note on %s:\n'
                    'Original subplot shape in the grid is '
                    'unknown. Please enter it as a tuple (x,y)\n') % (
                        self.name)).split(',')))
            elif mode == 'raise':
                raise ValueError(
                    'Original position of the subplot in the grid is '
                    'unknown!')
            elif mode == 'ignore':
                return None
            try:
                num = self._ax._AxesWrapper__num
            except AttributeError:
                if mode == 'ask':
                    num = int(raw_input((
                        'Note on %s:\n'
                        'Original number of the subplot in the grid is '
                        'unknown. Please enter it as an integer\n') % (
                            self.name)))
                elif mode == 'raise':
                    raise ValueError(
                        'Original number of the subplot in the grid is '
                        'unknown!')
                elif mode == 'ignore':
                    return shape
            self.ax = wrap_subplot(
                self._ax, num, shape, **self._get_fig_kwargs(
                    self._ax.get_figure()))
        return self._ax._AxesWrapper__shape

    def _get_axes_num(self, mode='ask'):
        """Returns the axes shape from the ax._AxesWrapper__shape attribute"""
        if not hasattr(self._ax, '_AxesWrapper__num'):
            try:
                shape = self._ax._AxesWrapper__shape
            except AttributeError:
                if mode == 'ask':
                    shape = tuple(map(int, raw_input((
                        'Note on %s:\n'
                        'Original subplot shape in the grid is '
                        'unknown. Please enter it as a tuple (x,y)\n') % (
                            self.name)).split(',')))
                elif mode == 'raise':
                    raise ValueError(
                        'Original position of the subplot in the grid is '
                        'unknown!')
                elif mode == 'ignore':
                    pass
            if mode == 'ask':
                num = int(raw_input((
                    'Note on %s:\n'
                    'Original number of the subplot in the grid is '
                    'unknown. Please enter it as an integer\n') % (
                        self.name)))
            elif mode == 'raise':
                raise ValueError(
                    'Original number of the subplot in the grid is '
                    'unknown!')
            elif mode == 'ignore':
                return None
            self.ax = wrap_subplot(
                self._ax, num, shape, **self._get_fig_kwargs(
                    self._ax.get_figure()))
        return self._ax._AxesWrapper__num

    def _get_fig_kwargs(self, fig):
        """Returns a dictionary for plt.figure to create the figure the same
        way as it was at the initialization"""
        return {'figsize': (fig.get_figwidth(), fig.get_figheight()),
                'dpi': fig.get_dpi(),
                'facecolor': fig.get_facecolor(),
                'edgecolor': fig.get_edgecolor(),
                'frameon': fig.get_frameon(),
                'FigureClass': fig.__class__}

    def set_meta(self, meta={}, **kwargs):
        """Set meta information

        This methods assigns meta informations to the instance

        Parameters
        ----------
        meta: dict
            Key value pairs must be the meta identifier and the value that
            shall be assigned (you can use this if ``**kwargs`` does not work)
        **kwargs
            Key value pairs must be the meta identifier and the value that
            shall be assigned

        Examples
        --------
        Set information about the experiment::

            mapo.set_meta(exp='RCP8.5')
            print mapo.meta['exp']
            # returns 'RCP8.5'"""
        kwargs.update(meta)
        for key, val in kwargs.items():
            self._meta[key] = val  # do not destroy the order

    def del_meta(self, *args):
        """Delete meta information for the instance

        This methods deletes meta informations identified by ``*args`` from
        the instance

        Parameters
        ----------
        *args
            Keys in self._meta that shall be deleted (not in the reader!)

        Examples
        --------
        Set information about the experiment::

            mapo.set_meta(test=5)
            print mapo.meta['test']
            # returns 5
            mapo.del_meta('test')
            print mapo.meta['test']
            # gives a KeyError"""
        for arg in args:
            try:
                del self._meta[arg]
            except KeyError:
                pass
