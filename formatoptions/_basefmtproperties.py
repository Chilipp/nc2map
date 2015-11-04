# -*- coding: utf-8 -*-
"""This file contains a class for basic formatoption properties of simple x-y
plot
"""
from copy import deepcopy
from ..defaults import texts

__author__ = "Philipp Sommer (philipp.sommer@studium.uni-hamburg.de)"
__version__ = '0.0'


class BaseFmtProperties(object):
    def default(self, x, doc):
        """default property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def fontsize(self, x, doc):
        """default fontsize options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if value is not None:
                for label in self._text_props:
                    if getattr(self, label + 'size') == \
                            self._default[label + 'size']:
                        getattr(self, '_' + label + 'ops').update(
                            {'fontsize': value})
            else:
                for label in self._text_props:
                    setattr(self, label + 'size',
                            getattr(self, label + 'size'))

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def ticksize(self, x, doc):
        """tick fontsize options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('size', '') not in self._text_props:
                self._text_props.append(x.replace('size', ''))
            self._tickops.update({'fontsize': value})
            try:  # reset xticks for pad option
                self.xticks = self.xticks
            except AttributeError:
                pass
            try:  # reset yticks for pad option
                self.yticks = self.yticks
            except AttributeError:
                pass

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def figtitlesize(self, x, doc):
        """figtitle fontsize options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('size', '') not in self._text_props:
                self._text_props.append(x.replace('size', ''))
            self._figtitleops.update({'fontsize': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def titlesize(self, x, doc):
        """title fontsize options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('size', '') not in self._text_props:
                self._text_props.append(x.replace('size', ''))
            self._titleops.update({'fontsize': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def labelsize(self, x, doc):
        """axes label fontsize options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('size', '') not in self._text_props:
                self._text_props.append(x.replace('size', ''))
            self._labelops.update({'fontsize': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def fontweight(self, x, doc):
        """default fontweight options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            for label in self._text_props:
                if getattr(self, label + 'weight') == \
                        self._default[label+'weight']:
                    getattr(self, '_' + label + 'ops').update(
                        {'fontweight': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def tickweight(self, x, doc):
        """tick fontweight options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('weight', '') not in self._text_props:
                self._text_props.append(x.replace('weight', ''))
            self._tickops.update({'fontweight': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def figtitleweight(self, x, doc):
        """figtitle fontweight options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('weight', '') not in self._text_props:
                self._text_props.append(x.replace('weight', ''))
            self._figtitleops.update({'fontweight': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def titleweight(self, x, doc):
        """title fontweight options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('weight', '') not in self._text_props:
                self._text_props.append(x.replace('weight', ''))
            self._titleops.update({'fontweight': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def labelweight(self, x, doc):
        """axes label fontweight options property"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            if x.replace('weight', '') not in self._text_props:
                self._text_props.append(x.replace('weight', ''))
            self._labelops.update({'fontweight': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def tickops(self, x, doc):
        """property which adds to tickops"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            self._tickops.update({x: value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def titleops(self, x, doc):
        """property which adds to tickops"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            self._titleops.update({x: value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def labelops(self, x, doc):
        """property which adds to tickops"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            setattr(self, '_' + x, value)
            self._labelops.update({x: value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def text(self, x, doc):
        """property to add text to the figure"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            value = deepcopy(value)
            if isinstance(value, (str, unicode)):
                xpos, ypos = texts['default_position']
                value = (xpos, ypos, value, 'axes', {'ha': 'right'})
            if value == []:
                oldtexts = value
                textstoupdate = []
            else:
                if isinstance(value, tuple):
                    value = [value]
                textstoupdate = []
                for text in value:
                    if all(trans not in text
                           for trans in ['axes', 'fig', 'data']):
                        text = list(text)
                        text.insert(3, 'data')
                        text = tuple(text)
                    oldtexts = getattr(self, '_' + x)
                    append = True
                    for oldtext in oldtexts:
                        if all(oldtext[i] == text[i]
                               for i in [0, 1, 3]):
                            if text[2] == '':
                                oldtexts.remove(oldtext)
                            else:
                                oldtexts[oldtexts.index(oldtext)] = text
                                textstoupdate.append(text)
                            append = False
                    if append:
                        oldtexts.append(text)
                        textstoupdate.append(text)
            self._textstoupdate = (text for text in textstoupdate)
            setattr(self, '_' + x, oldtexts)

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def axiscolor(self, x, doc):
        """property to set up the color of the axis"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            positions = ['right', 'left', 'top', 'bottom']
            if isinstance(value, dict):
                if any(key not in positions for key in value):
                    raise KeyError(
                        "The only keys allowed for axiscolor are %s" % (
                            ', '.join(positions)))
                setattr(self, '_' + x, value)
            else:
                setattr(self, '_' + x, {k: value for k in positions})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def legend(self, x, doc):
        """property to set up the color of the axis"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            if value is None:
                setattr(self, '_' + x, value)
            elif isinstance(value, dict):
                setattr(self, '_' + x, value)
            else:
                setattr(self, '_' + x, {'loc': value})

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)

    def ticks_and_labels(self, x, doc):
        """Property for x and yticks and ticklabels"""
        def getx(self):
            return getattr(self, '_' + x)

        def setx(self, value):
            try:  # try dictionary
                major_ticks = value.get('major')
                minor_ticks = value.get('minor')
                pad = value.get('pad')
                setattr(self, '_' + x, {'major': major_ticks,
                                        'minor': minor_ticks})
            except AttributeError:
                setattr(self, '_' + x, {'major': value,
                                        'minor': None})
                pad = None
            if x.endswith('labels'):
                return
            if pad:
                self._tickops['pad'] = pad
            elif getattr(self, '_' + x)['minor'] is not None:
                try:
                    ticksize = self.ticksize['minor']
                except TypeError:
                    ticksize = self.ticksize
                pads = {'xx-small': 9,
                        'x-small': 11,
                        'small': 13,
                        'medium': 18,
                        'large': 20,
                        'x-large': 22,
                        'xx-large': 24}
                try:
                    self._tickops['pad'] = pads[ticksize]
                except KeyError:
                    self._tickops['pad'] = ticksize + 3

        def delx(self):
            setattr(self, '_' + x, self._default[x])
        return property(getx, setx, delx, doc)
