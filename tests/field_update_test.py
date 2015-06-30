#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""this script tests (almost) all formatoption keywords for the
FieldPlot whether their update works. Check the output field_update_test.pdf"""
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nc2map
import time
import datetime as dt
testdict = {
    'axiscolor': {
        'bottom': 'red', 'left': 'blue', 'right': None, 'top': None},
    'bounds': ['rounded',11,[25,75]],
    'clabel': '%(long_name)s [%(units)s] in %B, %Y',
    'cmap': 'winter',
    'countries': True,
    'cticksize': 'x-large',
    'ctickweight': 'bold',
    'enable': False,
    'extend': 'both',
    'figtitle': '%(var)s',
    'figtitlesize': 16,
    'figtitleweight': 'bold',
    'fontsize': 16,
    'fontweight': 'bold',
    'labelsize': 'large',
    'labelweight': 'bold',
    'land_color': 'coral',
    #'latlon': False,
    'lonlatbox': {'CONTINENT': ['Europe']},
    'lsm': False,
    'mask': ['../demo/focus_regions.nc', 'mask', 2],
    'maskgreater': 300,
    'maskgeq': 300,
    'maskless': 280,
    'maskleq': 280,
    'maskbetween': (280, 290),
    'meridionals': 5,
    'merilabelpos': [1,1,1,1],
    'ocean_color': 'aqua',
    'paralabelpos': [1,1,1,1],
    'parallels': 3,
    'plotcbar': 'r',
    'proj': 'northpole',
    #'rasterized': False,  # don't update because it is horrible for pdfs!
    'lineshapes': {'SCOUNTRY': ['CG'], 'COUNTRY': ['Zambia'], 'linewidth': 3},
    'text': [(0,0,'mytest, %Y','axes')],
    'ticklabels': ['test%i' % i for i in xrange(6)],
    'ticks': 2,
    'ticksize': 'large',
    'tickweight': 'bold',
    'tight': True,
    #'title': 'my test title',
    'titlesize': 'small',
    'titleweight': 'bold',
    'time': 1,
    'level': 1,
    'var': 'u'
    }

output = "field_update_test.pdf"
pdf = PdfPages(output)
t = dt.datetime.now()
mymaps = nc2map.Maps('../demo/demo-t2m-u-v.nc', vlst='t2m')

for key, val in testdict.items():
    print("updating " + str(key) + " to " + str(val))
    fmt = {key: val}
    if key in ['land_color', 'ocean_color']:
        fmt['maskgreater'] = 280
    strval = str(val).replace('{', '{{').replace('}', '}}')
    mymaps.update(title=str(key) + ": " + strval, fmt=fmt)
    pdf.savefig(plt.gcf())
    #time.sleep(2)
    if key in ['tight']:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", '|tight|', nc2map.warning.Nc2MapWarning,
                'nc2map.formatoptions._base_fmt', 0)
            mymaps.update(todefault=True)
            mymaps.reset(ax=(1,1))
    elif key == 'text':
        mymaps.update(text=[(0,0,'','axes')], todefault=True)
    else:
        mymaps.update(todefault=True)
mymaps.update(title='Done')
print("Saving to " + output)
pdf.savefig(plt.gcf())
pdf.close()
print("Time needed: " + str(dt.datetime.now()-t))
