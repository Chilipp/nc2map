#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""this script tests (almost) all formatoption keywords for the
WindPlot whether their update works. Check the output stream_update_test.pdf
and quiver_update_test.pdf)."""
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nc2map
import time
import datetime as dt
testdict_both = {
    'axiscolor': {
        'bottom': 'red', 'left': 'blue', 'right': None, 'top': None},
    'bounds': ['rounded',11,[25,75]],
    'clabel': '%(long_name)s [%(units)s] in %B, %Y',
    'cmap': 'winter',
    'color': 'u',
    'cticksize': 'x-large',
    'ctickweight': 'bold',
    'countries': True,
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
    'lonlatbox': 'Europe',
    'lsm': False,
    'mask': ['../demo/focus_regions.nc', 'mask', 2],
    'meridionals': 5,
    'merilabelpos': [1,1,1,1],
    'ocean_color': 'aqua',
    'paralabelpos': [1,1,1,1],
    'parallels': 3,
    'plotcbar': 'r',
    #'proj': 'northpole',
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
    'u': 'v',
    'v': 'u',
    'linewidth': 'absolute',
    'scale': 2.0}

testdict_quiver = {
    'density': 0.5,
    'reduceabove': [.50, 50],
    'rasterized': False,
    #'legend': None
    }

testdict_stream = {
    'arrowsize': 2.0,
    'arrowstyle': 'fancy',
    'density': 2.0}

testdict_quiver.update(testdict_both)
testdict_stream.update(testdict_both)
t0 = dt.datetime.now()
#i=0
for txt, testdict in (['quiver', testdict_quiver],
                      ['stream', testdict_stream]):
    t = dt.datetime.now()
    print(str(t)+": Testing "+txt)
    output = txt+"_update_test.pdf"
    pdf = PdfPages(output)
    if txt == 'quiver':
        fmt = {'lonlatbox': [-80,20,-20,20]}
    elif txt == 'stream':
        fmt = {'lonlatbox': [-80,20,-20,20], 'streamplot': True}
    mymaps = nc2map.Maps('../demo/demo-t2m-u-v.nc', u='u', v='v',
                         windonly=True, fmt=fmt)
    cmapprops= mymaps.maps[0].fmt._cmapprops
    plt.show(block=False)

    for key, val in testdict.items():
        print("    updating " + str(key) + " to " + str(val))
        strval = str(val).replace('{', '{{').replace('}', '}}')
        if key in cmapprops and key != 'color':
            mymaps.update(title=str(key) + ": " + strval, color='absolute',
                          fmt={key: val})
        else:
            mymaps.update(title=str(key) + ": " + strval, fmt={key: val})
        pdf.savefig(plt.gcf())
        if key in ['tight']:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", '|tight|', nc2map.warning.Nc2MapWarning,
                    'nc2map.formatoptions._base_fmt', 0)
                mymaps.update(todefault=True, fmt=fmt)
                mymaps.reset(ax=(1,1))
        elif key == 'text':
            mymaps.update(text=[(0,0,'','axes')], todefault=True, fmt=fmt)
        else:
            mymaps.update(todefault=True, fmt=fmt)
        #i+=1
        #if i == 10: break
    mymaps.update(title='Done', todefault=True)
    pdf.savefig(plt.gcf())
    print("Saving to " + output)
    pdf.close()
    mymaps.close()
    print("Time needed: " + str(dt.datetime.now()-t))
print("Time needed in total: " + str(dt.datetime.now()-t0))
