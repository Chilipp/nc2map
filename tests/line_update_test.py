#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""this script tests (almost) all formatoption keywords for the
LinePlot whether their update works. Check the output line_update_test.pdf"""
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nc2map
import time
import datetime as dt
testdict = {
    'axiscolor': {
        'bottom': 'red', 'left': 'blue', 'right': None, 'top': None},
    'enable': False,
    'figtitle': '%(var)s',
    'figtitlesize': 16,
    'figtitleweight': 'bold',
    'fontsize': 16,
    'fontweight': 'bold',
    'labelsize': 'large',
    'labelweight': 'bold',
    'grid': True,
    'legend': True,
    'scale': 'logx',
    'text': [(0,0,'mytest, %(name)s','axes')],
    'ticksize': 'large',
    'tickweight': 'bold',
    #'tight': True,
    #'title': 'my test title',
    'titlesize': 'small',
    'titleweight': 'bold',
    'xdeci': 2,
    'xformat': 'sci',
    'xlabel': '%(dim_standard_name)s',
    'xlim': (40000, 80000),
    'xrotation': 90,
    'xticklabels': ['test%i' % i for i in xrange(9)],
    'xticks': 2,
    'ydeci': 2,
    'yformat': 'sci',
    'ylabel': '%(long_name)s [%(units)s]',
    'ylim': (-1., 2.),
    'yrotation': 90,
    'yticklabels': ['test%i' % i for i in xrange(8)],
    'yticks': 2}

output = "line_update_test.pdf"
pdf = PdfPages(output)
t = dt.datetime.now()
mymaps = nc2map.Maps('../demo/demo-t2m-u-v.nc', vlst='u', linesonly=True,
                     time=0, lon=0, lat=0)
mymaps.addline(mymaps.lines[0].reader, vlst='u', time=1, lon=0, lat=0,
               ax=mymaps.lines[0].ax, fmt=mymaps.lines[0].fmt)
plt.show(block=False)
for key, val in testdict.items():
    print("updating " + str(key) + " to " + str(val))
    strval = str(val).replace('{', '{{').replace('}', '}}')
    mymaps.update_lines(title=str(key) + ": " + strval, name='line1',
                        **{key: val})
    pdf.savefig(plt.gcf())
    #time.sleep(2)
    if key in ['tight']:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", '|tight|', nc2map.warning.Nc2MapWarning,
                'nc2map.formatoptions._base_fmt', 0)
            mymaps.update_lines(todefault=True)
            mymaps.reset(ax=(1,1))
    elif key == 'text':
        mymaps.update_lines(text=[(0,0,'','axes')], todefault=True)
    else:
        mymaps.update_lines(todefault=True)
mymaps.update_lines(title='Done')
pdf.savefig(plt.gcf())
print("Saving to " + output)
pdf.close()
print("Time needed: " + str(dt.datetime.now()-t))
