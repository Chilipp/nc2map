#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""this script tests whether the creation and update of one colorbar works.
Check the output (onecbar_test.pdf and onecbar_test.gif)"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nc2map
import time
import datetime as dt
testdict = {
    'bounds': ['rounded',11,[25,75]],
    'clabel': '%(long_name)s [%(units)s] in %B, %Y',
    'cmap': 'winter',
    'cticksize': 'x-large',
    'ctickweight': 'bold',
    'extend': 'both',
    'labelsize': 'large',
    'labelweight': 'bold',
    'plotcbar': 'r',
    'ticklabels': ['test%i' % i for i in xrange(5)],
    'ticks': 2,
    }

output = "onecbar_test.pdf"
pdf = PdfPages(output)
t = dt.datetime.now()
mymaps = nc2map.Maps('../demo/demo-t2m-u-v.nc', vlst=['u', 'v'], ax=(1,2))
plt.show(block=False)
# test creation and removing of cbar
mymaps.update(title='Colorbar created for %(var)s')
mymaps.update_cbar(var='u', clabel='%(var)s')
pdf.savefig(plt.gcf())
mymaps.update(title='Colorbar removed')
mymaps.removecbars(var='u')
pdf.savefig(plt.gcf())
mymaps.update(title='Colorbar recreated for %(var)s')
mymaps.update_cbar(var=['u', 'v'])
pdf.savefig(plt.gcf())
for key, val in testdict.items():
    print("updating " + str(key) + " to " + str(val))
    strval = str(val).replace('{', '{{').replace('}', '}}')
    mymaps.update(title=str(key) + ": " + strval)
    mymaps.update_cbar(**{key: val})
    pdf.savefig(plt.gcf())
    #time.sleep(2)
    mymaps.update_cbar(todefault=True, var=['u', 'v'])
mymaps.update(title='Done')
pdf.savefig(plt.gcf())
print("Saving to " + output)
pdf.close()
print("Time needed: " + str(dt.datetime.now()-t))
mymaps.update(title='%(long_name)s (%(var)s)')
mymaps.update_cbar(clabel='%B, %Y')
mymaps.make_movie(output.replace('.pdf', '')+'.gif')
