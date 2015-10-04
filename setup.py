#!/usr/bin/env python
from setuptools import setup
from glob import glob
import os.path

moduleFiles = glob('./_[a-zA-Z]*.py') + glob('./[a-zA-Z]*.py')
moduleNames = map(lambda f : os.path.splitext(os.path.basename(f))[0], moduleFiles)
print(moduleNames)

setup (name     = 'nc2map',
    version     = '0.0.0',
    author      = "Philipp Sommer",
    author_email= "philipp.sommer@studium.uni-hamburg.de",
    license     = "GPLv2",
    description = """ ===================== """,
    platforms   = ["any"],
    package_dir = {'nc2map':''},
    packages    = ['nc2map','nc2map.mapos','nc2map.formatoptions','nc2map.data'],
    py_modules  = moduleNames,
    url         = "https://github.com/Chilipp/nc2map",
    keywords    = ['netcdf','data','science','plotting'],
    classifiers = [
      "Development Status :: 4 - Beta",
      "Topic :: Utilities",
      "Operating System :: POSIX",
      "Programming Language :: Python",
      ],
    )

