#!/usr/bin/env python
from setuptools import setup

setup (name     = 'nc2map',
    version     = '0.1.0',
    author      = "Philipp Sommer",
    author_email= "philipp.sommer@studium.uni-hamburg.de",
    license     = "GPLv2",
    description = """ ===================== """,
    platforms   = ["any"],
    packages  = ["nc2map","nc2map/mapos","nc2map/formatoptions","nc2map/data"],
    url         = "https://github.com/Chilipp/nc2map",
    keywords    = ['netcdf','data','science','plotting'],
    classifiers = [
      "Development Status :: 4 - Beta",
      "Topic :: Utilities",
      "Operating System :: POSIX",
      "Programming Language :: Python",
      ],
    )

