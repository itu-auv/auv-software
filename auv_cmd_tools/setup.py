#!/usr/bin/env python3
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
      packages=['auv_cmd_tools'],
      package_dir={'': 'src'},
      name='auv_cmd_tools',
      scripts=['scripts/auv']
)
setup(**d)
