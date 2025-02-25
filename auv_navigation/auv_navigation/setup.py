from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(packages=["auv_navigation"], package_dir={"": "src"})

setup(**d)
