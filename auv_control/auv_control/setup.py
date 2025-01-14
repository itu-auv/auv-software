from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['auv_control'],
    package_dir={'': 'lib'}  
)

# Run the setup
setup(**d)
