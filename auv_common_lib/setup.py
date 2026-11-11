from setuptools import find_packages, setup

d = dict(
    name="auv_common_lib",
    version="0.1.0",
    packages=find_packages(),
    # scripts=['bin/myscript'],
)

setup(**d)
