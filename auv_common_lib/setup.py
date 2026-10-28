from setuptools import find_packages, setup

d = dict(
    name="auv_common_lib",
    version="0.1.0",
    packages=find_packages(where="python"),
    # scripts=['bin/myscript'],
    package_dir={"": "python"},
)

setup(**d)
