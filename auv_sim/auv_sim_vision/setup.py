from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup(
    **generate_distutils_setup(
        packages=["auv_sim_vision"],
        package_dir={"": "src"},
    )
)
