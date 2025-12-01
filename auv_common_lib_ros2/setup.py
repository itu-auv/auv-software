from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["auv_common_lib"],
    # scripts=['bin/myscript'],
    package_dir={"": "python"},
)

setup(**d)
