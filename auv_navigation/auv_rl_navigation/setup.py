from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        "auv_rl_navigation",
        "auv_rl_navigation.environments",
        "auv_rl_navigation.observation",
        "auv_rl_navigation.agents",
    ],
    package_dir={"": "src"},
)

setup(**d)
