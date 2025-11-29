from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["auv_reinforcement_learning"],
    package_dir={"": "src"},
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.21.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.11.0",
    ],
)

setup(**setup_args)
