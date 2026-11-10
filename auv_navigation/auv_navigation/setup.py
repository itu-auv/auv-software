from glob import glob

from setuptools import find_packages, setup

package_name = "auv_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sencer Yazici",
    maintainer_email="senceryazici@gmail.com",
    description="Navigation package",
    license="BSD-3-Clause",
    scripts=sorted(glob("scripts/*.py")),
)
