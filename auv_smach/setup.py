from glob import glob

from setuptools import find_packages, setup

package_name = "auv_smach"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sencer Yazici",
    maintainer_email="senceryazici@gmail.com",
    description="Mission state machines",
    license="BSD-3-Clause",
    scripts=sorted(glob("scripts/*.py")),
)
