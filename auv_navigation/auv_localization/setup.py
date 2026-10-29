from glob import glob
from pathlib import Path

from setuptools import setup

package_name = 'auv_localization'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
]
for directory in sorted(Path('config').iterdir()):
    if directory.is_dir():
        data_files.append((
            'share/' + package_name + '/' + directory.as_posix(),
            [str(path) for path in sorted(directory.glob('*.yaml'))],
        ))

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Batuhan Ozer',
    maintainer_email='batuhanozer52@gmail.com',
    description='Localization package',
    license='BSD-3-Clause',
    scripts=sorted(glob('scripts/*.py')),
)
