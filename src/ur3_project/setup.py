from setuptools import setup
import os
from glob import glob

package_name = 'ur3_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marlon',
    maintainer_email='marlon@example.com',
    description='UR3 Project Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pick_place_manager_node = ur3_project.pick_place_manager_node:main'
        ],
    },
)