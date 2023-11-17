from setuptools import setup
import os
from glob import *

package_name = 'gazebo_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'models', 'goal_box'), glob('models/goal_box/*.sdf'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Simone Cerrato',
    maintainer_email='simone.cerrato@polito.it',
    description='A ROS2 package to launch and manage the simulation environment',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
