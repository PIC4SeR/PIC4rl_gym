from setuptools import setup
import os 
from glob import *

package_name = 'pic4rl'
#submodules = package_name+'/tasks'

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
        (os.path.join('share', package_name, 'goals_and_poses'), glob('goals_and_poses/*.json'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mauromartini',
    maintainer_email='mauro.martini@polito.it',
    description='Deep Reinforcement Learning algorithms for multi-platform robotic autonomous navigation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'pic4rl_trainer = pic4rl.pic4rl_trainer:main',
        'plot_reward = pic4rl.plot_reward:main'
        ],
    },
)
