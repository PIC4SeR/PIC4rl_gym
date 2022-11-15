from setuptools import setup
import os 
from glob import glob

package_name = 'testing'
submodules = package_name+"/networks"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'goals_and_poses'), glob('goals_and_poses/*.*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='eiraleandrea@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'pic4rl_tester = testing.pic4rl_tester:main',
        'plot_training_rewards = testing.plot_training_rewards:main',
        ],
    },
)
