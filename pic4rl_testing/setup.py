from setuptools import setup
import os 
from glob import glob

package_name = 'pic4rl_testing'
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
        'pic4rl_tester = pic4rl_testing.pic4rl_tester:main',
        'evaluate_navigation = pic4rl_testing.evaluate_navigation:main',
        'evaluate_segmentation = pic4rl_testing.evaluate_segmentation:main'
        ],
    },
)
