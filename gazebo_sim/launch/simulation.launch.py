import os
from ament_index_python.packages import get_package_share_directory
import launch
import yaml
import json
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


configFilepath = os.path.join(
    get_package_share_directory("pic4rl"), 'config',
    'main_params.yaml'
    )
with open(configFilepath, 'r') as file:
    configParams = yaml.safe_load(file)['main_node']['ros__parameters']

# Fetching Goals and Poses
goals_path = os.path.join(
    get_package_share_directory("pic4rl"), 
    'goals_and_poses', 
    configParams['data_path']
    )
goal_and_poses = json.load(open(goals_path,'r'))
robot_pose, goal_pose = goal_and_poses["initial_pose"], goal_and_poses["goals"][0]

x_rob = '-x '+str(robot_pose[0])
y_rob = '-y '+str(robot_pose[1])
z_rob = '-z '+str(0.3)
yaw_rob = '-Y ' +str(robot_pose[2])

x_goal = '-x '+str(goal_pose[0])
y_goal = '-y '+str(goal_pose[1])
# z_goal = '-z 0.01'

# Fetching World, Robot and Goal models
world_path = os.path.join(
    get_package_share_directory("gazebo_sim"), 
    'worlds', 
    configParams["world_name"]
    )

robot_pkg = get_package_share_directory(configParams["robot_name"])

goal_entity = os.path.join(get_package_share_directory("gazebo_sim"), 'models', 
            'goal_box', 'model.sdf')

def generate_launch_description():
    
    use_sim_time_arg = DeclareLaunchArgument(
            'use_sim_time',
            default_value = "true",
            description = 'Use simulation clock if true')

    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
    		os.path.join(robot_pkg,'launch', 'description.launch.py')
            )
        )

    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=['-entity',configParams["robot_name"], x_rob, y_rob, z_rob, yaw_rob, '-topic','/robot_description'],
    )

    spawn_goal = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=['-entity', 'goal', '-file', goal_entity, x_goal, y_goal]
    )
    
    gazebo = launch.actions.ExecuteProcess(
        cmd=['gazebo','--verbose', world_path, '-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so'],
        output='screen'
        )
    
    return launch.LaunchDescription([
        use_sim_time_arg,
        robot_description,
        spawn_robot,
        spawn_goal,
        TimerAction(period=5., actions=[gazebo]),
    ])

