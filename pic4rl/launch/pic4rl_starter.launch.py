from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterFile, Parameter, ParameterValue
from pic4rl.utils.launch_utils import camel_to_snake
import yaml

from nav2_common.launch import ReplaceString, RewrittenYaml


def print_params(context, *args, **kwargs):
    sensor = LaunchConfiguration("sensor")
    task = LaunchConfiguration("task")
    pkg_name = LaunchConfiguration("pkg_name")
    main_params = LaunchConfiguration("main_params")
    mode = LaunchConfiguration("mode")
    
    if not (
        sensor.perform(context=context) == "" and task.perform(context=context) == "" and mode.perform(context=context) == ""
    ):
        configured_params = ParameterFile(
            RewrittenYaml(
                source_file=main_params,
                param_rewrites={
                    "sensor": sensor,
                    "task": task,
                    "package_name": pkg_name,
                    "mode": mode,
                },
                convert_types=True,
            ),
            allow_substs=True,
        )
    else:
        configured_params = ParameterFile(
            main_params,
            allow_substs=True,
        )
    # open file of Parameters
    with open(configured_params.param_file[0].perform(context=context), "r") as file:
        main_params = yaml.safe_load(file)["main_node"]["ros__parameters"]
        sensor_name = main_params["sensor"]
        task_name = main_params["task"]
        print(camel_to_snake(task_name) + "_" + camel_to_snake(sensor_name))
        executable_name = camel_to_snake(task_name) + "_" + camel_to_snake(sensor_name)

    task_node = Node(
        package=pkg_name,
        executable=executable_name,
        name="pic4rl_starter",
        output="screen",
        emulate_tty=True,
        parameters=[
            main_params,
        ],
    )
    return [task_node]

def generate_launch_description():
    # Launch configuration variables specific to simulation
    # Declare the launch arguments
    pkg_name = LaunchConfiguration("pkg_name")

    sensor_arg = DeclareLaunchArgument(
        "sensor", default_value="", description="sensor type: camera or lidar"
    )

    task_arg = DeclareLaunchArgument(
        "task",
        default_value="",
        description="task type: goToPose, Following, Vineyards",
    )

    pkg_name_arg = DeclareLaunchArgument(
        "pkg_name", default_value="pic4rl", description="package name"
    )

    main_params_arg = DeclareLaunchArgument(
        "main_params",
        default_value=PathJoinSubstitution([FindPackageShare(pkg_name), "config", "main_params.yaml"]),
        description="main_params.yaml",
    )

    mode_params_arg = DeclareLaunchArgument(
        "mode",
        default_value="",
        description="mode: training or testing",
    )

    # Specify the actions
    ld = LaunchDescription()
    ld.add_action(sensor_arg)
    ld.add_action(task_arg)
    ld.add_action(pkg_name_arg)
    ld.add_action(main_params_arg)
    ld.add_action(mode_params_arg)
    ld.add_action(OpaqueFunction(function=print_params))
    return ld
