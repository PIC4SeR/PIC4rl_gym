from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def cameltosnake(camel_string: str) -> str:
    # If the input string is empty, return an empty string
    if not camel_string:
        return ""
    # If the first character of the input string is uppercase,
    # add an underscore before it and make it lowercase
    elif camel_string[0].isupper():
        return f"_{camel_string[0].lower()}{cameltosnake(camel_string[1:])}"
    # If the first character of the input string is lowercase,
    # simply return it and call the function recursively on the remaining string
    else:
        return f"{camel_string[0]}{cameltosnake(camel_string[1:])}"


def camel_to_snake(s):
    if len(s) <= 1:
        return s.lower()
    # Changing the first character of the input string to lowercase
    # and calling the recursive function on the modified string
    return cameltosnake(s[0].lower() + s[1:])


def generate_launch_description():
    # Launch configuration variables specific to simulation
    sensor = LaunchConfiguration("sensor")
    task = LaunchConfiguration("task")
    pkg_name = LaunchConfiguration("pkg_name")
    executable_name = LaunchConfiguration("executable_name")

    main_params = PathJoinSubstitution(
        [FindPackageShare(pkg_name), "config", "main_params.yaml"]
    )

    # Get the executable name from the parameters

    executable_name = camel_to_snake(executable_name)

    # Specify the task node

    task_node = Node(
        package=pkg_name,
        executable=executable_name,
        name="pic4rl_starter",
        output="screen",
        emulate_tty=True,
        parameters=[main_params],
        arguments=["--sensor", sensor, "--task", task],
    )

    # Declare the launch arguments

    sensor_arg = DeclareLaunchArgument(
        "sensor", default_value="camera", description="sensor type: camera or lidar"
    )

    task_arg = DeclareLaunchArgument(
        "task",
        default_value="goToPose",
        description="task type: goToPose, Following, Vineyards",
    )

    pkg_name_arg = DeclareLaunchArgument(
        "pkg_name", default_value="pic4rl", description="package name"
    )

    executable_name_arg = DeclareLaunchArgument(
        "executable_name",
        default_value="go_to_pose_lidar",
        description="executable name",
    )

    # Specify the actions
    ld = LaunchDescription()
    ld.add_action(sensor_arg)
    ld.add_action(task_arg)
    ld.add_action(pkg_name_arg)
    ld.add_action(executable_name_arg)
    ld.add_action(task_node)

    return ld
