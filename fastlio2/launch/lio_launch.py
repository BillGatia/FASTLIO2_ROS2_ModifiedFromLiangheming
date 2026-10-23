import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # 控制是否启动 RViz
    use_rviz = LaunchConfiguration('use_rviz')

    declare_use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to use rviz2'
    )

    config_file = LaunchConfiguration('config')
    declare_config_arg = DeclareLaunchArgument(
        'config',
        default_value='lio.yaml',
        description='LIO YAML filename in the fastlio2 config directory'
    )

    # RViz 配置文件路径
    rviz_cfg = PathJoinSubstitution(
        [
            FindPackageShare("fastlio2"),
            "rviz",
            "fastlio2.rviz"
        ]
    )

    # LIO 配置文件路径
    config_path = PathJoinSubstitution(
        [
            FindPackageShare("fastlio2"),
            "config",
            config_file
        ]
    )

    # LIO 节点
    lio_node = Node(
        package="fastlio2",
        namespace="fastlio2",
        executable="lio_node",
        name="lio_node",
        output="screen",
        parameters=[
            {
                "config_path": config_path
            }
        ]
    )

    # RViz 节点
    rviz_node = Node(
        package="rviz2",
        namespace="fastlio2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(use_rviz)
    )

    # 启动
    return LaunchDescription(
        [
            declare_use_rviz_arg,
            declare_config_arg,
            lio_node,
            rviz_node
        ]
    )
