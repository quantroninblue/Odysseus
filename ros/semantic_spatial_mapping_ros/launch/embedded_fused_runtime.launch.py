from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("semantic_spatial_mapping_ros")
    default_runtime_config = PathJoinSubstitution(
        [package_share, "config", "embedded_oakd_fused.yaml"]
    )
    default_ekf_config = PathJoinSubstitution(
        [package_share, "config", "embedded_odom_imu_ekf.yaml"]
    )
    runtime_config_path = LaunchConfiguration("runtime_config_path")
    ekf_config_path = LaunchConfiguration("ekf_config_path")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "runtime_config_path",
                default_value=default_runtime_config,
                description="Semantic runtime YAML config.",
            ),
            DeclareLaunchArgument(
                "ekf_config_path",
                default_value=default_ekf_config,
                description="robot_localization EKF YAML config.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulated ROS time if a /clock bridge is running.",
            ),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[ekf_config_path, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="semantic_spatial_mapping_ros",
                executable="semantic_spatial_node",
                name="semantic_spatial_node",
                output="screen",
                parameters=[
                    {
                        "config_path": runtime_config_path,
                        "use_sim_time": use_sim_time,
                    }
                ],
            ),
        ]
    )
