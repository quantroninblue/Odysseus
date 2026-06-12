from pathlib import Path

from setuptools import find_namespace_packages, setup


package_name = "semantic_spatial_mapping_ros"
repo_root = Path(__file__).resolve().parents[2]

repo_packages = find_namespace_packages(
    where=str(repo_root),
    include=[
        "geometry", "geometry.*",
        "mapping", "mapping.*",
        "motion", "motion.*",
        "planning", "planning.*",
        "runtime", "runtime.*",
        "segmentation", "segmentation.*",
        "tracking", "tracking.*",
        "world", "world.*",
    ],
    exclude=["external_anirudh_vslam*"],
)

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name] + repo_packages,
    package_dir={
        "": ".",
        "geometry": "../../geometry",
        "mapping": "../../mapping",
        "motion": "../../motion",
        "planning": "../../planning",
        "runtime": "../../runtime",
        "segmentation": "../../segmentation",
        "tracking": "../../tracking",
        "world": "../../world",
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            [
                "config/gazebo.yaml",
                "config/gazebo_fused.yaml",
                "config/gazebo_odom_imu_ekf.yaml",
                "config/embedded_oakd.yaml",
                "config/embedded_oakd_fused.yaml",
                "config/embedded_odom_imu_ekf.yaml",
            ],
        ),
        (
            f"share/{package_name}/launch",
            [
                "launch/gazebo_runtime.launch.py",
                "launch/gazebo_fused_runtime.launch.py",
                "launch/embedded_runtime.launch.py",
                "launch/embedded_fused_runtime.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="semantic_spatial_mapping",
    maintainer_email="robotics@example.com",
    description="Deployment ROS2 runtime for semantic spatial mapping and visual SLAM.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "semantic_spatial_node = semantic_spatial_mapping_ros.semantic_spatial_node:main",
        ],
    },
)
