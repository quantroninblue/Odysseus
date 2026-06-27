FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-colcon-common-extensions \
    python3-pip \
    python3-pytest \
    python3-numpy \
    python3-yaml \
    python3-venv \
    ros-jazzy-robot-localization \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-sim \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir torch

WORKDIR /opt/odysseus
COPY . /opt/odysseus

RUN python3 -m compileall \
    planning runtime/core mapping/global_map motion/vo segmentation \
    ros/semantic_spatial_mapping_ros/semantic_spatial_mapping_ros tools tests

RUN mkdir -p /opt/odysseus_ws/src \
    && ln -s /opt/odysseus/ros/semantic_spatial_mapping_ros /opt/odysseus_ws/src/semantic_spatial_mapping_ros \
    && source /opt/ros/jazzy/setup.bash \
    && cd /opt/odysseus_ws \
    && colcon build --packages-select semantic_spatial_mapping_ros

RUN printf '%s\n' \
    'source /opt/ros/jazzy/setup.bash' \
    'source /opt/odysseus_ws/install/setup.bash' \
    'cd /opt/odysseus' \
    >> /root/.bashrc

CMD ["bash"]
