
import numpy as np

from motion.vo.pipeline import (
    VisualOdometry,
    VOConfig,
)

from world.transforms.world_frame_projections import (
    WorldFrameProjector,
)

from mapping.global_map.world_map import (
    WorldMap,
)



class SemanticSpatialRuntime:
    """Minimal semantic-spatial mapping runtime.

    Responsibilities
    ----------------
    - pose estimation
    - world-frame projection
    - persistent map accumulation
    """

    def __init__(self, camera_model):
        self.vo = VisualOdometry(
            camera=camera_model,
            config=VOConfig(),
        )

        self.world_projector = WorldFrameProjector()

        self.world_map = WorldMap()

    def update(self, rgb_frame, object_points_camera, timestamp=0.0):
        """Update runtime with current frame and object points.

        Parameters
        ----------
        rgb_frame
            Current RGB image

        object_points_camera : (N, 3)
            Object-local pointcloud in camera coordinates
        """

        pose_update = self.vo.update(img=rgb_frame, timestamp=timestamp)

        if not pose_update.success:
            return pose_update

        points_world = self.world_projector.project_points_to_world(
            points_camera=object_points_camera,
            T_world_cam=pose_update.T_world_cam,
        )

        self.world_map.add_points(points_world)

        return pose_update

