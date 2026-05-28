
import numpy as np

from segmentation.segmentation_reference import (
    SegmentationModule,
)

from motion.vo.pipeline import (
    VisualOdometry,
    VOConfig,
)

from geometry.pointclouds.object_pointclouds import (
    ObjectPointCloudExtractor,
)

from world.transforms.world_frame_projections import (
    WorldFrameProjector,
)

from mapping.global_map.world_map import (
    WorldMap,
)


class SemanticSpatialRuntime:

    """
    Integrated semantic-spatial mapping runtime.

    Pipeline
    --------
    RGB
      ↓
    Segmentation
      ↓
    Semantic Masks
      ↓
    RGBD Pointcloud Extraction
      ↓
    VO Pose Estimation
      ↓
    World-frame Projection
      ↓
    Persistent Semantic Map
    """

    def __init__(

        self,

        camera_model,

        pointcloud_generator,

        projector,
    ):

        # --------------------------------------------------
        # Semantic segmentation
        # --------------------------------------------------

        self.segmentation = (
            SegmentationModule()
        )

        # --------------------------------------------------
        # Motion estimation
        # --------------------------------------------------

        self.vo = VisualOdometry(

            camera=camera_model,

            config=VOConfig(),
        )

        # --------------------------------------------------
        # Semantic geometry extraction
        # --------------------------------------------------

        self.pointcloud_extractor = (

            ObjectPointCloudExtractor(

                pointcloud_generator=
                pointcloud_generator,

                projector=projector,
            )
        )

        # --------------------------------------------------
        # World-frame projection
        # --------------------------------------------------

        self.world_projector = (
            WorldFrameProjector()
        )

        # --------------------------------------------------
        # Persistent world map
        # --------------------------------------------------

        self.world_map = (
            WorldMap()
        )

    # ======================================================
    # Main Runtime Update
    # ======================================================

    def update(

        self,

        rgb_frame,

        depth_frame,

        timestamp=0.0,
    ):

        # --------------------------------------------------
        # Semantic segmentation
        # --------------------------------------------------

        segmentation_result = (

            self.segmentation.segment(
                rgb_frame
            )
        )

        masks = (
            segmentation_result["masks"]
        )

        # --------------------------------------------------
        # VO pose update
        # --------------------------------------------------

        pose_update = self.vo.update(

            img=rgb_frame,

            timestamp=timestamp,
        )

        if not pose_update.success:

            return {

                "pose_update": pose_update,

                "segmentation": segmentation_result,

                "world_points": [],

                "tracking_ok": False,
            }

        # --------------------------------------------------
        # Per-object semantic geometry
        # --------------------------------------------------

        world_points_all = []

        for mask in masks:

            object_points_camera = (

                self.pointcloud_extractor
                .extract_object_pointcloud(

                    depth_frame=depth_frame,

                    segmentation_mask=mask,
                )
            )

            if len(object_points_camera) == 0:
                continue

            # ----------------------------------------------
            # Camera frame → World frame
            # ----------------------------------------------

            object_points_world = (

                self.world_projector
                .project_points_to_world(

                    points_camera=
                    object_points_camera,

                    T_world_cam=
                    pose_update.T_world_cam,
                )
            )

            self.world_map.add_points(
                object_points_world
            )

            world_points_all.append(
                object_points_world
            )

        # --------------------------------------------------
        # Runtime outputs
        # --------------------------------------------------

        return {

            "pose_update": pose_update,

            "segmentation": segmentation_result,

            "world_points": world_points_all,

            "tracking_ok": True,
        }

