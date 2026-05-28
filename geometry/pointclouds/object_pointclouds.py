import numpy as np

from geometry.transforms.depth_to_rgb_projection import (
    DepthToRGBProjector
)


class ObjectPointCloudExtractor:

    def __init__(

        self,

        pointcloud_generator,

        projector: DepthToRGBProjector
    ):

        self.generator = (
            pointcloud_generator
        )

        self.projector = (
            projector
        )

    # ========================================================
    # Reprojection-aware semantic point cloud extraction
    # ========================================================

    def extract_object_pointcloud(

        self,

        depth_frame,

        segmentation_mask,

        depth_min_m=0.1,
        depth_max_m=5.0,

        stride=4
    ):

        intr = (
            self.generator.intrinsics
        )

        depth_h, depth_w = (
            depth_frame.shape
        )

        mask_h, mask_w = (
            segmentation_mask.shape
        )

        points = []

        # ----------------------------------------------------
        # Sparse depth traversal
        # ----------------------------------------------------

        for v_depth in range(
            0,
            depth_h,
            stride
        ):

            for u_depth in range(
                0,
                depth_w,
                stride
            ):

                # --------------------------------------------
                # Read depth
                # --------------------------------------------

                depth_mm = depth_frame[
                    v_depth,
                    u_depth
                ]

                # --------------------------------------------
                # Invalid depth rejection
                # --------------------------------------------

                if (
                    depth_mm == 0 or
                    depth_mm == 65535
                ):
                    continue

                depth_m = (
                    depth_mm / 1000.0
                )

                # --------------------------------------------
                # Depth range filtering
                # --------------------------------------------

                if (
                    depth_m < depth_min_m or
                    depth_m > depth_max_m
                ):
                    continue

                # --------------------------------------------
                # Reproject depth pixel into RGB image
                # --------------------------------------------

                rgb_pixel = (

                    self.projector
                    .depth_pixel_to_rgb_pixel(

                        u_depth=u_depth,
                        v_depth=v_depth,

                        depth_m=depth_m
                    )
                )

                if rgb_pixel is None:
                    continue

                u_rgb, v_rgb = rgb_pixel

                # --------------------------------------------
                # RGB bounds check
                # --------------------------------------------

                if (
                    u_rgb < 0 or
                    u_rgb >= mask_w
                ):
                    continue

                if (
                    v_rgb < 0 or
                    v_rgb >= mask_h
                ):
                    continue

                # --------------------------------------------
                # Semantic occupancy test
                # --------------------------------------------

                if (
                    segmentation_mask[
                        v_rgb,
                        u_rgb
                    ] == 0
                ):
                    continue

                # --------------------------------------------
                # Backprojection
                # --------------------------------------------

                x = (

                    (u_depth - intr.cx) *

                    depth_m /

                    intr.fx
                )

                y = (

                    (v_depth - intr.cy) *

                    depth_m /

                    intr.fy
                )

                z = depth_m

                points.append(
                    [x, y, z]
                )

        # ----------------------------------------------------
        # Empty cloud handling
        # ----------------------------------------------------

        if len(points) == 0:

            return np.empty(
                (0, 3),
                dtype=np.float32
            )

        return np.array(
            points,
            dtype=np.float32
        )

    # ========================================================
    # Geometry statistics
    # ========================================================

    def compute_geometry_stats(

        self,

        pointcloud
    ):

        if len(pointcloud) == 0:

            return None

        centroid = np.mean(
            pointcloud,
            axis=0
        )

        min_xyz = np.min(
            pointcloud,
            axis=0
        )

        max_xyz = np.max(
            pointcloud,
            axis=0
        )

        dimensions = (
            max_xyz - min_xyz
        )

        stats = {

            "point_count": len(pointcloud),

            "centroid": centroid,

            "dimensions": dimensions,

            "min_xyz": min_xyz,

            "max_xyz": max_xyz
        }

        return stats