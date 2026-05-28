import numpy as np

from geometry.transforms.camera_models import (
    CameraIntrinsics
)


class PointCloudGenerator:

    def __init__(

        self,

        intrinsics: CameraIntrinsics
    ):

        self.intrinsics = intrinsics

    # --------------------------------------------------------
    # Generate point cloud
    # --------------------------------------------------------

    def generate_pointcloud(

        self,

        depth_frame,

        depth_min_m=0.1,
        depth_max_m=5.0,

        stride=4
    ):

        height, width = (
            depth_frame.shape
        )

        points = []

        # ----------------------------------------------------
        # Sparse sampling
        # ----------------------------------------------------

        for v in range(
            0,
            height,
            stride
        ):

            for u in range(
                0,
                width,
                stride
            ):

                depth_mm = depth_frame[
                    v,
                    u
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

                if (
                    depth_m < depth_min_m or
                    depth_m > depth_max_m
                ):
                    continue

                # --------------------------------------------
                # Backprojection
                # --------------------------------------------

                x = (

                    (u - self.intrinsics.cx) *

                    depth_m /

                    self.intrinsics.fx
                )

                y = (

                    (v - self.intrinsics.cy) *

                    depth_m /

                    self.intrinsics.fy
                )

                z = depth_m

                points.append(
                    [x, y, z]
                )

        # ----------------------------------------------------
        # Convert to numpy
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