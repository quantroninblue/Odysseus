"""
extrinsics.py

Rigid camera-frame transforms.

Defines transforms between:
- depth camera
- RGB camera
- world frame (future)

This becomes the canonical transform layer
for all RGBD geometry fusion.
"""

import numpy as np


class ExtrinsicTransform:

    def __init__(

        self,

        rotation_matrix,

        translation_vector
    ):

        self.R = np.array(

            rotation_matrix,

            dtype=np.float32
        )

        self.t = np.array(

            translation_vector,

            dtype=np.float32
        ).reshape(3)

    # --------------------------------------------------------
    # Transform point
    # --------------------------------------------------------

    def transform_point(

        self,

        point_3d
    ):

        point_3d = np.array(

            point_3d,

            dtype=np.float32
        ).reshape(3)

        transformed = (

            self.R @ point_3d +

            self.t
        )

        return transformed

    # --------------------------------------------------------
    # Transform point cloud
    # --------------------------------------------------------

    def transform_pointcloud(

        self,

        pointcloud
    ):

        if len(pointcloud) == 0:

            return pointcloud

        transformed = (

            pointcloud @ self.R.T +

            self.t
        )

        return transformed.astype(
            np.float32
        )


# ============================================================
# OAK-D Approximate Extrinsics
# ============================================================

# IMPORTANT:
# These are placeholder bootstrap transforms.
#
# Proper calibration extraction from OAK-D
# should replace these later.
#
# Units:
# meters
# ============================================================

DEPTH_TO_RGB_EXTRINSIC = ExtrinsicTransform(

    rotation_matrix=[

        [1.0, 0.0, 0.0],

        [0.0, 1.0, 0.0],

        [0.0, 0.0, 1.0]
    ],

    translation_vector=[

        0.03,   # ~3 cm baseline

        0.0,

        0.0
    ]
)