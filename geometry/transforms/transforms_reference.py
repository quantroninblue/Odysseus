import numpy as np


def tf_to_matrix(transform) -> np.ndarray:
    """
    Convert a ROS TF transform into a 4x4 homogeneous transform matrix.

    Extracted from:
    vision_node2.py

    Original source:
    VctrVisionNode._tf_to_matrix()
    """

    tx = transform.transform.translation.x
    ty = transform.transform.translation.y
    tz = transform.transform.translation.z

    qx = transform.transform.rotation.x
    qy = transform.transform.rotation.y
    qz = transform.transform.rotation.z
    qw = transform.transform.rotation.w

    rotation_matrix = np.array([
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw)
        ],

        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw)
        ],

        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy)
        ]

    ], dtype=np.float64)

    transform_matrix = np.eye(4, dtype=np.float64)

    transform_matrix[:3, :3] = rotation_matrix

    transform_matrix[:3, 3] = [
        tx,
        ty,
        tz
    ]

    return transform_matrix