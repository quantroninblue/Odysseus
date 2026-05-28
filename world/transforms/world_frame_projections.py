
import numpy as np


class WorldFrameProjector:

    """
    Projects camera-frame geometry into
    persistent world coordinates.

    Input:
        - camera-frame pointcloud
        - T_world_cam

    Output:
        - world-frame pointcloud
    """

    @staticmethod
    def project_points_to_world(

        points_camera: np.ndarray,

        T_world_cam: np.ndarray

    ) -> np.ndarray:

        """
        Parameters
        ----------
        points_camera : (N, 3)
            Pointcloud in camera coordinates

        T_world_cam : (4, 4)
            Camera pose in world frame

        Returns
        -------
        points_world : (N, 3)
            Pointcloud transformed into world coordinates
        """

        if len(points_camera) == 0:

            return np.empty((0, 3))

        ones = np.ones(
            (len(points_camera), 1),
            dtype=np.float32
        )

        points_h = np.concatenate(
            [points_camera, ones],
            axis=1
        )

        points_world_h = (
            T_world_cam @ points_h.T
        ).T

        return points_world_h[:, :3]

