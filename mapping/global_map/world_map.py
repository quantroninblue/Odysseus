
import numpy as np


class WorldMap:

    """
    Persistent world-frame spatial map.

    Stores:
    - accumulated semantic pointclouds
    - future semantic entities
    - future landmarks
    """

    def __init__(self):

        self.global_points = []

    def add_points(

        self,

        points_world: np.ndarray

    ):

        if len(points_world) == 0:

            return

        self.global_points.append(
            points_world.copy()
        )

    def get_all_points(self):

        if not self.global_points:

            return np.empty((0, 3))

        return np.concatenate(
            self.global_points,
            axis=0
        )

