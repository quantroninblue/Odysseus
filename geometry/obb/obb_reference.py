"""
obb_reference.py

Reference-oriented bounding box geometry extraction.

Purpose:
- Extract contour geometry
- Estimate 2D oriented bounding boxes
- Estimate object orientation (yaw)
- Estimate object dimensions from image-space geometry

Extracted from:
    _measure_box_face()

inside vision_node2.py
"""

import cv2
import math
import numpy as np


class OBBEstimator:

    def __init__(
        self,
        fx,
        fy
    ):

        self.fx = fx
        self.fy = fy

    def estimate_obb(
        self,
        binary_mask,
        depth_m
    ):
        """
        Estimate oriented bounding box geometry
        from a binary object mask.

        Parameters
        ----------
        binary_mask : np.ndarray
            Binary segmentation mask

        depth_m : float
            Object depth in meters

        Returns
        -------
        dict or None
        """

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        contour = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = cv2.contourArea(contour)

        if contour_area < 50:
            return None

        rect = cv2.minAreaRect(contour)

        center_xy = rect[0]

        width_px = rect[1][0]
        height_px = rect[1][1]

        angle_deg = rect[2]

        if width_px < height_px:

            width_px, height_px = (
                height_px,
                width_px
            )

            angle_deg += 90.0

        while angle_deg >= 90.0:
            angle_deg -= 180.0

        while angle_deg < -90.0:
            angle_deg += 180.0

        yaw_rad = math.radians(angle_deg)

        width_m = (
            width_px *
            depth_m /
            self.fx
        )

        height_m = (
            height_px *
            depth_m /
            self.fy
        )

        box_points = cv2.boxPoints(rect)
        box_points = box_points.astype(np.int32)

        return {

            "contour": contour,

            "rect": rect,

            "box_points": box_points,

            "center_x": center_xy[0],
            "center_y": center_xy[1],

            "width_px": width_px,
            "height_px": height_px,

            "width_m": width_m,
            "height_m": height_m,

            "angle_deg": angle_deg,
            "yaw_rad": yaw_rad,

            "contour_area": contour_area
        }


def main():

    print("\nOBB reference module loaded.\n")


if __name__ == "__main__":
    main()