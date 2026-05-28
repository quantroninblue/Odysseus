"""
backprojection_reference.py

Reference extraction from vision_node2.py

Purpose:
- Convert depth pixels into 3D camera-frame coordinates
- Transform camera-frame coordinates into world-frame coordinates
- Build sparse point clouds from depth ROIs
- Expose BOTH camera-frame and world-frame geometry

This module is a direct modular extraction of:
    _backproject_roi_to_world()

from the original monolithic perception pipeline.
"""

import numpy as np


class Backprojector:

    def __init__(
        self,
        fx,
        fy,
        cx,
        cy,
        cloud_stride=4
    ):

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.cloud_stride = cloud_stride

    def backproject_roi_to_world(
        self,
        depth_frame,
        img_x1,
        img_y1,
        img_x2,
        img_y2,
        tf_matrix,
        depth_min_mm,
        depth_max_mm,
        exclude_inner_img=None,
        img_w=640,
        img_h=640
    ):
        """
        Convert a depth ROI into:
        - sparse camera-frame point cloud
        - sparse world-frame point cloud

        Returns:
        --------
        wx, wy, wz :
            World-frame XYZ arrays

        X_cam, Y_cam, Z_cam :
            Camera-frame XYZ arrays
        """

        dep_h, dep_w = depth_frame.shape[:2]

        sx = dep_w / img_w
        sy = dep_h / img_h

        stride = self.cloud_stride

        dx1 = max(0, int(img_x1 * sx))
        dy1 = max(0, int(img_y1 * sy))

        dx2 = min(dep_w, int(img_x2 * sx))
        dy2 = min(dep_h, int(img_y2 * sy))

        # --------------------------------------------------------
        # Invalid ROI
        # --------------------------------------------------------

        if dx2 <= dx1 or dy2 <= dy1:

            return (
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0)
            )

        # --------------------------------------------------------
        # Sparse sampling grid
        # --------------------------------------------------------

        rows = np.arange(
            dy1,
            dy2,
            stride
        )

        cols = np.arange(
            dx1,
            dx2,
            stride
        )

        rr, cc = np.meshgrid(
            rows,
            cols,
            indexing='ij'
        )

        # --------------------------------------------------------
        # Sample depth
        # --------------------------------------------------------

        depth_sampled = depth_frame[
            rr,
            cc
        ].astype(np.float32)

        # --------------------------------------------------------
        # Depth validity filtering
        # --------------------------------------------------------

        valid = (
            (depth_sampled > depth_min_mm) &
            (depth_sampled < depth_max_mm)
        )

        # --------------------------------------------------------
        # Optional exclusion ROI
        # --------------------------------------------------------

        if exclude_inner_img is not None:

            ex1, ey1, ex2, ey2 = exclude_inner_img

            in_inner = (

                (rr >= max(0, int(ey1 * sy))) &
                (rr < min(dep_h, int(ey2 * sy))) &

                (cc >= max(0, int(ex1 * sx))) &
                (cc < min(dep_w, int(ex2 * sx)))
            )

            valid &= ~in_inner

        # --------------------------------------------------------
        # No valid points
        # --------------------------------------------------------

        if valid.sum() == 0:

            return (
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0)
            )

        # --------------------------------------------------------
        # Metric depth conversion
        # --------------------------------------------------------

        Z_cam = (
            depth_sampled[valid]
            .astype(np.float64) / 1000.0
        )

        # --------------------------------------------------------
        # RGB-space coordinates
        # --------------------------------------------------------

        u_rgb = (
            cc[valid]
            .astype(np.float64) / sx
        )

        v_rgb = (
            rr[valid]
            .astype(np.float64) / sy
        )

        # --------------------------------------------------------
        # Camera-frame backprojection
        # --------------------------------------------------------

        X_cam = (
            (u_rgb - self.cx) *
            Z_cam / self.fx
        )

        Y_cam = (
            (v_rgb - self.cy) *
            Z_cam / self.fy
        )

        # --------------------------------------------------------
        # Homogeneous camera-frame points
        # --------------------------------------------------------

        pts_h = np.vstack([
            X_cam,
            Y_cam,
            Z_cam,
            np.ones(
                len(Z_cam),
                dtype=np.float64
            )
        ])

        # --------------------------------------------------------
        # World-frame transform
        # --------------------------------------------------------

        world_pts = tf_matrix @ pts_h

        wx = world_pts[0]
        wy = world_pts[1]
        wz = world_pts[2]

        # --------------------------------------------------------
        # Return BOTH camera + world coordinates
        # --------------------------------------------------------

        return (
            wx,
            wy,
            wz,
            X_cam,
            Y_cam,
            Z_cam
        )


def main():

    print(
        "\nBackprojection reference module loaded.\n"
    )

    fx = 500.87
    fy = 501.20
    cx = 333.30
    cy = 316.46

    bp = Backprojector(
        fx,
        fy,
        cx,
        cy,
        cloud_stride=8
    )

    depth_frame = np.full(
        (640, 640),
        1000,
        dtype=np.uint16
    )

    tf_matrix = np.eye(4)

    (
        wx,
        wy,
        wz,
        X_cam,
        Y_cam,
        Z_cam
    ) = bp.backproject_roi_to_world(

        depth_frame=depth_frame,

        img_x1=200,
        img_y1=200,
        img_x2=400,
        img_y2=400,

        tf_matrix=tf_matrix,

        depth_min_mm=500,
        depth_max_mm=2000
    )

    print("Point cloud size:")
    print(len(wx))

    print("\nSample world point:")
    print(wx[0], wy[0], wz[0])

    print("\nSample camera point:")
    print(
        X_cam[0],
        Y_cam[0],
        Z_cam[0]
    )


if __name__ == "__main__":
    main()