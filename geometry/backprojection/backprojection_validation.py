import numpy as np

from backprojection_reference import Backprojector


def main():

    print("\n=== Backprojection Validation ===\n")

    fx = 500.87
    fy = 501.20
    cx = 333.30
    cy = 316.46

    print("Camera Intrinsics:")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    depth_frame = np.full(
        (640, 640),
        1000,
        dtype=np.uint16
    )

    print("\nDepth Frame Shape:")
    print(depth_frame.shape)

    tf_matrix = np.eye(4)

    print("\nTransform Matrix:")
    print(tf_matrix)

    bp = Backprojector(
        fx,
        fy,
        cx,
        cy,
        cloud_stride=8
    )

    wx, wy, wz = bp.backproject_roi_to_world(
        depth_frame=depth_frame,

        img_x1=200,
        img_y1=200,
        img_x2=400,
        img_y2=400,

        tf_matrix=tf_matrix,

        depth_min_mm=500,
        depth_max_mm=2000
    )

    print("\nGenerated Point Cloud Size:")
    print(len(wx))

    if len(wx) == 0:
        print("\nERROR: Empty point cloud.")
        return

    print("\nSample Points:\n")

    sample_count = min(5, len(wx))

    for i in range(sample_count):

        print(
            f"Point {i}: "
            f"X={wx[i]:.4f}, "
            f"Y={wy[i]:.4f}, "
            f"Z={wz[i]:.4f}"
        )

    print("\nPoint Cloud Statistics:\n")

    print(f"X Range: {wx.min():.4f} -> {wx.max():.4f}")
    print(f"Y Range: {wy.min():.4f} -> {wy.max():.4f}")
    print(f"Z Range: {wz.min():.4f} -> {wz.max():.4f}")

    print("\nValidation complete.\n")


if __name__ == "__main__":
    main()