import cv2
import numpy as np

from obb_reference import OBBEstimator


def main():

    print("\n=== OBB Validation ===\n")

    fx = 500.87
    fy = 501.20

    estimator = OBBEstimator(
        fx=fx,
        fy=fy
    )

    # ------------------------------------------------------------
    # Synthetic binary mask
    # ------------------------------------------------------------

    mask = np.zeros(
        (640, 640),
        dtype=np.uint8
    )

    rect_center = (320, 320)

    rect_size = (
        180,
        100
    )

    rect_angle_deg = 25

    rect = (
        rect_center,
        rect_size,
        rect_angle_deg
    )

    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    cv2.drawContours(
        mask,
        [box],
        0,
        255,
        -1
    )

    print("Synthetic mask created.")

    # ------------------------------------------------------------
    # OBB estimation
    # ------------------------------------------------------------

    depth_m = 1.0

    result = estimator.estimate_obb(
        binary_mask=mask,
        depth_m=depth_m
    )

    if result is None:

        print("\nERROR: OBB estimation failed.")
        return

    print("\nOBB Estimation Results:\n")

    print(f"Center X: {result['center_x']:.2f}")
    print(f"Center Y: {result['center_y']:.2f}")

    print(f"Width PX : {result['width_px']:.2f}")
    print(f"Height PX: {result['height_px']:.2f}")

    print(f"Width M  : {result['width_m']:.4f}")
    print(f"Height M : {result['height_m']:.4f}")

    print(f"Angle Deg: {result['angle_deg']:.2f}")

    print(f"Yaw Rad  : {result['yaw_rad']:.4f}")

    print(f"Contour Area: {result['contour_area']:.2f}")

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    vis = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.drawContours(
        vis,
        [result["box_points"]],
        0,
        (0, 255, 0),
        2
    )

    center_pt = (
        int(result["center_x"]),
        int(result["center_y"])
    )

    cv2.circle(
        vis,
        center_pt,
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        vis,
        f"Yaw: {result['angle_deg']:.1f} deg",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imwrite(
        "obb_validation_output.png",
        vis
    )

    print("\nVisualization saved:")
    print("obb_validation_output.png")

    print("\nValidation complete.\n")


if __name__ == "__main__":
    main()