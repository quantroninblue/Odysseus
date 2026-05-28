import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from geometry.transforms.camera_models import (
    CameraIntrinsics
)

from geometry.transforms.depth_to_rgb_projection import (
    DepthToRGBProjector
)


def main():

    print(
        "\n=== Depth To RGB Projection Validation ===\n"
    )

    # --------------------------------------------------------
    # RGB intrinsics
    # --------------------------------------------------------

    rgb_intrinsics = CameraIntrinsics(

        fx=500.87,
        fy=501.20,

        cx=333.30,
        cy=316.46,

        width=640,
        height=640
    )

    # --------------------------------------------------------
    # Approx depth intrinsics
    # --------------------------------------------------------

    depth_intrinsics = (
        rgb_intrinsics.scaled_to_resolution(

            1920,
            1080
        )
    )

    # --------------------------------------------------------
    # Projector
    # --------------------------------------------------------

    projector = DepthToRGBProjector(

        depth_intrinsics=depth_intrinsics,

        rgb_intrinsics=rgb_intrinsics
    )

    # --------------------------------------------------------
    # Sample depth pixel
    # --------------------------------------------------------

    u_depth = 1300
    v_depth = 700

    depth_m = 0.359

    print(
        f"Depth Pixel: "
        f"({u_depth}, {v_depth})"
    )

    print(
        f"Depth: "
        f"{depth_m:.3f} m"
    )

    # --------------------------------------------------------
    # Reproject
    # --------------------------------------------------------

    rgb_pixel = (
        projector.depth_pixel_to_rgb_pixel(

            u_depth=u_depth,
            v_depth=v_depth,

            depth_m=depth_m
        )
    )

    print()

    print(
        f"Projected RGB Pixel: "
        f"{rgb_pixel}"
    )

    print()

    print(
        "Validation complete.\n"
    )


if __name__ == "__main__":
    main()