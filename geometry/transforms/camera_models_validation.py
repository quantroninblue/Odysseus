import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from geometry.transforms.camera_models import (
    CameraIntrinsics
)


def main():

    print(
        "\n=== Camera Model Validation ===\n"
    )

    # --------------------------------------------------------
    # RGB Intrinsics
    # --------------------------------------------------------

    rgb_intrinsics = CameraIntrinsics(

        fx=500.87,
        fy=501.20,

        cx=333.30,
        cy=316.46,

        width=640,
        height=640
    )

    print("RGB Intrinsics:")
    rgb_intrinsics.print_summary()

    # --------------------------------------------------------
    # Depth-scaled intrinsics
    # --------------------------------------------------------

    depth_intrinsics = (
        rgb_intrinsics.scaled_to_resolution(

            1920,
            1080
        )
    )

    print("Depth Intrinsics:")
    depth_intrinsics.print_summary()


if __name__ == "__main__":
    main()