import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from rosbags.highlevel import AnyReader

import numpy as np

from geometry.backprojection.backprojection_reference import (
    Backprojector
)


# ------------------------------------------------------------
# ROS Bag Path
# ------------------------------------------------------------

BAG_PATH = Path(
    "datasets/rosbags/rosbags/"
    "metric_depth_val_1779181947"
)

# ------------------------------------------------------------
# OAK-D Intrinsics
# ------------------------------------------------------------

FX = 500.87
FY = 501.20

CX = 333.30
CY = 316.46

# ------------------------------------------------------------
# Backprojector
# ------------------------------------------------------------

backprojector = Backprojector(

    fx=FX,
    fy=FY,

    cx=CX,
    cy=CY
)


def main():

    print("\n=== Backprojection Validation ===\n")

    with AnyReader(
        [BAG_PATH]
    ) as reader:

        for connection, timestamp, rawdata in reader.messages():

            # ------------------------------------------------
            # Depth stream only
            # ------------------------------------------------

            if connection.topic != "/vctr/depth_raw":
                continue

            msg = reader.deserialize(
                rawdata,
                connection.msgtype
            )

            print("Depth frame loaded.\n")

            # ------------------------------------------------
            # Decode depth image
            # ------------------------------------------------

            depth = np.frombuffer(
                msg.data,
                dtype=np.uint16
            )

            depth = depth.reshape(
                msg.height,
                msg.width
            )

            print(
                f"Depth Shape: "
                f"{depth.shape}"
            )

            print(
                f"Depth Encoding: "
                f"{msg.encoding}"
            )

            print()

            # ------------------------------------------------
            # Sample pixels
            # ------------------------------------------------

            sample_pixels = [

                (960, 540),

                (700, 500),

                (1100, 600),

                (1300, 700)
            ]

            print(
                "\nSample Backprojections:\n"
            )

            for (u, v) in sample_pixels:

                depth_mm = depth[v, u]

                # --------------------------------------------
                # Invalid depth
                # --------------------------------------------

                if (
                    depth_mm == 0 or
                    depth_mm == 65535
                ):

                    print(
                        f"Pixel ({u}, {v}) "
                        f"-> INVALID DEPTH"
                    )

                    continue

                # --------------------------------------------
                # Convert mm -> meters
                # --------------------------------------------

                depth_m = (
                    depth_mm / 1000.0
                )

                # --------------------------------------------
                # Camera-frame backprojection
                # --------------------------------------------

                x = (
                    (u - CX) *
                    depth_m / FX
                )

                y = (
                    (v - CY) *
                    depth_m / FY
                )

                z = depth_m

                # --------------------------------------------
                # Print results
                # --------------------------------------------

                print("=" * 60)

                print(
                    f"Pixel: ({u}, {v})"
                )

                print(
                    f"Depth: {depth_m:.3f} m"
                )

                print()

                print("3D Camera Point:")

                print(
                    f"X = {x:.3f} m"
                )

                print(
                    f"Y = {y:.3f} m"
                )

                print(
                    f"Z = {z:.3f} m"
                )

                print()

            break

    print("\nValidation complete.\n")


if __name__ == "__main__":
    main()