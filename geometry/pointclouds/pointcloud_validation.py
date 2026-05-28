import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from rosbags.highlevel import AnyReader

import numpy as np

from geometry.transforms.camera_models import (
    CameraIntrinsics
)

from geometry.pointclouds.pointcloud_generation import (
    PointCloudGenerator
)


# ------------------------------------------------------------
# ROS bag path
# ------------------------------------------------------------

BAG_PATH = Path(
    "datasets/rosbags/rosbags/"
    "metric_depth_val_1779181947"
)


def main():

    print(
        "\n=== Point Cloud Validation ===\n"
    )

    # --------------------------------------------------------
    # Approx depth intrinsics
    # --------------------------------------------------------

    depth_intrinsics = CameraIntrinsics(

        fx=1502.61,
        fy=845.775,

        cx=999.90,
        cy=534.026,

        width=1920,
        height=1080
    )

    # --------------------------------------------------------
    # Point cloud generator
    # --------------------------------------------------------

    generator = PointCloudGenerator(
        depth_intrinsics
    )

    # --------------------------------------------------------
    # Read ROS bag
    # --------------------------------------------------------

    with AnyReader(
        [BAG_PATH]
    ) as reader:

        for connection, timestamp, rawdata in reader.messages():

            if connection.topic != "/vctr/depth_raw":
                continue

            msg = reader.deserialize(
                rawdata,
                connection.msgtype
            )

            print(
                "Depth frame loaded.\n"
            )

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

            # ------------------------------------------------
            # Generate point cloud
            # ------------------------------------------------

            pointcloud = (
                generator.generate_pointcloud(

                    depth_frame=depth,

                    depth_min_m=0.1,
                    depth_max_m=5.0,

                    stride=8
                )
            )

            print()

            print(
                f"Point Cloud Shape: "
                f"{pointcloud.shape}"
            )

            print()

            if len(pointcloud) > 0:

                print(
                    "Sample Points:\n"
                )

                for idx in range(

                    min(10, len(pointcloud))
                ):

                    x, y, z = (
                        pointcloud[idx]
                    )

                    print(

                        f"[{idx}] "

                        f"X={x:.3f}  "

                        f"Y={y:.3f}  "

                        f"Z={z:.3f}"
                    )

            break

    print(
        "\nValidation complete.\n"
    )


if __name__ == "__main__":
    main()