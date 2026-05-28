import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

from rosbags.highlevel import AnyReader

import numpy as np
import cv2


# ------------------------------------------------------------
# ROS Bag Path
# ------------------------------------------------------------

BAG_PATH = Path(
    "datasets/rosbags/rosbags/"
    "metric_depth_val_1779181947"
)


def main():

    print(
        "\n=== RGB Depth Alignment Validation ===\n"
    )

    rgb_frame = None

    depth_frame = None

    # --------------------------------------------------------
    # Read synchronized frames
    # --------------------------------------------------------

    with AnyReader(
        [BAG_PATH]
    ) as reader:

        for connection, timestamp, rawdata in reader.messages():

            msg = reader.deserialize(
                rawdata,
                connection.msgtype
            )

            # ------------------------------------------------
            # RGB
            # ------------------------------------------------

            if (
                connection.topic == "/vctr/rgb_raw"
                and rgb_frame is None
            ):

                print(
                    "Loading RGB frame..."
                )

                rgb = np.frombuffer(
                    msg.data,
                    dtype=np.uint8
                )

                rgb = rgb.reshape(
                    msg.height,
                    msg.width,
                    3
                )

                rgb_frame = rgb

                print(
                    f"RGB Shape: "
                    f"{rgb_frame.shape}"
                )

            # ------------------------------------------------
            # Depth
            # ------------------------------------------------

            if (
                connection.topic == "/vctr/depth_raw"
                and depth_frame is None
            ):

                print(
                    "Loading depth frame..."
                )

                depth = np.frombuffer(
                    msg.data,
                    dtype=np.uint16
                )

                depth = depth.reshape(
                    msg.height,
                    msg.width
                )

                depth_frame = depth

                print(
                    f"Depth Shape: "
                    f"{depth_frame.shape}"
                )

            # ------------------------------------------------
            # Stop once both loaded
            # ------------------------------------------------

            if (
                rgb_frame is not None and
                depth_frame is not None
            ):
                break

    # --------------------------------------------------------
    # Normalize depth visualization
    # --------------------------------------------------------

    depth_vis = cv2.normalize(

        depth_frame,

        None,

        0,

        255,

        cv2.NORM_MINMAX
    )

    depth_vis = depth_vis.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Apply colormap
    # --------------------------------------------------------

    depth_vis = cv2.applyColorMap(

        depth_vis,

        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # Resize depth to RGB space
    # --------------------------------------------------------

    depth_resized = cv2.resize(

        depth_vis,

        (
            rgb_frame.shape[1],
            rgb_frame.shape[0]
        ),

        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Overlay visualization
    # --------------------------------------------------------

    overlay = cv2.addWeighted(

        rgb_frame,
        0.7,

        depth_resized,
        0.3,

        0
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    cv2.imwrite(

        "geometry_validation/rgb_alignment_rgb.png",

        rgb_frame
    )

    cv2.imwrite(

        "geometry_validation/rgb_alignment_depth.png",

        depth_resized
    )

    cv2.imwrite(

        "geometry_validation/rgb_depth_overlay.png",

        overlay
    )

    print(
        "\nSaved alignment outputs:"
    )

    print(
        "rgb_alignment_rgb.png"
    )

    print(
        "rgb_alignment_depth.png"
    )

    print(
        "rgb_depth_overlay.png"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "RGB",
        rgb_frame
    )

    cv2.imshow(
        "Depth",
        depth_resized
    )

    cv2.imshow(
        "RGB Depth Overlay",
        overlay
    )

    cv2.waitKey(1)

    cv2.destroyAllWindows()

    print(
        "\nAlignment validation complete.\n"
    )


if __name__ == "__main__":
    main()