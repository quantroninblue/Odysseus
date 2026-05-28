"""
tracking_validation.py

Validation runtime for:
- Kalman prediction
- Temporal smoothing
- Motion-aware tracking
- Persistent object identity
- Missed-frame continuity

Tracking Pipeline
-----------------
predict
    ->
associate
    ->
update
"""

import random
import time

from tracking.tracker_reference import (
    MultiObjectTracker
)


def build_detection(
    center_x,
    center_y,
    noise_std=0.0
):
    """
    Build noisy synthetic detection.
    """

    noisy_x = (
        center_x +
        random.uniform(
            -noise_std,
            noise_std
        )
    )

    noisy_y = (
        center_y +
        random.uniform(
            -noise_std,
            noise_std
        )
    )

    return {

        "center_x": noisy_x,

        "center_y": noisy_y,

        "yaw_rad": 0.0,

        "width_px": 120,

        "height_px": 80,

        "confidence": 1.0
    }


def print_tracks(
    tracks
):
    """
    Print active tracker state.
    """

    print("\nActive Tracks:\n")

    if len(tracks) == 0:

        print("No active tracks.")

        return

    for track in tracks:

        print(

            f"Track ID: {track.track_id} | "

            f"Center: "
            f"({track.center_x:.2f}, "
            f"{track.center_y:.2f}) | "

            f"Velocity: "
            f"({track.velocity_x:.2f}, "
            f"{track.velocity_y:.2f}) | "

            f"Missed: "
            f"{track.missed_frames} | "

            f"Age: "
            f"{track.age}"
        )


def main():

    print(
        "\n=== Kalman Tracking Validation ===\n"
    )

    # ------------------------------------------------------------
    # Tracker
    # ------------------------------------------------------------

    tracker = MultiObjectTracker(

        max_missed_frames=5,

        association_distance=150.0
    )

    # ------------------------------------------------------------
    # Simulated trajectory
    # ------------------------------------------------------------

    simulated_frames = [

        # --------------------------------------------------------
        # Stable motion
        # --------------------------------------------------------

        [

            build_detection(
                100,
                100,
                noise_std=8
            )
        ],

        [

            build_detection(
                112,
                108,
                noise_std=8
            )
        ],

        [

            build_detection(
                126,
                118,
                noise_std=8
            )
        ],

        [

            build_detection(
                141,
                129,
                noise_std=8
            )
        ],

        # --------------------------------------------------------
        # Temporary dropout
        # --------------------------------------------------------

        [],

        [],

        # --------------------------------------------------------
        # Reappearance
        # --------------------------------------------------------

        [

            build_detection(
                175,
                152,
                noise_std=8
            )
        ],

        [

            build_detection(
                191,
                164,
                noise_std=8
            )
        ],

        [

            build_detection(
                208,
                178,
                noise_std=8
            )
        ]
    ]

    # ------------------------------------------------------------
    # Runtime loop
    # ------------------------------------------------------------

    for frame_idx, detections in enumerate(
        simulated_frames
    ):

        print(
            "\n================================="
        )

        print(
            f"Frame {frame_idx}"
        )

        print(
            "================================="
        )

        print(
            f"\nDetections: "
            f"{len(detections)}"
        )

        # --------------------------------------------------------
        # Tracker update
        # --------------------------------------------------------

        tracks = tracker.update(
            detections
        )

        # --------------------------------------------------------
        # Print tracker state
        # --------------------------------------------------------

        print_tracks(
            tracks
        )

        # --------------------------------------------------------
        # Print temporal history
        # --------------------------------------------------------

        for track in tracks:

            print(
                f"\nTrack {track.track_id} "
                f"History Size: "
                f"{len(track.history)}"
            )

        time.sleep(0.5)

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print(
        "\n=== Validation Complete ===\n"
    )

    print(
        "Expected behavior:"
    )

    print(
        "- Stable persistent IDs"
    )

    print(
        "- Reduced jitter"
    )

    print(
        "- Smooth motion continuity"
    )

    print(
        "- Track survival during dropout"
    )

    print(
        "- Motion-aware prediction"
    )


if __name__ == "__main__":
    main()