"""
track_state.py

Persistent object track state container.

Purpose:
- Maintain object lifecycle state
- Maintain temporal continuity
- Maintain motion state
- Integrate Kalman-based prediction
- Persist masks/OBBs during dropout
- Smooth temporal geometry
"""

from tracking.state_estimation.kalman_filter import (
    KalmanFilter2D
)


class TrackState:

    def __init__(
        self,
        track_id,
        detection
    ):

        # --------------------------------------------------------
        # Identity
        # --------------------------------------------------------

        self.track_id = track_id

        # --------------------------------------------------------
        # Spatial state
        # --------------------------------------------------------

        self.center_x = detection[
            "center_x"
        ]

        self.center_y = detection[
            "center_y"
        ]

        self.yaw_rad = detection[
            "yaw_rad"
        ]

        self.width_px = detection[
            "width_px"
        ]

        self.height_px = detection[
            "height_px"
        ]

        # --------------------------------------------------------
        # Smoothed geometry state
        # --------------------------------------------------------

        self.smoothed_center_x = (
            self.center_x
        )

        self.smoothed_center_y = (
            self.center_y
        )

        self.smoothed_yaw_rad = (
            self.yaw_rad
        )

        self.smoothed_width_px = (
            self.width_px
        )

        self.smoothed_height_px = (
            self.height_px
        )

        # --------------------------------------------------------
        # Velocity state
        # --------------------------------------------------------

        self.velocity_x = 0.0

        self.velocity_y = 0.0

        # --------------------------------------------------------
        # Confidence
        # --------------------------------------------------------

        self.confidence = detection.get(
            "confidence",
            1.0
        )

        # --------------------------------------------------------
        # Lifecycle
        # --------------------------------------------------------

        self.age = 1

        self.missed_frames = 0

        self.is_active = True

        # --------------------------------------------------------
        # Temporal persistence
        # --------------------------------------------------------

        self.persistence_frames = 0

        self.max_persistence_frames = 10

        # --------------------------------------------------------
        # Temporal history
        # --------------------------------------------------------

        self.history = []

        # --------------------------------------------------------
        # Detection references
        # --------------------------------------------------------

        self.mask = detection.get(
            "mask",
            None
        )

        self.obb = detection.get(
            "obb",
            None
        )

        # --------------------------------------------------------
        # Last valid references
        # --------------------------------------------------------

        self.last_valid_mask = (
            self.mask
        )

        self.last_valid_obb = (
            self.obb
        )

        # --------------------------------------------------------
        # Kalman filter
        # --------------------------------------------------------

        self.kalman_filter = (
            KalmanFilter2D()
        )

        self.kalman_filter.initialize(

            x=self.center_x,

            y=self.center_y
        )

        # --------------------------------------------------------
        # EMA smoothing
        # --------------------------------------------------------

        self.ema_alpha = 0.25

        self._append_history()

    # ============================================================
    # Prediction
    # ============================================================

    def predict(self):
        """
        Predict next object state.
        """

        predicted = (
            self.kalman_filter.predict()
        )

        self.center_x = predicted["x"]

        self.center_y = predicted["y"]

        self.velocity_x = predicted["vx"]

        self.velocity_y = predicted["vy"]

        # --------------------------------------------------------
        # Continue persistence during dropout
        # --------------------------------------------------------

        self.persistence_frames += 1

    # ============================================================
    # EMA smoothing
    # ============================================================

    def _ema(
        self,
        previous,
        current
    ):

        return (

            self.ema_alpha * current +

            (1.0 - self.ema_alpha)
            * previous
        )

    # ============================================================
    # Update
    # ============================================================

    def update(
        self,
        detection
    ):
        """
        Update track using new detection.
        """

        # --------------------------------------------------------
        # Kalman update
        # --------------------------------------------------------

        updated = (
            self.kalman_filter.update(

                detection["center_x"],

                detection["center_y"]
            )
        )

        # --------------------------------------------------------
        # Motion state
        # --------------------------------------------------------

        self.center_x = updated["x"]

        self.center_y = updated["y"]

        self.velocity_x = updated["vx"]

        self.velocity_y = updated["vy"]

        # --------------------------------------------------------
        # Geometry update
        # --------------------------------------------------------

        self.yaw_rad = detection[
            "yaw_rad"
        ]

        self.width_px = detection[
            "width_px"
        ]

        self.height_px = detection[
            "height_px"
        ]

        # --------------------------------------------------------
        # EMA smoothing
        # --------------------------------------------------------

        self.smoothed_center_x = self._ema(

            self.smoothed_center_x,

            self.center_x
        )

        self.smoothed_center_y = self._ema(

            self.smoothed_center_y,

            self.center_y
        )

        self.smoothed_width_px = self._ema(

            self.smoothed_width_px,

            self.width_px
        )

        self.smoothed_height_px = self._ema(

            self.smoothed_height_px,

            self.height_px
        )

        self.smoothed_yaw_rad = self._ema(

            self.smoothed_yaw_rad,

            self.yaw_rad
        )

        # --------------------------------------------------------
        # Detection references
        # --------------------------------------------------------

        self.mask = detection.get(
            "mask",
            None
        )

        self.obb = detection.get(
            "obb",
            None
        )

        # --------------------------------------------------------
        # Persist valid geometry
        # --------------------------------------------------------

        if self.mask is not None:

            self.last_valid_mask = (
                self.mask
            )

        if self.obb is not None:

            self.last_valid_obb = (
                self.obb
            )

        # --------------------------------------------------------
        # Reset persistence
        # --------------------------------------------------------

        self.persistence_frames = 0

        # --------------------------------------------------------
        # Lifecycle update
        # --------------------------------------------------------

        self.age += 1

        self.missed_frames = 0

        self.is_active = True

        self._append_history()

    # ============================================================
    # Missed handling
    # ============================================================

    def mark_missed(self):
        """
        Mark track as missed.
        """

        self.missed_frames += 1

        # --------------------------------------------------------
        # Persist previous geometry
        # --------------------------------------------------------

        if self.persistence_frames < (
            self.max_persistence_frames
        ):

            self.mask = (
                self.last_valid_mask
            )

            self.obb = (
                self.last_valid_obb
            )

        else:

            self.mask = None

            self.obb = None

    # ============================================================
    # Deactivation
    # ============================================================

    def deactivate(self):
        """
        Deactivate track.
        """

        self.is_active = False

    # ============================================================
    # History
    # ============================================================

    def _append_history(self):
        """
        Store temporal history.
        """

        state_snapshot = {

            "center_x":
                self.center_x,

            "center_y":
                self.center_y,

            "velocity_x":
                self.velocity_x,

            "velocity_y":
                self.velocity_y,

            "yaw_rad":
                self.yaw_rad
        }

        self.history.append(
            state_snapshot
        )

        if len(self.history) > 50:

            self.history.pop(0)

    # ============================================================
    # Serialization
    # ============================================================

    def to_dict(self):
        """
        Convert track state to dictionary.
        """

        return {

            "track_id": self.track_id,

            "center_x": self.center_x,

            "center_y": self.center_y,

            "yaw_rad": self.yaw_rad,

            "width_px": self.width_px,

            "height_px": self.height_px,

            "velocity_x": self.velocity_x,

            "velocity_y": self.velocity_y,

            "confidence": self.confidence,

            "age": self.age,

            "missed_frames": self.missed_frames,

            "is_active": self.is_active,

            "persistence_frames":
                self.persistence_frames
        }


def main():

    print(
        "\n=== TrackState Validation ===\n"
    )

    detection = {

        "center_x": 320,

        "center_y": 240,

        "yaw_rad": 0.0,

        "width_px": 120,

        "height_px": 80,

        "confidence": 1.0
    }

    track = TrackState(

        track_id=1,

        detection=detection
    )

    print(
        track.to_dict()
    )

    print(
        "\nValidation complete."
    )


if __name__ == "__main__":
    main()