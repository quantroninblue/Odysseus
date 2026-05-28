"""
tracker_reference.py

Persistent multi-object tracking runtime.

Purpose:
- Maintain persistent object identities
- Manage track lifecycle
- Perform motion-aware association
- Integrate Kalman prediction
- Maintain temporal continuity
- Preserve overlays during dropout
- Preserve persistent geometry state

Tracking Pipeline
-----------------
predict
    ->
associate
    ->
update
    ->
persistence
    ->
lifecycle management
"""

from tracking.track_state import (
    TrackState
)

from tracking.association import (
    associate_tracks_and_detections
)


class MultiObjectTracker:

    def __init__(

        self,

        max_missed_frames=15,

        association_cost_threshold=150.0
    ):

        # --------------------------------------------------------
        # Active tracks
        # --------------------------------------------------------

        self.tracks = []

        # --------------------------------------------------------
        # ID generation
        # --------------------------------------------------------

        self.next_track_id = 0

        # --------------------------------------------------------
        # Lifecycle configuration
        # --------------------------------------------------------

        self.max_missed_frames = (
            max_missed_frames
        )

        self.association_cost_threshold = (
            association_cost_threshold
        )

        # --------------------------------------------------------
        # Runtime bookkeeping
        # --------------------------------------------------------

        self.frame_index = 0

    # ============================================================
    # Prediction
    # ============================================================

    def _predict_tracks(self):
        """
        Predict all active tracks.
        """

        for track in self.tracks:

            if track.is_active:

                track.predict()

    # ============================================================
    # Track creation
    # ============================================================

    def _create_track(
        self,
        detection
    ):
        """
        Create persistent track.
        """

        track = TrackState(

            track_id=self.next_track_id,

            detection=detection
        )

        self.next_track_id += 1

        self.tracks.append(track)

    # ============================================================
    # Track update
    # ============================================================

    def _update_track(
        self,
        track,
        detection
    ):
        """
        Update matched track.
        """

        track.update(
            detection
        )

    # ============================================================
    # Missed handling
    # ============================================================

    def _handle_unmatched_tracks(
        self,
        unmatched_track_indices
    ):
        """
        Handle unmatched tracks.
        """

        for idx in unmatched_track_indices:

            track = self.tracks[idx]

            # ----------------------------------------------------
            # Mark missed
            # ----------------------------------------------------

            track.mark_missed()

            # ----------------------------------------------------
            # Continue prediction
            # ----------------------------------------------------

            track.predict()

    # ============================================================
    # Cleanup
    # ============================================================

    def _remove_dead_tracks(self):
        """
        Remove dead tracks.
        """

        surviving_tracks = []

        for track in self.tracks:

            if (
                track.missed_frames <=
                self.max_missed_frames
            ):

                surviving_tracks.append(
                    track
                )

            else:

                track.deactivate()

        self.tracks = surviving_tracks

    # ============================================================
    # Main update
    # ============================================================

    def update(
        self,
        detections
    ):
        """
        Main tracker update.
        """

        self.frame_index += 1

        # --------------------------------------------------------
        # Initial bootstrap
        # --------------------------------------------------------

        if len(self.tracks) == 0:

            for detection in detections:

                self._create_track(
                    detection
                )

            return self.tracks

        # --------------------------------------------------------
        # Predict all tracks
        # --------------------------------------------------------

        self._predict_tracks()

        # --------------------------------------------------------
        # Association
        # --------------------------------------------------------

        (
            matches,
            unmatched_tracks,
            unmatched_detections

        ) = associate_tracks_and_detections(

            tracks=self.tracks,

            detections=detections,

            max_cost=(
                self.association_cost_threshold
            )
        )

        # --------------------------------------------------------
        # Update matched tracks
        # --------------------------------------------------------

        for (

            track_idx,
            detection_idx

        ) in matches:

            track = self.tracks[
                track_idx
            ]

            detection = detections[
                detection_idx
            ]

            self._update_track(

                track,
                detection
            )

        # --------------------------------------------------------
        # Handle unmatched tracks
        # --------------------------------------------------------

        self._handle_unmatched_tracks(
            unmatched_tracks
        )

        # --------------------------------------------------------
        # Create unmatched detections
        # --------------------------------------------------------

        for det_idx in unmatched_detections:

            detection = detections[
                det_idx
            ]

            self._create_track(
                detection
            )

        # --------------------------------------------------------
        # Remove dead tracks
        # --------------------------------------------------------

        self._remove_dead_tracks()

        return self.tracks

    # ============================================================
    # Active tracks
    # ============================================================

    def get_active_tracks(self):
        """
        Return active tracks only.
        """

        return [

            track for track in self.tracks

            if track.is_active
        ]


# ================================================================
# Validation
# ================================================================

def main():

    print(
        "\n=== MultiObjectTracker Validation ===\n"
    )

    tracker = MultiObjectTracker(

        max_missed_frames=10,

        association_cost_threshold=150.0
    )

    simulated_frames = [

        [
            {
                "center_x": 100,
                "center_y": 100,
                "yaw_rad": 0.0,
                "width_px": 120,
                "height_px": 80
            }
        ],

        [
            {
                "center_x": 110,
                "center_y": 105,
                "yaw_rad": 0.0,
                "width_px": 120,
                "height_px": 80
            }
        ],

        [
            {
                "center_x": 122,
                "center_y": 112,
                "yaw_rad": 0.0,
                "width_px": 120,
                "height_px": 80
            }
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
            {
                "center_x": 145,
                "center_y": 125,
                "yaw_rad": 0.0,
                "width_px": 120,
                "height_px": 80
            }
        ]
    ]

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

        tracks = tracker.update(
            detections
        )

        for track in tracks:

            print(

                f"Track ID: "
                f"{track.track_id} | "

                f"Center: "
                f"({track.center_x:.2f}, "
                f"{track.center_y:.2f}) | "

                f"Smoothed: "
                f"({track.smoothed_center_x:.2f}, "
                f"{track.smoothed_center_y:.2f}) | "

                f"Missed: "
                f"{track.missed_frames} | "

                f"Persistence: "
                f"{track.persistence_frames}"
            )

    print(
        "\nValidation complete."
    )


if __name__ == "__main__":
    main()