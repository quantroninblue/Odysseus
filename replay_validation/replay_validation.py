"""
replay_validation.py

Canonical semantic-spatial replay runtime.

Architecture
------------
MCAP Replay
    ->
Segmentation
    ->
Tracking
    ->
Object Point Cloud Extraction
    ->
Canonical RGBD Reprojection
    ->
Projected Point Cloud Visualization
"""

import os
import cv2
import numpy as np

from ingestion.rosbags.mcap_replay_loader import (
    MCAPReplayLoader
)

from segmentation.segmentation_reference import (
    SegmentationModule
)

from tracking.tracker_reference import (
    MultiObjectTracker
)

from geometry.transforms.camera_models import (
    CameraIntrinsics
)

from geometry.transforms.depth_to_rgb_projection import (
    DepthToRGBProjector
)

from geometry.pointclouds.pointcloud_generation import (
    PointCloudGenerator
)

from geometry.pointclouds.object_pointclouds import (
    ObjectPointCloudExtractor
)


# ============================================================
# Detection Builder
# ============================================================

def build_detection_from_obb(
    obb,
    mask=None
):

    return {

        "center_x": float(
            obb["center_x"]
        ),

        "center_y": float(
            obb["center_y"]
        ),

        "yaw_rad": float(
            obb["yaw_rad"]
        ),

        "width_px": float(
            obb["width_px"]
        ),

        "height_px": float(
            obb["height_px"]
        ),

        "confidence": 1.0,

        "mask": mask,

        "obb": obb
    }


# ============================================================
# Mask Overlay
# ============================================================

def draw_mask_overlay(
    frame,
    mask
):

    if mask is None:
        return frame

    if len(mask.shape) != 2:
        return frame

    colored_mask = np.zeros_like(frame)

    colored_mask[:, :, 1] = (
        mask * 180
    ).astype(np.uint8)

    frame = cv2.addWeighted(

        frame,
        1.0,

        colored_mask,
        0.35,

        0
    )

    return frame


# ============================================================
# OBB Overlay
# ============================================================

def draw_obb_overlay(
    frame,
    obb
):

    if obb is None:
        return frame

    if "box_points" not in obb:
        return frame

    box_points = obb[
        "box_points"
    ]

    cv2.polylines(

        frame,

        [box_points],

        True,

        (0, 255, 255),

        2
    )

    return frame


# ============================================================
# Canonical Point Cloud Projection
# ============================================================

def draw_projected_pointcloud(

    frame,

    pointcloud,

    projector
):

    if pointcloud is None:
        return frame

    if len(pointcloud) == 0:
        return frame

    projected_pixels = (

        projector.pointcloud_to_rgb_pixels(
            pointcloud
        )
    )

    for (u, v) in projected_pixels:

        # ----------------------------------------------------
        # Bounds check
        # ----------------------------------------------------

        if (
            u < 0 or
            u >= frame.shape[1]
        ):
            continue

        if (
            v < 0 or
            v >= frame.shape[0]
        ):
            continue

        # ----------------------------------------------------
        # Draw point
        # ----------------------------------------------------

        cv2.circle(

            frame,

            (u, v),

            1,

            (255, 255, 0),

            -1
        )

    return frame


# ============================================================
# Geometry Telemetry
# ============================================================

def draw_geometry_stats(

    frame,

    stats,

    x,
    y
):

    if stats is None:
        return frame

    dimensions = stats[
        "dimensions"
    ]

    centroid = stats[
        "centroid"
    ]

    telemetry_lines = [

        f"PTS: {stats['point_count']}",

        (
            f"DIM: "
            f"{dimensions[0]:.2f} "
            f"{dimensions[1]:.2f} "
            f"{dimensions[2]:.2f}"
        ),

        (
            f"CEN: "
            f"{centroid[0]:.2f} "
            f"{centroid[1]:.2f} "
            f"{centroid[2]:.2f}"
        )
    ]

    offset_y = y

    for line in telemetry_lines:

        cv2.putText(

            frame,

            line,

            (x, offset_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 255),

            2
        )

        offset_y += 22

    return frame


# ============================================================
# Track Overlay
# ============================================================

def draw_track_overlay(

    frame,

    track,

    geometry_stats=None
):

    obb = track.last_valid_obb

    mask = track.last_valid_mask

    # --------------------------------------------------------
    # Persistence decay
    # --------------------------------------------------------

    if track.missed_frames > 2:

        mask = None
        obb = None

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    frame = draw_mask_overlay(
        frame,
        mask
    )

    # --------------------------------------------------------
    # OBB
    # --------------------------------------------------------

    frame = draw_obb_overlay(
        frame,
        obb
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center = (

        int(
            track.smoothed_center_x
        ),

        int(
            track.smoothed_center_y
        )
    )

    cv2.circle(

        frame,

        center,

        5,

        (0, 0, 255),

        -1
    )

    # --------------------------------------------------------
    # Motion vector
    # --------------------------------------------------------

    predicted_center = (

        int(
            track.smoothed_center_x +
            track.velocity_x
        ),

        int(
            track.smoothed_center_y +
            track.velocity_y
        )
    )

    cv2.line(

        frame,

        center,

        predicted_center,

        (255, 0, 0),

        2
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (

        f"ID:{track.track_id} "

        f"M:{track.missed_frames} "

        f"P:{track.persistence_frames}"
    )

    cv2.putText(

        frame,

        label,

        (
            center[0] + 10,
            center[1]
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )

    # --------------------------------------------------------
    # Geometry telemetry
    # --------------------------------------------------------

    frame = draw_geometry_stats(

        frame,

        geometry_stats,

        center[0] + 10,

        center[1] + 25
    )

    return frame


# ============================================================
# Runtime Telemetry
# ============================================================

def draw_runtime_telemetry(

    frame,

    frame_id,

    inference_ms,

    masks_count,

    obb_count,

    track_count
):

    telemetry_lines = [

        f"Frame: {frame_id}",

        f"Inference: {inference_ms:.2f} ms",

        f"Masks: {masks_count}",

        f"OBBs: {obb_count}",

        f"Tracks: {track_count}"
    ]

    y = 40

    for line in telemetry_lines:

        cv2.putText(

            frame,

            line,

            (20, y),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 255, 0),

            2
        )

        y += 45

    return frame


# ============================================================
# Main Runtime
# ============================================================

def main():

    print(
        "\n=== Semantic Spatial Runtime ===\n"
    )

    # --------------------------------------------------------
    # MCAP replay source
    # --------------------------------------------------------

    BAG_PATH = (

        "datasets/rosbags/rosbags/"
        "metric_depth_val_1779181947"
    )

    # --------------------------------------------------------
    # Replay loader
    # --------------------------------------------------------

    loader = MCAPReplayLoader(
        BAG_PATH
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    segmentation = SegmentationModule(
        model_path="yolov8n-seg.pt"
    )

    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker = MultiObjectTracker(

        max_missed_frames=15,

        association_cost_threshold=150.0
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
    # Depth intrinsics
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
    # Canonical projector
    # --------------------------------------------------------

    projector = DepthToRGBProjector(

        depth_intrinsics=depth_intrinsics,

        rgb_intrinsics=rgb_intrinsics
    )

    # --------------------------------------------------------
    # Point cloud subsystem
    # --------------------------------------------------------

    pointcloud_generator = (
        PointCloudGenerator(
            depth_intrinsics
        )
    )

    object_pointcloud_extractor = (
    ObjectPointCloudExtractor(

        pointcloud_generator,

        projector
    )
)

    # --------------------------------------------------------
    # Export setup
    # --------------------------------------------------------

    ENABLE_RECORDING = False

    export_dir = (
        "replay_validation/exports"
    )

    os.makedirs(

        export_dir,

        exist_ok=True
    )

    writer = None

    # ========================================================
    # Runtime Loop
    # ========================================================

    while loader.has_next():

        packet = loader.get_next_packet()

        if packet is None:
            break

        rgb_frame = packet.rgb_frame

        depth_frame = packet.depth_frame

        overlay = rgb_frame.copy()

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        result = segmentation.segment(
            rgb_frame
        )

        # ----------------------------------------------------
        # Build detections
        # ----------------------------------------------------

        detections = []

        for idx, obb in enumerate(
            result["obbs"]
        ):

            mask = None

            if idx < len(result["masks"]):

                mask = result[
                    "masks"
                ][idx]

            detection = (
                build_detection_from_obb(
                    obb,
                    mask
                )
            )

            detections.append(
                detection
            )

        # ----------------------------------------------------
        # Tracker update
        # ----------------------------------------------------

        active_tracks = tracker.update(
            detections
        )

        # ----------------------------------------------------
        # Per-track geometry
        # ----------------------------------------------------

        for track in active_tracks:

            geometry_stats = None

            pointcloud = None

            # ------------------------------------------------
            # Mask exists
            # ------------------------------------------------

            if track.last_valid_mask is not None:

                pointcloud = (

                    object_pointcloud_extractor
                    .extract_object_pointcloud(

                        depth_frame=depth_frame,

                        segmentation_mask=(
                            track.last_valid_mask
                        ),

                        stride=12
                    )
                )

                geometry_stats = (

                    object_pointcloud_extractor
                    .compute_geometry_stats(
                        pointcloud
                    )
                )

            # ------------------------------------------------
            # Projected cloud rendering
            # ------------------------------------------------

            overlay = draw_projected_pointcloud(

                overlay,

                pointcloud,

                projector
            )

            # ------------------------------------------------
            # Track rendering
            # ------------------------------------------------

            overlay = draw_track_overlay(

                overlay,

                track,

                geometry_stats
            )

        # ----------------------------------------------------
        # Runtime telemetry
        # ----------------------------------------------------

        overlay = draw_runtime_telemetry(

            frame=overlay,

            frame_id=packet.frame_id,

            inference_ms=result[
                "elapsed_ms"
            ],

            masks_count=len(
                result["masks"]
            ),

            obb_count=len(
                result["obbs"]
            ),

            track_count=len(
                active_tracks
            )
        )

        # ----------------------------------------------------
        # Export
        # ----------------------------------------------------

        if writer is not None:

            writer.write(
                overlay
            )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        cv2.imshow(
            "Semantic Spatial Runtime",
            overlay
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    # ========================================================
    # Cleanup
    # ========================================================

    if writer is not None:

        writer.release()

    loader.release()

    cv2.destroyAllWindows()

    print(
        "\nReplay validation complete.\n"
    )


if __name__ == "__main__":
    main()