"""
association.py

Advanced detection-to-track association logic.

Purpose:
- Match detections to persistent tracks
- Use motion-aware association
- Use IoU overlap
- Use velocity-aware prediction
- Use area consistency
- Use Hungarian assignment

This module DOES NOT:
- mutate tracks
- perform filtering
- manage lifecycle
- perform visualization
"""

import math
import numpy as np

from scipy.optimize import linear_sum_assignment


# ============================================================
# Distance metric
# ============================================================

def compute_center_distance(
    track,
    detection
):
    """
    Euclidean centroid distance.
    """

    dx = (
        track.center_x -
        detection["center_x"]
    )

    dy = (
        track.center_y -
        detection["center_y"]
    )

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# ============================================================
# Velocity-aware prediction cost
# ============================================================

def compute_velocity_cost(
    track,
    detection
):
    """
    Compare predicted motion direction.
    """

    predicted_x = (
        track.center_x +
        track.velocity_x
    )

    predicted_y = (
        track.center_y +
        track.velocity_y
    )

    dx = (
        predicted_x -
        detection["center_x"]
    )

    dy = (
        predicted_y -
        detection["center_y"]
    )

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# ============================================================
# Bounding-box area consistency
# ============================================================

def compute_area_cost(
    track,
    detection
):
    """
    Penalize unrealistic area changes.
    """

    track_area = (
        track.width_px *
        track.height_px
    )

    detection_area = (
        detection["width_px"] *
        detection["height_px"]
    )

    if track_area <= 0:
        return 1.0

    ratio = (
        detection_area /
        track_area
    )

    return abs(
        1.0 - ratio
    )


# ============================================================
# IoU computation
# ============================================================

def compute_iou(
    track,
    detection
):
    """
    Compute axis-aligned IoU.
    """

    tx1 = (
        track.center_x -
        track.width_px / 2
    )

    ty1 = (
        track.center_y -
        track.height_px / 2
    )

    tx2 = (
        track.center_x +
        track.width_px / 2
    )

    ty2 = (
        track.center_y +
        track.height_px / 2
    )

    dx1 = (
        detection["center_x"] -
        detection["width_px"] / 2
    )

    dy1 = (
        detection["center_y"] -
        detection["height_px"] / 2
    )

    dx2 = (
        detection["center_x"] +
        detection["width_px"] / 2
    )

    dy2 = (
        detection["center_y"] +
        detection["height_px"] / 2
    )

    inter_x1 = max(tx1, dx1)
    inter_y1 = max(ty1, dy1)

    inter_x2 = min(tx2, dx2)
    inter_y2 = min(ty2, dy2)

    inter_w = max(
        0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_w * inter_h
    )

    track_area = (
        (tx2 - tx1) *
        (ty2 - ty1)
    )

    detection_area = (
        (dx2 - dx1) *
        (dy2 - dy1)
    )

    union = (
        track_area +
        detection_area -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Hybrid association cost
# ============================================================

def compute_association_cost(
    track,
    detection
):
    """
    Hybrid weighted association score.
    Lower = better.
    """

    center_distance = (
        compute_center_distance(
            track,
            detection
        )
    )

    velocity_cost = (
        compute_velocity_cost(
            track,
            detection
        )
    )

    area_cost = (
        compute_area_cost(
            track,
            detection
        )
    )

    iou = compute_iou(
        track,
        detection
    )

    # --------------------------------------------------------
    # Weighted cost
    # --------------------------------------------------------

    total_cost = (

        0.40 * center_distance +

        0.30 * velocity_cost +

        0.20 * area_cost * 100 +

        0.10 * (1.0 - iou) * 100
    )

    return total_cost


# ============================================================
# Main association
# ============================================================

def associate_tracks_and_detections(

    tracks,
    detections,

    max_cost=150.0
):
    """
    Associate detections using Hungarian assignment.
    """

    matches = []

    unmatched_tracks = list(
        range(len(tracks))
    )

    unmatched_detections = list(
        range(len(detections))
    )

    # --------------------------------------------------------
    # Empty cases
    # --------------------------------------------------------

    if len(tracks) == 0:

        return (
            matches,
            unmatched_tracks,
            unmatched_detections
        )

    if len(detections) == 0:

        return (
            matches,
            unmatched_tracks,
            unmatched_detections
        )

    # --------------------------------------------------------
    # Cost matrix
    # --------------------------------------------------------

    cost_matrix = np.zeros(

        (
            len(tracks),
            len(detections)
        ),

        dtype=np.float32
    )

    for track_idx, track in enumerate(
        tracks
    ):

        for det_idx, detection in enumerate(
            detections
        ):

            cost = (
                compute_association_cost(
                    track,
                    detection
                )
            )

            cost_matrix[
                track_idx,
                det_idx
            ] = cost

    # --------------------------------------------------------
    # Hungarian assignment
    # --------------------------------------------------------

    track_indices, detection_indices = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    used_tracks = set()

    used_detections = set()

    # --------------------------------------------------------
    # Accept valid matches
    # --------------------------------------------------------

    for track_idx, det_idx in zip(

        track_indices,
        detection_indices
    ):

        cost = cost_matrix[
            track_idx,
            det_idx
        ]

        if cost > max_cost:
            continue

        matches.append(
            (
                track_idx,
                det_idx
            )
        )

        used_tracks.add(
            track_idx
        )

        used_detections.add(
            det_idx
        )

    # --------------------------------------------------------
    # Unmatched bookkeeping
    # --------------------------------------------------------

    unmatched_tracks = [

        idx for idx in range(len(tracks))

        if idx not in used_tracks
    ]

    unmatched_detections = [

        idx for idx in range(len(detections))

        if idx not in used_detections
    ]

    return (

        matches,

        unmatched_tracks,

        unmatched_detections
    )