
"""
pipeline.py
-----------
Subsystem-oriented monocular visual odometry frontend.

This module is intentionally designed as a reusable motion-estimation
subsystem inside a larger semantic-spatial runtime.

Core Responsibilities
---------------------
1. Feature detection + matching
2. Relative motion estimation
3. Monocular pose accumulation
4. Sparse triangulation
5. Keyframe management
6. Pose-state exposure to external runtimes

This module DOES NOT own:
- application runtime orchestration
- visualization ownership
- semantic mapping
- tracking pipelines
- replay pipelines
- world-state accumulation

Those are handled externally by the parent runtime.

Architecture
------------
RGB Frame
    ↓
VO.update()
    ↓
PoseUpdate
    ↓
External Runtime
    ↓
World-frame semantic accumulation
"""

from __future__ import annotations

import cv2
import numpy as np
import time

from dataclasses import dataclass, field
from enum import Enum, auto

from typing import (
    Callable,
    List,
    Optional,
)

from .camera import CameraModel

from .features import (
    DetectorType,
    MatcherType,
    FeatureDetector,
    FeatureMatcher,
    FrameFeatures,
)

from .motion import (
    MotionEstimator,
    PoseEstimate,
    compose_pose,
    invert_pose,
)

from .triangulation import (
    Triangulator,
    MapPoint,
)

from .keyframe import (
    Keyframe,
    KeyframeSelector,
)


# ═══════════════════════════════════════════════════════════════════════ #
#  Enums                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

class TrackingMode(Enum):

    DESCRIPTOR = auto()

    OPTICAL_FLOW = auto()


class VOState(Enum):

    NOT_INIT = auto()

    OK = auto()

    LOST = auto()


# ═══════════════════════════════════════════════════════════════════════ #
#  Configuration                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class VOConfig:

    # Feature detection
    detector_type: DetectorType = DetectorType.ORB

    max_features: int = 2000

    grid_rows: int = 4

    grid_cols: int = 4

    # Feature matching
    matcher_type: MatcherType = MatcherType.BF_HAMMING

    ratio_thresh: float = 0.75

    tracking_mode: TrackingMode = TrackingMode.DESCRIPTOR

    # Motion estimation
    ransac_thresh: float = 1.0

    ransac_prob: float = 0.999

    min_inliers: int = 20

    # Monocular scale
    scale_mode: str = "median_depth"

    fixed_scale: float = 1.0

    # Triangulation
    max_reproj_err: float = 2.0

    min_parallax_deg: float = 1.0

    min_depth: float = 0.1

    max_depth: float = 200.0

    # Keyframe insertion
    kf_min_parallax: float = 2.0

    kf_max_feat_ratio: float = 0.75

    kf_max_rot_deg: float = 15.0

    kf_min_frames: int = 3

    kf_max_frames: int = 20

    # Storage
    store_images: bool = False


# ═══════════════════════════════════════════════════════════════════════ #
#  Diagnostics                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class FrameStats:

    frame_id: int = 0

    num_detected: int = 0

    num_matched: int = 0

    num_inliers: int = 0

    num_map_pts: int = 0

    is_keyframe: bool = False

    kf_reason: str = ""

    h_score: float = 0.0

    process_ms: float = 0.0

    state: str = "OK"


# ═══════════════════════════════════════════════════════════════════════ #
#  Pose Update                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class PoseUpdate:

    success: bool = False

    frame_id: int = 0

    timestamp: float = 0.0

    T_world_cam: np.ndarray = field(
        default_factory=lambda: np.eye(4)
    )

    T_cam_world: np.ndarray = field(
        default_factory=lambda: np.eye(4)
    )

    translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )

    rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )

    num_inliers: int = 0

    tracking_state: str = "NOT_INIT"

    is_keyframe: bool = False

    process_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════ #
#  Pose Graph                                                            #
# ═══════════════════════════════════════════════════════════════════════ #

class PoseGraph:

    """
    Lightweight absolute pose accumulator.

    Stores:
        T_world_cam

    for each processed frame.

    Future extensions:
    - loop closure
    - graph optimization
    - backend correction
    """

    def __init__(self):

        self._poses: List[np.ndarray] = []

    def add(
        self,
        T_world_cam: np.ndarray
    ):

        self._poses.append(
            T_world_cam.copy()
        )

    def update(
        self,
        frame_id: int,
        T_world_cam: np.ndarray
    ):

        if frame_id < len(self._poses):

            self._poses[frame_id] = (
                T_world_cam.copy()
            )

    @property
    def poses(self) -> List[np.ndarray]:

        return self._poses

    @property
    def positions(self) -> np.ndarray:

        if not self._poses:

            return np.empty((0, 3))

        return np.array([
            T[:3, 3]
            for T in self._poses
        ])

    def __len__(self):

        return len(self._poses)


# ═══════════════════════════════════════════════════════════════════════ #
#  Visual Odometry Frontend                                              #
# ═══════════════════════════════════════════════════════════════════════ #

class VisualOdometry:

    """
    Reusable monocular VO frontend subsystem.

    This class exposes:
    - persistent camera pose
    - motion estimation
    - sparse map generation
    - keyframe management

    It intentionally does NOT own:
    - visualization
    - replay orchestration
    - semantic mapping
    - runtime scheduling
    """

    def __init__(
        self,
        camera: CameraModel,
        config: Optional[VOConfig] = None,
    ):

        self.camera = camera

        self.cfg = config or VOConfig()

        # ───────────────────────────────────────────────────────────── #
        #  Subsystems
        # ───────────────────────────────────────────────────────────── #

        self.detector = FeatureDetector(

            detector_type=self.cfg.detector_type,

            max_features=self.cfg.max_features,

            grid_rows=self.cfg.grid_rows,

            grid_cols=self.cfg.grid_cols,
        )

        self.matcher = FeatureMatcher(

            matcher_type=self.cfg.matcher_type,

            ratio_thresh=self.cfg.ratio_thresh,
        )

        self.estimator = MotionEstimator(

            camera=camera,

            ransac_thresh=self.cfg.ransac_thresh,

            ransac_prob=self.cfg.ransac_prob,

            min_inliers=self.cfg.min_inliers,
        )

        self.triangulator = Triangulator(

            camera=camera,

            max_reproj_err=self.cfg.max_reproj_err,

            min_depth=self.cfg.min_depth,

            max_depth=self.cfg.max_depth,

            min_parallax=self.cfg.min_parallax_deg,
        )

        self.kf_selector = KeyframeSelector(

            min_parallax_deg=self.cfg.kf_min_parallax,

            max_feature_ratio=self.cfg.kf_max_feat_ratio,

            max_rotation_deg=self.cfg.kf_max_rot_deg,

            min_frames=self.cfg.kf_min_frames,

            max_frames=self.cfg.kf_max_frames,
        )

        self.pose_graph = PoseGraph()

        # ───────────────────────────────────────────────────────────── #
        #  Runtime State
        # ───────────────────────────────────────────────────────────── #

        self.state: VOState = VOState.NOT_INIT

        self.frame_id: int = 0

        self.kf_id: int = 0

        self.keyframes: List[Keyframe] = []

        self.map_points: List[MapPoint] = []

        self.T_world_cam: np.ndarray = np.eye(4)

        self._last_kf: Optional[Keyframe] = None

        self._last_gray: Optional[np.ndarray] = None

        self._last_features: Optional[
            FrameFeatures
        ] = None

        # ───────────────────────────────────────────────────────────── #
        #  External Runtime Hooks
        # ───────────────────────────────────────────────────────────── #

        self.on_new_keyframe: Optional[
            Callable[[Keyframe], None]
        ] = None

        self.on_pose_update: Optional[
            Callable[[PoseUpdate], None]
        ] = None

        # ───────────────────────────────────────────────────────────── #
        #  Diagnostics
        # ───────────────────────────────────────────────────────────── #

        self.stats_history: List[
            FrameStats
        ] = []

    # ═══════════════════════════════════════════════════════════════ #
    #  Public API                                                     #
    # ═══════════════════════════════════════════════════════════════ #

    def update(
        self,
        img: np.ndarray,
        timestamp: float = 0.0,
    ) -> PoseUpdate:

        """
        Update VO state using a new frame.

        Returns
        -------
        PoseUpdate
            Current world-frame camera pose and
            motion-estimation metadata.
        """

        t0 = time.perf_counter()

        gray = self._to_gray(img)

        stats = FrameStats(
            frame_id=self.frame_id
        )

        if self.state == VOState.NOT_INIT:

            stats = self._initialize(

                gray=gray,

                img=img,

                timestamp=timestamp,

                stats=stats,
            )

        else:

            stats = self._track(

                gray=gray,

                img=img,

                timestamp=timestamp,

                stats=stats,
            )

        stats.process_ms = (
            time.perf_counter() - t0
        ) * 1000

        stats.state = self.state.name

        self.stats_history.append(stats)

        pose_update = PoseUpdate(

            success=(
                self.state == VOState.OK
            ),

            frame_id=self.frame_id,

            timestamp=timestamp,

            T_world_cam=self.T_world_cam.copy(),

            T_cam_world=invert_pose(
                self.T_world_cam
            ),

            translation=self.T_world_cam[
                :3, 3
            ].copy(),

            rotation=self.T_world_cam[
                :3, :3
            ].copy(),

            num_inliers=stats.num_inliers,

            tracking_state=self.state.name,

            is_keyframe=stats.is_keyframe,

            process_ms=stats.process_ms,
        )

        if self.on_pose_update:

            self.on_pose_update(
                pose_update
            )

        self.frame_id += 1

        return pose_update

    # ═══════════════════════════════════════════════════════════════ #
    #  Reset                                                          #
    # ═══════════════════════════════════════════════════════════════ #

    def reset(self):

        self.state = VOState.NOT_INIT

        self.frame_id = 0

        self.kf_id = 0

        self.keyframes = []

        self.map_points = []

        self.T_world_cam = np.eye(4)

        self._last_kf = None

        self._last_gray = None

        self._last_features = None

        self.pose_graph = PoseGraph()

        self.stats_history = []

        self.kf_selector.reset()

    # ═══════════════════════════════════════════════════════════════ #
    #  Properties                                                     #
    # ═══════════════════════════════════════════════════════════════ #

    @property
    def trajectory(self) -> np.ndarray:

        return self.pose_graph.positions

    @property
    def current_pose(self) -> np.ndarray:

        return self.T_world_cam.copy()

    # ═══════════════════════════════════════════════════════════════ #
    #  Initialization                                                 #
    # ═══════════════════════════════════════════════════════════════ #

    def _initialize(
        self,
        gray: np.ndarray,
        img: np.ndarray,
        timestamp: float,
        stats: FrameStats,
    ) -> FrameStats:

        feats = self.detector.detect_and_compute(
            gray
        )

        stats.num_detected = len(feats)

        if len(feats) < 10:

            return stats

        self.T_world_cam = np.eye(4)

        self.pose_graph.add(
            self.T_world_cam
        )

        kf = Keyframe(

            frame_id=self.frame_id,

            kf_id=self.kf_id,

            T_world_cam=self.T_world_cam.copy(),

            features=feats,

            timestamp=timestamp,

            image=(
                gray.copy()
                if self.cfg.store_images
                else None
            ),
        )

        self.keyframes.append(kf)

        self._last_kf = kf

        self._last_gray = gray.copy()

        self._last_features = feats

        self.state = VOState.OK

        self.kf_id += 1

        stats.is_keyframe = True

        stats.kf_reason = "init"

        return stats

    # ═══════════════════════════════════════════════════════════════ #
    #  Tracking                                                       #
    # ═══════════════════════════════════════════════════════════════ #

    def _track(
        self,
        gray: np.ndarray,
        img: np.ndarray,
        timestamp: float,
        stats: FrameStats,
    ) -> FrameStats:

        kf = self._last_kf

        cur_feats = (
            self.detector.detect_and_compute(
                gray
            )
        )

        stats.num_detected = len(cur_feats)

        match_result = self.matcher.match(

            kf.features,

            cur_feats
        )

        stats.num_matched = len(match_result)

        if len(match_result) < self.cfg.min_inliers:

            self.state = VOState.LOST

            return stats

        pose: PoseEstimate = (
            self.estimator.estimate(

                match_result.pts_ref,

                match_result.pts_cur,
            )
        )

        stats.num_inliers = pose.num_inliers

        stats.h_score = pose.H_score

        if not pose.success:

            self.state = VOState.LOST

            return stats

        # ───────────────────────────────────────────────────────── #
        #  Monocular Scale Recovery
        # ───────────────────────────────────────────────────────── #

        scale = self._recover_scale(
            pose
        )

        # ───────────────────────────────────────────────────────── #
        #  Relative Pose Composition
        # ───────────────────────────────────────────────────────── #

        R_cur_ref = pose.R.T

        t_cur_ref = -(
            pose.R.T @ (
                pose.t.ravel() * scale
            )
        )

        T_rel = np.eye(4)

        T_rel[:3, :3] = R_cur_ref

        T_rel[:3, 3] = t_cur_ref

        self.T_world_cam = compose_pose(

            self.T_world_cam,

            T_rel
        )

        self.pose_graph.add(
            self.T_world_cam
        )

        # ───────────────────────────────────────────────────────── #
        #  Triangulation
        # ───────────────────────────────────────────────────────── #

        inlier_mask = pose.inlier_mask

        inlier_ref = (
            match_result.pts_ref[
                inlier_mask
            ]
        )

        inlier_cur = (
            match_result.pts_cur[
                inlier_mask
            ]
        )

        inlier_idx_ref = (
            match_result.idx_ref[
                inlier_mask
            ]
        )

        inlier_idx_cur = (
            match_result.idx_cur[
                inlier_mask
            ]
        )

        T_kf_world = kf.T_cam_world

        T_cur_world = invert_pose(
            self.T_world_cam
        )

        new_mps, _ = (
            self.triangulator.triangulate(

                T_ref_world=T_kf_world,

                T_cur_world=T_cur_world,

                pts_ref=inlier_ref,

                pts_cur=inlier_cur,

                idx_ref=inlier_idx_ref,

                idx_cur=inlier_idx_cur,

                descriptors=(
                    kf.features.descriptors
                ),
            )
        )

        self.map_points.extend(
            new_mps
        )

        stats.num_map_pts = len(
            self.map_points
        )

        # ───────────────────────────────────────────────────────── #
        #  Keyframe Selection
        # ───────────────────────────────────────────────────────── #

        do_kf, kf_reason = (
            self.kf_selector.should_insert(

                last_kf=kf,

                R_rel=pose.R,

                pts_ref=inlier_ref,

                pts_cur=inlier_cur,

                num_tracked=pose.num_inliers,
            )
        )

        stats.is_keyframe = do_kf

        stats.kf_reason = kf_reason

        if do_kf:

            new_kf = Keyframe(

                frame_id=self.frame_id,

                kf_id=self.kf_id,

                T_world_cam=(
                    self.T_world_cam.copy()
                ),

                features=cur_feats,

                timestamp=timestamp,

                map_points=new_mps,

                image=(
                    gray.copy()
                    if self.cfg.store_images
                    else None
                ),
            )

            self.keyframes.append(
                new_kf
            )

            self._last_kf = new_kf

            self.kf_id += 1

            if self.on_new_keyframe:

                self.on_new_keyframe(
                    new_kf
                )

        self._last_gray = gray.copy()

        self._last_features = cur_feats

        self.state = VOState.OK

        return stats

    # ═══════════════════════════════════════════════════════════════ #
    #  Scale Recovery                                                #
    # ═══════════════════════════════════════════════════════════════ #

    def _recover_scale(
        self,
        pose: PoseEstimate,
    ) -> float:

        """
        Temporary monocular scale heuristic.

        This will eventually be replaced by:
        - RGBD metric grounding
        - semantic geometry constraints
        - sensor-fusion scale estimation
        """

        mode = self.cfg.scale_mode

        if mode == "fixed":

            return self.cfg.fixed_scale

        if mode == "none":

            return 1.0

        if (
            self.map_points
            and mode == "median_depth"
        ):

            T_cur_world = invert_pose(
                self.T_world_cam
            )

            depth = (
                self.triangulator.compute_median_depth(

                    self.map_points[
                        -min(
                            200,
                            len(self.map_points)
                        ):
                    ],

                    T_cur_world,
                )
            )

            if depth > 0:

                return depth

        return 1.0

    # ═══════════════════════════════════════════════════════════════ #
    #  Helpers                                                        #
    # ═══════════════════════════════════════════════════════════════ #

    @staticmethod
    def _to_gray(
        img: np.ndarray
    ) -> np.ndarray:

        if img.ndim == 3:

            return cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

        return img

    def summary(self) -> str:

        lines = [

            "=== Visual Odometry Summary ===",

            f"Frames processed : {self.frame_id}",

            f"Keyframes        : {len(self.keyframes)}",

            f"Map points       : {len(self.map_points)}",

            f"State            : {self.state.name}",
        ]

        if self.stats_history:

            proc_times = [

                s.process_ms
                for s in self.stats_history[1:]
            ]

            if proc_times:

                lines.append(

                    f"Avg process time : "
                    f"{np.mean(proc_times):.1f} ms "
                    f"({1000/np.mean(proc_times):.1f} fps)"
                )

        return "\n".join(lines)

