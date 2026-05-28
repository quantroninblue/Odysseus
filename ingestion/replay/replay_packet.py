from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ReplayFramePacket:

    # ------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------

    frame_id: int

    timestamp: float

    # ------------------------------------------------------------
    # Core sensor streams
    # ------------------------------------------------------------

    rgb_frame: np.ndarray

    depth_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    # Optional semantic streams
    # ------------------------------------------------------------

    segmentation_mask: Optional[np.ndarray] = None

    overlay_frame: Optional[np.ndarray] = None

    pointcloud: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    # Geometry outputs
    # ------------------------------------------------------------

    obb_result: Optional[dict] = None

    pose_result: Optional[dict] = None

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    rgb_path: Optional[str] = None

    depth_path: Optional[str] = None

    metadata: Optional[dict] = None