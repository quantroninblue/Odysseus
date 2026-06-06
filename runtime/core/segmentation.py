from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import SegmentationConfig


class SegmentationProvider(Protocol):
    def segment(self, rgb_frame, depth_frame=None) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DisabledSegmentationProvider:
    def segment(self, rgb_frame, depth_frame=None) -> dict:
        return {
            "overlay": rgb_frame,
            "masks": [],
            "obbs": [],
            "elapsed_ms": 0.0,
        }


class MockSegmentationProvider:
    def segment(self, rgb_frame, depth_frame=None) -> dict:
        mask = np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
        h, w = mask.shape
        y0, y1 = h // 4, max(h // 4 + 1, 3 * h // 4)
        x0, x1 = w // 4, max(w // 4 + 1, 3 * w // 4)
        mask[y0:y1, x0:x1] = 255
        return {
            "overlay": rgb_frame.copy(),
            "masks": [mask],
            "obbs": [],
            "elapsed_ms": 0.0,
        }




class GazeboDepthSegmentationProvider:
    """Segment nearby Gazebo obstacles directly from the depth image.

    This backend is intended for simulation validation where generic RGB object
    detectors may not recognize synthetic crates/racks, but the perception stack
    still needs real masks, RGB-D point extraction, semantic object fusion, and
    diagnostics over live sensor data.
    """

    def __init__(self, config: SegmentationConfig):
        self.config = config

    def segment(self, rgb_frame, depth_frame=None) -> dict:
        if depth_frame is None:
            return {"overlay": rgb_frame, "masks": [], "obbs": [], "elapsed_ms": 0.0}

        depth = np.asarray(depth_frame, dtype=np.float32)
        h, w = depth.shape[:2]
        finite = np.isfinite(depth)
        candidate = finite & (depth > 0.25) & (depth < 4.2)

        # Ignore the very top of the frame and the bottom floor-heavy strip.
        roi = np.zeros_like(candidate, dtype=bool)
        roi[int(h * 0.20):int(h * 0.78), int(w * 0.06):int(w * 0.94)] = True
        candidate &= roi

        # Prefer vertical obstacle surfaces over floor returns by rejecting the
        # nearest few rows at the bottom and tiny speckles.
        masks = _connected_components(candidate, max_components=self.config.max_masks_per_frame)
        boxes = []
        labels = []
        class_ids = []
        confidences = []
        filtered = []
        min_area = max(350, int(self.config.minimum_mask_area))
        for mask in masks:
            area = int(np.count_nonzero(mask))
            if area < min_area:
                continue
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            # Very wide bottom bands are usually floor, not objects.
            if (x1 - x0) > int(w * 0.75) and y1 > int(h * 0.68):
                continue
            filtered.append((mask.astype(np.uint8) * 255))
            boxes.append((x0, y0, x1, y1))
            labels.append("depth_obstacle")
            class_ids.append(1000)
            confidences.append(0.90)

        return {
            "overlay": rgb_frame.copy() if hasattr(rgb_frame, "copy") else rgb_frame,
            "masks": filtered,
            "boxes": boxes,
            "labels": labels,
            "class_ids": class_ids,
            "confidences": confidences,
            "obbs": [],
            "elapsed_ms": 0.0,
        }


def _connected_components(binary: np.ndarray, max_components: int) -> list[np.ndarray]:
    binary = np.asarray(binary, dtype=bool)
    visited = np.zeros(binary.shape, dtype=bool)
    h, w = binary.shape
    components: list[tuple[int, np.ndarray]] = []
    for y in range(0, h, 2):
        xs = np.flatnonzero(binary[y] & ~visited[y])
        for x_start in xs:
            if visited[y, x_start] or not binary[y, x_start]:
                continue
            stack = [(y, int(x_start))]
            coords = []
            visited[y, x_start] = True
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w or visited[ny, nx] or not binary[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(coords) == 0:
                continue
            mask = np.zeros(binary.shape, dtype=bool)
            ys, xs2 = zip(*coords)
            mask[np.asarray(ys), np.asarray(xs2)] = True
            components.append((len(coords), mask))
    components.sort(key=lambda item: item[0], reverse=True)
    return [mask for _, mask in components[:max_components]]


class YoloSegmentationProvider:
    def __init__(self, config: SegmentationConfig):
        from segmentation.segmentation_reference import SegmentationModule

        self.module = SegmentationModule(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            minimum_mask_area=config.minimum_mask_area,
        )

    def segment(self, rgb_frame) -> dict:
        return self.module.segment(rgb_frame)


def build_segmentation_provider(config: SegmentationConfig) -> SegmentationProvider:
    if not config.enabled:
        return DisabledSegmentationProvider()

    backend = config.backend.lower()
    if backend == "disabled":
        return DisabledSegmentationProvider()
    if backend == "mock":
        return MockSegmentationProvider()
    if backend in {"gazebo_depth", "depth"}:
        return GazeboDepthSegmentationProvider(config)
    if backend == "yolo":
        return YoloSegmentationProvider(config)
    raise ValueError(f"Unsupported segmentation backend: {config.backend}")
