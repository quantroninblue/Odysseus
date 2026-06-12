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
        valid_depth = finite & (depth > 0.25)

        mid_roi = np.zeros_like(valid_depth, dtype=bool)
        mid_roi[int(h * 0.20):int(h * 0.78), int(w * 0.06):int(w * 0.94)] = True
        low_roi = np.zeros_like(valid_depth, dtype=bool)
        low_roi[int(h * 0.36):int(h * 0.88), int(w * 0.04):int(w * 0.96)] = True

        # The low ROI keeps short bollards/poles visible to semantic mapping.
        # It is range-limited harder than the mid ROI to avoid swallowing floor.
        candidate = (
            (valid_depth & (depth < 4.2) & mid_roi)
            | (valid_depth & (depth < 3.6) & low_roi)
        )

        masks = _connected_components(
            candidate,
            max_components=max(self.config.max_masks_per_frame * 4, self.config.max_masks_per_frame),
        )
        boxes = []
        labels = []
        class_ids = []
        confidences = []
        filtered = []
        min_area = max(40, int(self.config.minimum_mask_area))
        close_thin_min_area = max(40, min_area)
        for mask in masks:
            area = int(np.count_nonzero(mask))
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox_w = x1 - x0 + 1
            bbox_h = y1 - y0 + 1
            component_depth = depth[mask]
            component_depth = component_depth[np.isfinite(component_depth)]
            min_depth = float(np.min(component_depth)) if component_depth.size else float("inf")
            is_low_component = y1 >= int(h * 0.50)
            is_close_thin = (
                area >= close_thin_min_area
                and min_depth <= 3.6
                and bbox_h >= max(8, int(h * 0.035))
                and bbox_w <= max(6, int(w * 0.18))
            )
            is_close_low = (
                area >= close_thin_min_area
                and min_depth <= 3.6
                and is_low_component
                and bbox_h >= max(6, int(h * 0.025))
                and bbox_w <= max(8, int(w * 0.40))
            )
            if area < min_area and not (is_close_thin or is_close_low):
                continue
            # Very wide bottom bands are usually floor, not objects.
            if (x1 - x0) > int(w * 0.75) and y1 > int(h * 0.68):
                continue
            filtered.append((mask.astype(np.uint8) * 255))
            boxes.append((x0, y0, x1, y1))
            labels.append("low_depth_obstacle" if is_low_component else "depth_obstacle")
            class_ids.append(1001 if is_low_component else 1000)
            confidences.append(0.92 if is_close_thin or is_close_low else 0.90)
            if len(filtered) >= self.config.max_masks_per_frame:
                break

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
