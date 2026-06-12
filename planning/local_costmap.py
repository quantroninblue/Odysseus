from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalCostmap:
    resolution_m: float
    x_max_m: float
    y_max_m: float
    inflation_radius_m: float
    grid: np.ndarray
    raw_points_xy: np.ndarray

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def world_to_grid(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        if x_m < 0.0 or x_m > self.x_max_m or abs(y_m) > self.y_max_m:
            return None
        col = int(x_m / self.resolution_m)
        row = int((y_m + self.y_max_m) / self.resolution_m)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        return row, col

    def is_occupied(self, x_m: float, y_m: float) -> bool:
        idx = self.world_to_grid(x_m, y_m)
        if idx is None:
            return False
        row, col = idx
        return bool(self.grid[row, col])

    def clearance_m(self, x_m: float, y_m: float, default: float = 8.0) -> float:
        if self.raw_points_xy.size == 0:
            return default
        delta = self.raw_points_xy - np.array([x_m, y_m], dtype=np.float32)
        distances = np.linalg.norm(delta, axis=1)
        return float(np.min(distances)) if distances.size else default


def build_local_costmap_from_depth(
    depth: np.ndarray,
    *,
    hfov_rad: float = 1.047,
    min_depth_m: float = 0.25,
    max_depth_m: float = 4.8,
    x_max_m: float = 5.0,
    y_max_m: float = 2.4,
    resolution_m: float = 0.08,
    inflation_radius_m: float = 0.46,
    row_min_frac: float = 0.18,
    row_max_frac: float = 0.72,
    col_min_frac: float = 0.05,
    col_max_frac: float = 0.95,
    stride: int = 3,
) -> LocalCostmap:
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape[:2]
    y0 = int(h * row_min_frac)
    y1 = max(y0 + 1, int(h * row_max_frac))
    x0 = int(w * col_min_frac)
    x1 = max(x0 + 1, int(w * col_max_frac))
    band = depth[y0:y1:stride, x0:x1:stride]

    rows, cols = np.indices(band.shape)
    image_cols = cols.reshape(-1).astype(np.float32) * float(stride) + float(x0)
    ranges = band.reshape(-1).astype(np.float32)
    valid = np.isfinite(ranges) & (ranges >= min_depth_m) & (ranges <= max_depth_m)
    if np.count_nonzero(valid) == 0:
        return _empty_costmap(resolution_m, x_max_m, y_max_m, inflation_radius_m)

    ranges = ranges[valid]
    image_cols = image_cols[valid]
    angles = (0.5 - image_cols / max(float(w - 1), 1.0)) * hfov_rad
    forward = ranges * np.cos(angles)
    lateral = ranges * np.sin(angles)
    useful = (
        (forward >= min_depth_m)
        & (forward <= x_max_m)
        & (np.abs(lateral) <= y_max_m)
    )
    if np.count_nonzero(useful) == 0:
        return _empty_costmap(resolution_m, x_max_m, y_max_m, inflation_radius_m)

    points = np.stack([forward[useful], lateral[useful]], axis=1).astype(np.float32)
    return _costmap_from_points(points, resolution_m, x_max_m, y_max_m, inflation_radius_m)


def _empty_costmap(
    resolution_m: float,
    x_max_m: float,
    y_max_m: float,
    inflation_radius_m: float,
) -> LocalCostmap:
    height = int(np.ceil(2.0 * y_max_m / resolution_m)) + 1
    width = int(np.ceil(x_max_m / resolution_m)) + 1
    return LocalCostmap(
        resolution_m=float(resolution_m),
        x_max_m=float(x_max_m),
        y_max_m=float(y_max_m),
        inflation_radius_m=float(inflation_radius_m),
        grid=np.zeros((height, width), dtype=bool),
        raw_points_xy=np.zeros((0, 2), dtype=np.float32),
    )


def _costmap_from_points(
    points_xy: np.ndarray,
    resolution_m: float,
    x_max_m: float,
    y_max_m: float,
    inflation_radius_m: float,
) -> LocalCostmap:
    costmap = _empty_costmap(resolution_m, x_max_m, y_max_m, inflation_radius_m)
    raw_grid = np.zeros_like(costmap.grid, dtype=bool)
    for x_m, y_m in points_xy:
        idx = costmap.world_to_grid(float(x_m), float(y_m))
        if idx is None:
            continue
        raw_grid[idx] = True

    occupied = np.argwhere(raw_grid)
    inflate_cells = int(np.ceil(inflation_radius_m / resolution_m))
    inflated = np.zeros_like(raw_grid, dtype=bool)
    for row, col in occupied:
        r0 = max(0, int(row) - inflate_cells)
        r1 = min(costmap.height, int(row) + inflate_cells + 1)
        c0 = max(0, int(col) - inflate_cells)
        c1 = min(costmap.width, int(col) + inflate_cells + 1)
        for rr in range(r0, r1):
            dy = (rr - int(row)) * resolution_m
            for cc in range(c0, c1):
                dx = (cc - int(col)) * resolution_m
                if dx * dx + dy * dy <= inflation_radius_m * inflation_radius_m:
                    inflated[rr, cc] = True

    costmap.grid = inflated
    costmap.raw_points_xy = points_xy.astype(np.float32, copy=False)
    return costmap
