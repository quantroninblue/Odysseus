from __future__ import annotations

import math

import numpy as np


class GlobalCostmap:
    """Persistent world-frame occupancy evidence for global route planning."""

    def __init__(
        self,
        *,
        x_min_m: float,
        x_max_m: float,
        y_min_m: float,
        y_max_m: float,
        resolution_m: float = 0.20,
        obstacle_half_life_sec: float = 45.0,
        occupied_threshold: float = 1.2,
    ):
        if x_max_m <= x_min_m or y_max_m <= y_min_m or resolution_m <= 0.0:
            raise ValueError("global costmap bounds and resolution must be positive")
        self.x_min_m = float(x_min_m)
        self.x_max_m = float(x_max_m)
        self.y_min_m = float(y_min_m)
        self.y_max_m = float(y_max_m)
        self.resolution_m = float(resolution_m)
        self.obstacle_half_life_sec = float(obstacle_half_life_sec)
        self.occupied_threshold = float(occupied_threshold)
        self.width = int(math.ceil((self.x_max_m - self.x_min_m) / self.resolution_m)) + 1
        self.height = int(math.ceil((self.y_max_m - self.y_min_m) / self.resolution_m)) + 1
        self.evidence = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_update_sec: float | None = None

    def world_to_grid(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        col = int(math.floor((x_m - self.x_min_m) / self.resolution_m))
        row = int(math.floor((y_m - self.y_min_m) / self.resolution_m))
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.x_min_m + (float(col) + 0.5) * self.resolution_m,
            self.y_min_m + (float(row) + 0.5) * self.resolution_m,
        )

    def update_from_local_points(
        self,
        points_xy: np.ndarray,
        *,
        pose_x_m: float,
        pose_y_m: float,
        pose_yaw_rad: float,
        now_sec: float,
    ) -> None:
        """Fuse robot-frame obstacle endpoints and free rays into the world grid."""
        self._decay(now_sec)
        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if points.size == 0:
            return

        c = math.cos(pose_yaw_rad)
        s = math.sin(pose_yaw_rad)
        world_x = pose_x_m + c * points[:, 0] - s * points[:, 1]
        world_y = pose_y_m + s * points[:, 0] + c * points[:, 1]
        endpoints = {
            cell
            for cell in (self.world_to_grid(float(x), float(y)) for x, y in zip(world_x, world_y))
            if cell is not None
        }
        origin = self.world_to_grid(pose_x_m, pose_y_m)
        if origin is None or not endpoints:
            return

        free_cells: set[tuple[int, int]] = set()
        for endpoint in endpoints:
            ray = _grid_line(origin, endpoint)
            free_cells.update(ray[:-1])
        free_cells.difference_update(endpoints)

        if free_cells:
            rows, cols = zip(*free_cells)
            self.evidence[np.asarray(rows), np.asarray(cols)] -= 0.18
        rows, cols = zip(*endpoints)
        self.evidence[np.asarray(rows), np.asarray(cols)] += 0.85
        np.clip(self.evidence, -3.0, 5.0, out=self.evidence)

    def add_obstacles_world(self, points_xy: np.ndarray, evidence: float = 2.0) -> None:
        """Add world-frame obstacle evidence, primarily for mapped obstacle adapters."""
        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        cells = {
            cell
            for cell in (self.world_to_grid(float(x), float(y)) for x, y in points)
            if cell is not None
        }
        for row, col in cells:
            self.evidence[row, col] = min(5.0, self.evidence[row, col] + float(evidence))

    def inflated_occupancy(self, inflation_radius_m: float) -> np.ndarray:
        occupied = self.evidence >= self.occupied_threshold
        radius_cells = int(math.ceil(max(0.0, inflation_radius_m) / self.resolution_m))
        if radius_cells == 0 or not np.any(occupied):
            return occupied.copy()

        inflated = np.zeros_like(occupied)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                src_r0 = max(0, -dr)
                src_r1 = min(self.height, self.height - dr)
                src_c0 = max(0, -dc)
                src_c1 = min(self.width, self.width - dc)
                dst_r0 = src_r0 + dr
                dst_r1 = src_r1 + dr
                dst_c0 = src_c0 + dc
                dst_c1 = src_c1 + dc
                inflated[dst_r0:dst_r1, dst_c0:dst_c1] |= occupied[src_r0:src_r1, src_c0:src_c1]
        return inflated

    def _decay(self, now_sec: float) -> None:
        now_sec = float(now_sec)
        if self.last_update_sec is None:
            self.last_update_sec = now_sec
            return
        dt = max(0.0, now_sec - self.last_update_sec)
        self.last_update_sec = now_sec
        if dt <= 0.0 or self.obstacle_half_life_sec <= 0.0:
            return
        self.evidence *= math.exp(-math.log(2.0) * dt / self.obstacle_half_life_sec)


def _grid_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Integer Bresenham line including both endpoints."""
    r0, c0 = start
    r1, c1 = end
    dc = abs(c1 - c0)
    dr = -abs(r1 - r0)
    step_c = 1 if c0 < c1 else -1
    step_r = 1 if r0 < r1 else -1
    error = dc + dr
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((r0, c0))
        if r0 == r1 and c0 == c1:
            return cells
        twice_error = 2 * error
        if twice_error >= dr:
            error += dr
            c0 += step_c
        if twice_error <= dc:
            error += dc
            r0 += step_r
