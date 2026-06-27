from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from .global_costmap import GlobalCostmap, _grid_line


@dataclass(frozen=True)
class GlobalRoute:
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    status: str = "unavailable"
    reason: str = ""
    length_m: float = 0.0
    expanded_cells: int = 0


class AStarRoutePlanner:
    def __init__(self, *, inflation_radius_m: float = 0.55, unknown_cost: float = 0.04):
        self.inflation_radius_m = float(inflation_radius_m)
        self.unknown_cost = float(unknown_cost)

    def plan(
        self,
        costmap: GlobalCostmap,
        *,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> GlobalRoute:
        start = costmap.world_to_grid(*start_xy)
        goal = costmap.world_to_grid(*goal_xy)
        if start is None or goal is None:
            return GlobalRoute(status="out_of_bounds", reason="start or goal is outside global costmap")

        requested_start = start
        requested_goal = goal
        occupied = costmap.inflated_occupancy(self.inflation_radius_m)
        start = _nearest_free(occupied, start)
        goal = _nearest_free(occupied, goal)
        if start is None or goal is None:
            return GlobalRoute(status="blocked", reason="no free start or goal cell")

        frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        best_cost = {start: 0.0}
        expanded = 0
        while frontier:
            _, cost, current = heapq.heappop(frontier)
            if cost > best_cost.get(current, float("inf")):
                continue
            expanded += 1
            if current == goal:
                cells = _reconstruct(came_from, current)
                cells = _simplify(cells, occupied)
                waypoints = [costmap.grid_to_world(row, col) for row, col in cells]
                if start == requested_start:
                    waypoints[0] = (float(start_xy[0]), float(start_xy[1]))
                if goal == requested_goal:
                    waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))
                return GlobalRoute(
                    waypoints=waypoints,
                    status="route_ready",
                    reason=f"A* route with {len(waypoints)} waypoints",
                    length_m=_path_length(waypoints),
                    expanded_cells=expanded,
                )
            for neighbor, step_cost in _neighbors(current, occupied):
                unknown = abs(float(costmap.evidence[neighbor])) < 0.05
                candidate = cost + step_cost * costmap.resolution_m + (self.unknown_cost if unknown else 0.0)
                if candidate >= best_cost.get(neighbor, float("inf")):
                    continue
                best_cost[neighbor] = candidate
                came_from[neighbor] = current
                heuristic = math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1]) * costmap.resolution_m
                heapq.heappush(frontier, (candidate + heuristic, candidate, neighbor))

        return GlobalRoute(status="no_route", reason="A* could not connect start to goal", expanded_cells=expanded)


def select_lookahead_waypoint(
    route: GlobalRoute,
    current_xy: tuple[float, float],
    lookahead_m: float = 1.2,
) -> tuple[float, float] | None:
    if route.status != "route_ready" or not route.waypoints:
        return None
    nearest = min(
        range(len(route.waypoints)),
        key=lambda idx: math.hypot(route.waypoints[idx][0] - current_xy[0], route.waypoints[idx][1] - current_xy[1]),
    )
    travelled = math.hypot(
        route.waypoints[nearest][0] - current_xy[0],
        route.waypoints[nearest][1] - current_xy[1],
    )
    for idx in range(nearest + 1, len(route.waypoints)):
        previous = route.waypoints[idx - 1]
        waypoint = route.waypoints[idx]
        travelled += math.hypot(waypoint[0] - previous[0], waypoint[1] - previous[1])
        if travelled >= lookahead_m:
            return waypoint
    return route.waypoints[-1]


def remaining_route_length(route: GlobalRoute, current_xy: tuple[float, float]) -> float:
    if route.status != "route_ready" or not route.waypoints:
        return float("inf")
    nearest = min(
        range(len(route.waypoints)),
        key=lambda idx: math.hypot(route.waypoints[idx][0] - current_xy[0], route.waypoints[idx][1] - current_xy[1]),
    )
    points = [current_xy, *route.waypoints[nearest:]]
    return _path_length(points)


def _neighbors(cell: tuple[int, int], occupied: np.ndarray):
    row, col = cell
    height, width = occupied.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= height or nc < 0 or nc >= width or occupied[nr, nc]:
            continue
        if dr != 0 and dc != 0 and (occupied[row + dr, col] or occupied[row, col + dc]):
            continue
        yield (nr, nc), math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0


def _nearest_free(occupied: np.ndarray, cell: tuple[int, int], max_radius: int = 12) -> tuple[int, int] | None:
    if not occupied[cell]:
        return cell
    row, col = cell
    height, width = occupied.shape
    for radius in range(1, max_radius + 1):
        candidates: list[tuple[float, tuple[int, int]]] = []
        for rr in range(max(0, row - radius), min(height, row + radius + 1)):
            for cc in range(max(0, col - radius), min(width, col + radius + 1)):
                if max(abs(rr - row), abs(cc - col)) != radius or occupied[rr, cc]:
                    continue
                candidates.append((math.hypot(rr - row, cc - col), (rr, cc)))
        if candidates:
            return min(candidates)[1]
    return None


def _reconstruct(came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _simplify(path: list[tuple[int, int]], occupied: np.ndarray) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1:
            if all(not occupied[cell] for cell in _grid_line(path[anchor], path[candidate])):
                break
            candidate -= 1
        simplified.append(path[candidate])
        anchor = candidate
    return simplified


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
