# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Geodesic reference-path utilities for navigation metrics."""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch


# Diagonal step cost on a unit grid.
_SQRT2 = math.sqrt(2.0)
# Octile-heuristic constant: cost of one diagonal beyond the orthogonal component.
_OCTILE_DIAG_EXTRA = _SQRT2 - 1.0

# 8-connected neighbor offsets (di, dj, step_cost_in_cells).
_NEIGHBOR_OFFSETS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
)


def _path_turn_cost(path: list[tuple[int, int]]) -> float:
    """Return cumulative absolute heading change along a grid path in radians."""
    if len(path) < 3:
        return 0.0

    headings = []
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        headings.append(math.atan2(j1 - j0, i1 - i0))

    turn_cost = 0.0
    for h0, h1 in zip(headings[:-1], headings[1:]):
        turn_cost += abs(math.atan2(math.sin(h1 - h0), math.cos(h1 - h0)))
    return turn_cost


def astar_geodesic_and_turns_2d(
    valid_mask: np.ndarray,
    src_i: int,
    src_j: int,
    dst_i: int,
    dst_j: int,
    cell_size: float,
) -> tuple[float, float]:
    """Geodesic distance and heading-change cost on a 2D occupancy grid via A*.

    Uses 8-connectivity with no corner cutting (a diagonal move is only legal
    if both cardinal neighbors it touches are also navigable) and an octile
    heuristic. Returns (distance_m, turn_cost_rad), or (math.inf, math.inf) if
    unreachable.
    """
    height, width = valid_mask.shape
    if not (0 <= src_i < height and 0 <= src_j < width and 0 <= dst_i < height and 0 <= dst_j < width):
        return math.inf, math.inf
    if not valid_mask[src_i, src_j] or not valid_mask[dst_i, dst_j]:
        return math.inf, math.inf
    if src_i == dst_i and src_j == dst_j:
        return 0.0, 0.0

    def heuristic(i: int, j: int) -> float:
        di = abs(i - dst_i)
        dj = abs(j - dst_j)
        return max(di, dj) + _OCTILE_DIAG_EXTRA * min(di, dj)

    # g[node] holds the best-known cost in cell units.
    g = {(src_i, src_j): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap = [(heuristic(src_i, src_j), 0.0, src_i, src_j)]

    while open_heap:
        _, current_g, i, j = heapq.heappop(open_heap)
        if i == dst_i and j == dst_j:
            node = (i, j)
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            path.reverse()
            return current_g * cell_size, _path_turn_cost(path)
        if current_g > g.get((i, j), math.inf):
            continue
        for di, dj, step in _NEIGHBOR_OFFSETS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < height and 0 <= nj < width):
                continue
            if not valid_mask[ni, nj]:
                continue
            # No corner cutting: diagonal requires both cardinal neighbors free.
            if di != 0 and dj != 0:
                if not valid_mask[i + di, j] or not valid_mask[i, j + dj]:
                    continue
            tentative_g = current_g + step
            key = (ni, nj)
            if tentative_g < g.get(key, math.inf):
                g[key] = tentative_g
                parent[key] = (i, j)
                heapq.heappush(open_heap, (tentative_g + heuristic(ni, nj), tentative_g, ni, nj))

    return math.inf, math.inf


class GeodesicReferenceComputer:
    """Computes spawn-to-goal reference distances on cached terrain masks."""

    def __init__(
        self,
        valid_mask: np.ndarray,
        cell_size: float,
        border_pixels: int,
        mesh_center: float,
    ):
        self.valid_mask = valid_mask
        self.cell_size = cell_size
        self.border_pixels = border_pixels
        self.mesh_center = mesh_center

    def compute(
        self,
        env_ids: torch.Tensor,
        terrain_indices: torch.Tensor,
        start_x: torch.Tensor,
        start_y: torch.Tensor,
        goal_x: torch.Tensor,
        goal_y: torch.Tensor,
        initial_distance_to_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return geodesic distance and turn-cost tensors for the given envs."""
        start_i, start_j = self._local_to_cell_indices(start_x, start_y)
        goal_i, goal_j = self._local_to_cell_indices(goal_x, goal_y)
        terrain_idx_np = terrain_indices.cpu().numpy()
        env_ids_np = env_ids.cpu().numpy()
        height, width = self.valid_mask.shape[1], self.valid_mask.shape[2]

        geodesic = np.empty(len(env_ids_np), dtype=np.float32)
        turn_cost = np.empty(len(env_ids_np), dtype=np.float32)
        for k in range(len(env_ids_np)):
            terrain_idx = int(terrain_idx_np[k])
            si = int(np.clip(start_i[k], 0, height - 1))
            sj = int(np.clip(start_j[k], 0, width - 1))
            gi = int(np.clip(goal_i[k], 0, height - 1))
            gj = int(np.clip(goal_j[k], 0, width - 1))
            distance, theta = astar_geodesic_and_turns_2d(
                self.valid_mask[terrain_idx], si, sj, gi, gj, self.cell_size
            )
            if not math.isfinite(distance):
                # Unreachable on the grid: fall back to Euclidean so metrics
                # remain defined, if optimistic.
                distance = float(initial_distance_to_goal[int(env_ids_np[k])].item())
                theta = 0.0
            geodesic[k] = distance
            turn_cost[k] = theta

        return (
            torch.from_numpy(geodesic).to(device=initial_distance_to_goal.device, dtype=initial_distance_to_goal.dtype),
            torch.from_numpy(turn_cost).to(device=initial_distance_to_goal.device, dtype=initial_distance_to_goal.dtype),
        )

    def _local_to_cell_indices(
        self, local_x: torch.Tensor, local_y: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert PositionSampler's cell-to-local mapping to recover cell indices."""
        i = ((local_x + self.mesh_center) / self.cell_size - self.border_pixels).round().long().cpu().numpy()
        j = ((local_y + self.mesh_center) / self.cell_size - self.border_pixels).round().long().cpu().numpy()
        return i, j
