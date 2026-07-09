# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Global path generation: A* search over a :class:`Prm` roadmap, line-of-sight smoothing, and
path-category sampling (optimal / mildly non-optimal / highly non-optimal / infeasible).

:class:`MultiPath` owns one :class:`GlobalPath` per environment and is rebuilt whenever an
environment's goal/spawn is resampled - see
:class:`isaaclab_nav_task.navigation.mdp.navigation.goal_commands.RobotNavigationGoalCommand`.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import torch

from isaaclab.utils.math import transform_points

from .prm import MultiPRM, Prm


class GlobalPathConfig:
    """Configuration for global path generation."""

    def __init__(
        self,
        n_waypoints: int = 1,
        num_smooth_points: int = 15,
        add_noise: float = 10,  # noise level in height-map units, applied to infeasible paths
        z_offset: float = 0.5,  # z offset added to all path points (in world units)
        path_categories: list = ["optimal", "mildly_non_optimal", "highly_non_optimal", "infeasible"],
        category_weights: list = [0, 0, 0.5, 0.5],
        visualise: bool = False,
    ):
        self.n_waypoints = n_waypoints
        self.num_smooth_points = num_smooth_points
        self.add_noise = add_noise
        self.z_offset = z_offset
        self.path_categories = path_categories
        if not np.isclose(sum(category_weights), 1.0):
            raise ValueError(f"category_weights must sum to 1.0, got {sum(category_weights)}")
        self.category_weights = category_weights
        self.visualise = visualise

    def copy(self) -> "GlobalPathConfig":
        return GlobalPathConfig(
            n_waypoints=self.n_waypoints,
            num_smooth_points=self.num_smooth_points,
            add_noise=self.add_noise,
            z_offset=self.z_offset,
            path_categories=self.path_categories,
            category_weights=self.category_weights,
            visualise=self.visualise,
        )


class MultiPath:
    """Manager for one :class:`GlobalPath` per environment."""

    def __init__(self, num_envs: int, cfg_path: GlobalPathConfig, prm_manager: MultiPRM, device: torch.device):
        self.path_class = GlobalPath
        self.prm_manager = prm_manager
        self.num_envs = num_envs
        self.device = device
        self.size_path = cfg_path.num_smooth_points

        self.paths: list[GlobalPath] = []
        self.paths_tensor: torch.Tensor = torch.zeros((num_envs, self.size_path, 4), dtype=torch.float32, device=self.device)
        self.paths_w: torch.Tensor = torch.zeros((num_envs, self.size_path, 3), dtype=torch.float32, device=self.device)
        self.paths_unscaled: torch.Tensor = torch.zeros((num_envs, self.size_path, 3), dtype=torch.float32, device=self.device)
        self.last_best_progress: torch.Tensor = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
        self.closest_point: torch.Tensor = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.final_lookahead: torch.Tensor = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.device)
        self.final_lookahead_normalised: torch.Tensor = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)

        self.build_all(cfg_path.copy())

    def build_all(self, cfg_path: GlobalPathConfig, tile_ids: Optional[torch.Tensor] = None, starts: Optional[torch.Tensor] = None, goals: Optional[torch.Tensor] = None):
        """Build (or rebuild) paths for all environments.

        Args:
            cfg_path: path configuration.
            tile_ids: terrain-tile index per env (shape `(num_envs,)`), or `None` to build without a PRM yet.
            starts: start position per env, `(num_envs, 2)` in height-map coordinates.
            goals: goal position per env, `(num_envs, 2)` in height-map coordinates.
        """
        if cfg_path.num_smooth_points != self.size_path:
            raise RuntimeError(
                f"cfg_path.num_smooth_points ({cfg_path.num_smooth_points}) must match "
                f"MultiPath.size_path ({self.size_path})."
            )
        for env_id in range(self.num_envs):
            prm = self.prm_manager.get(int(tile_ids[env_id])) if tile_ids is not None else None
            start = tuple(starts[env_id].tolist()) if starts is not None else None
            goal = tuple(goals[env_id].tolist()) if goals is not None else None
            self.paths.append(self.path_class(cfg_path.copy(), env_id, prm, start, goal))
            self.paths_tensor[env_id, : cfg_path.num_smooth_points] = self.paths[env_id].path_b
            self.paths_w[env_id, : cfg_path.num_smooth_points] = self.paths[env_id].path_w
            self.paths_unscaled[env_id, : cfg_path.num_smooth_points] = self.paths[env_id].path_b_unscaled
            self.last_best_progress[env_id] = 0.0
        print(f"[MultiPath] Built paths for {self.num_envs} environments.")

    def rebuild(self, cfg_path: GlobalPathConfig, starts: torch.Tensor, goals: torch.Tensor, tile_ids: torch.Tensor, env_ids: Sequence[int]):
        """Rebuild paths for the given environments.

        Args:
            cfg_path: path configuration.
            starts: start position per selected env, `(k, 2)`, world-frame (x, y) - rescaled to
                height-map coordinates internally via `Prm.rescale_points_to_heightmap`.
            goals: goal position per selected env, `(k, 2)`, world-frame (x, y).
            tile_ids: terrain-tile index per selected env, `(k,)`.
            env_ids: environment indices to rebuild paths for.
        """
        if starts.shape[1] != 2 or goals.shape[1] != 2:
            raise RuntimeError("starts and goals must be of shape (k, 2) representing world-frame (x, y).")

        for i, env_id in enumerate(env_ids):
            prm = self.prm_manager.get(int(tile_ids[i]))
            self.paths[env_id].prm = prm
            self.paths[env_id].modify_cfg(**cfg_path.__dict__)

            start_goal = torch.stack((starts[i], goals[i]), dim=0)
            start_goal = prm.rescale_points_to_heightmap(start_goal).detach().cpu().numpy()
            prm.reset_start_goal(start_goal[0], start_goal[1])

            self.paths[env_id].find_path()
            self.paths_tensor[env_id, : cfg_path.num_smooth_points] = self.paths[env_id].path_b
            self.paths_w[env_id, : cfg_path.num_smooth_points] = self.paths[env_id].path_w
            self.paths_unscaled[env_id, : cfg_path.num_smooth_points] = self.paths[env_id].path_b_unscaled
            self.last_best_progress[env_id] = 0.0

    def get(self, env_id: int) -> "GlobalPath":
        """Access a specific `GlobalPath`."""
        return self.paths[env_id]

    def get_all_tensors(self) -> torch.Tensor:
        """All paths in body frame, `(num_envs, num_smooth_points, 4)` = `(direction_xyz, log_distance)`."""
        return self.paths_tensor

    def update(self, inv_pos: torch.Tensor, inv_rot: torch.Tensor, robot_pos_w: torch.Tensor, robot_quat_w: torch.Tensor) -> torch.Tensor:
        """Vectorized per-step update: transform all paths into the current body frame.

        Args:
            inv_pos: inverse robot position, `(num_envs, 3)`.
            inv_rot: inverse robot rotation, `(num_envs, 4)` quaternion.
            robot_pos_w: robot position in world frame, `(num_envs, 3)` (for marker visualization only).
            robot_quat_w: robot orientation in world frame, `(num_envs, 4)` (for marker visualization only).

        Returns:
            Mean distance from the robot to its path, across all environments (scalar, for logging).
        """
        path_b_unscaled = transform_points(self.paths_w, inv_pos, inv_rot)  # (B, N, 3)
        self.paths_unscaled.copy_(path_b_unscaled)

        distances = torch.norm(path_b_unscaled, dim=-1, keepdim=True) + 1e-9
        path_dirs = path_b_unscaled / distances
        path_logdist = torch.log(distances + 1.0)
        max_dist = torch.clamp(path_logdist.max(dim=1, keepdim=True)[0], min=1e-9)
        norm_dist = path_logdist / max_dist

        self.paths_tensor[:, :, :3].copy_(path_dirs)
        self.paths_tensor[:, :, 3:].copy_(norm_dist)

        self.compute_lookahead_and_closest()
        distance_look_ahead = torch.norm(self.final_lookahead[:, :3], dim=-1, keepdim=True) + 1e-6
        self.final_lookahead_normalised[:, :3] = self.final_lookahead[:, :3] / distance_look_ahead
        self.final_lookahead_normalised[:, 3:] = torch.log(distance_look_ahead + 1.0)

        distance_to_path = torch.norm(self.closest_point, dim=-1, keepdim=True)
        mean_distance_to_path = torch.mean(distance_to_path)
        return mean_distance_to_path

    def compute_lookahead_and_closest(self, lookahead_dist: float = 3.0):
        """Compute, per env, the closest point on the path and a fixed-distance lookahead point ahead of it.

        Populates `self.closest_point` and `self.final_lookahead`, both `(num_envs, 3)` in body frame.
        """
        B, N, D = self.paths_unscaled.shape
        if D == 2:
            raise RuntimeError("2D paths not supported yet.")

        last_point = self.paths_unscaled[:, -1, :]

        a = self.paths_unscaled[:, :-1, :]
        b = self.paths_unscaled[:, 1:, :]
        seg_vecs = b - a
        seg_lens = torch.norm(seg_vecs, dim=-1) + 1e-9

        t = (-torch.sum(a * seg_vecs, dim=-1)) / (seg_lens**2)
        t_clamped = torch.clamp(t, 0.0, 1.0)

        proj = a + t_clamped.unsqueeze(-1) * seg_vecs
        dist = torch.norm(proj, dim=-1)

        best_dist, best_idx = torch.min(dist, dim=1)
        closest_point = proj[torch.arange(B), best_idx]

        curr_seg = best_idx.clone()
        curr_t = t_clamped.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        remaining = torch.full_like(curr_t, lookahead_dist)
        active = torch.ones_like(curr_t, dtype=torch.bool)
        last_seg_index = seg_vecs.shape[1] - 1

        while active.any():
            seg_len_curr = seg_lens.gather(1, curr_seg.unsqueeze(1)).squeeze(1)
            remain_t = 1.0 - curr_t
            remain_len = remain_t * seg_len_curr

            at_last_seg = curr_seg == last_seg_index
            must_stop = at_last_seg & (remaining > remain_len)
            if must_stop.any():
                active[must_stop] = False
                remaining[must_stop] = -1e9
                curr_t[must_stop] = 1.0

            reached = (remaining <= remain_len) & active
            curr_t[reached] = curr_t[reached] + remaining[reached] / seg_len_curr[reached]
            active[reached] = False

            unfinished = active & (~reached) & (~must_stop)
            remaining = remaining - remain_len
            curr_seg = curr_seg + unfinished * 1
            curr_t[unfinished] = 0.0
            curr_seg = torch.clamp(curr_seg, max=last_seg_index)

        index = curr_seg.unsqueeze(1).unsqueeze(1).expand(-1, -1, D)
        seg_vec_final = seg_vecs.gather(1, index).squeeze(1)
        a_final = a.gather(1, index).squeeze(1)
        final_lookahead = a_final + curr_t.unsqueeze(-1) * seg_vec_final

        beyond_end = curr_t == 1.0
        final_lookahead[beyond_end] = last_point[beyond_end]

        self.closest_point = closest_point
        self.final_lookahead = final_lookahead

    def get_all_path_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized distance-to-path and progress-along-path, across all environments.

        Returns:
            dists: distance from the robot to the closest point on its path, `(num_envs,)`.
            progress: positive progress made since the last call (0 if none, or on regression), `(num_envs,)`.
        """
        a = self.paths_unscaled[:, :-1, :]
        b = self.paths_unscaled[:, 1:, :]
        seg_vecs = b - a
        seg_lens = torch.norm(seg_vecs, dim=-1) + 1e-9

        t = (-torch.sum(a * seg_vecs, dim=-1)) / (seg_lens**2)
        t_clamped = torch.clamp(t, 0.0, 1.0)

        proj = a + t_clamped.unsqueeze(-1) * seg_vecs
        dist = torch.norm(proj, dim=-1)
        best_dist, best_idx = torch.min(dist, dim=1)

        cum_lens = torch.cumsum(seg_lens, dim=1)
        cum_lens_prev = torch.cat([torch.zeros((seg_lens.shape[0], 1), device=seg_lens.device), cum_lens[:, :-1]], dim=1)
        total_len = torch.sum(seg_lens, dim=1, keepdim=True)

        progress_along_path = (cum_lens_prev + t_clamped * seg_lens) / total_len
        best_progress = progress_along_path.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        progress_diff = torch.where(
            best_progress > self.last_best_progress,
            best_progress - self.last_best_progress,
            torch.zeros_like(best_progress),
        )
        best_progress = torch.where(best_progress < self.last_best_progress, torch.zeros_like(best_progress), best_progress)
        self.last_best_progress = torch.max(self.last_best_progress, best_progress)

        # only reward progress above a small threshold, to filter out sensor/geometry noise
        progress = torch.where(progress_diff > 0.15, progress_diff, torch.zeros_like(progress_diff))

        return best_dist, progress


class GlobalPath:
    """A single environment's global path: A* over the PRM roadmap + line-of-sight smoothing."""

    prm: Optional[Prm]
    cfg: GlobalPathConfig
    waypoints: list[int]  # indices into prm.nodes
    path: list[tuple]  # path nodes in height-map coordinates
    path_w: torch.Tensor  # path nodes in world coordinates
    path_b: torch.Tensor  # path nodes in body frame, with log-distance as 4th element
    path_b_unscaled: torch.Tensor  # path nodes in body frame, without direction/distance scaling

    def __init__(
        self,
        cfg: GlobalPathConfig,
        env_id: int,
        prm: Optional[Prm] = None,
        start: Optional[tuple[int, int]] = None,
        goal: Optional[tuple[int, int]] = None,
    ) -> None:
        self.cfg = cfg
        self.env_id = env_id
        self.path_w = torch.zeros((cfg.num_smooth_points, 3), dtype=torch.float32)
        self.path_b = torch.zeros((cfg.num_smooth_points, 4), dtype=torch.float32)
        self.path_b_unscaled = torch.zeros((cfg.num_smooth_points, 3), dtype=torch.float32)
        self.path = []
        self.waypoints = []
        self.prm = None

        if prm is not None:
            self.prm = prm
            self.sample_waypoints()
        if start is not None and goal is not None:
            start_3d = (start[0], start[1], int(prm.height_map[start[0], start[1]]))
            goal_3d = (goal[0], goal[1], int(prm.height_map[goal[0], goal[1]]))
            self.prm.reset_start_goal(np.array(start_3d), np.array(goal_3d))
            self.find_path()

    def modify_cfg(self, **kwargs):
        """Modify the `GlobalPathConfig`."""
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
            else:
                raise AttributeError(f"GlobalPathConfig has no attribute '{key}'")

    def sample_waypoints(self):
        """Sample `cfg.n_waypoints` random intermediate waypoints from the PRM's free nodes."""
        if self.cfg.n_waypoints == 0:
            return
        self.waypoints = random.sample(self.prm.free_nodes, self.cfg.n_waypoints)

    def find_path(self):
        """Find a new path from the PRM's current start/goal and store it in `self.path`/`self.path_w`.

        Samples a path category (optimal / mildly non-optimal / highly non-optimal / infeasible) per
        `cfg.path_categories`/`cfg.category_weights`, runs A* accordingly, smooths with line-of-sight,
        and (for infeasible paths) perturbs the result with noise.
        """
        if self.prm.cfg.start_idx is None or self.prm.cfg.goal_idx is None:
            raise RuntimeError("Start and goal indices must be set in the PRM before finding a path.")

        path = np.array([self.prm.nodes[self.prm.cfg.start_idx]])
        choice = random.choices(self.cfg.path_categories, weights=self.cfg.category_weights, k=1)[0]

        if choice == "optimal":
            optimal, los = True, True
        elif choice == "mildly_non_optimal":
            optimal, los = False, True
            if self.cfg.n_waypoints > 0:
                self.sample_waypoints()
        elif choice == "highly_non_optimal":
            optimal, los = False, False
            if self.cfg.n_waypoints > 0:
                self.sample_waypoints()
        elif choice == "infeasible":
            optimal, los = False, False
            if self.cfg.n_waypoints > 0:
                self.sample_waypoints()
        else:
            raise RuntimeError(f"Invalid path category choice: {choice}")

        path_segment = self.astar(self.prm.cfg.start_idx, self.prm.cfg.goal_idx, optimal=optimal)
        if path_segment.size != 0:
            if los:
                path = np.vstack((path, self.smooth_path_los(path_segment)[1:]))
            else:
                path = np.vstack((path, self.smooth_path_los_for_non_optimal(path_segment)[1:]))

        # A* failing to connect start->goal should not happen except in pathological cases; fall back
        # to a direct start->goal segment so the rest of the pipeline always has a valid path.
        if not np.array_equal(np.array(path[-1]), self.prm.nodes[self.prm.cfg.goal_idx]):
            print(f"[GlobalPath] WARNING: Path does not reach goal! Path length: {len(path)}")
            if len(path) == 1:
                path = np.vstack((path, self.prm.nodes[self.prm.cfg.goal_idx]))

        for i in range(len(path) - 1):
            if not self.prm.is_traversable(path[i], path[i + 1], check_padding=False):
                print(f"[GlobalPath] WARNING: Path segment from {i} to {i + 1} is not traversable! (path length: {len(path)})")

        path = self.resample_path(path, num_points=self.cfg.num_smooth_points)

        if not np.array_equal(path[-1], self.prm.nodes[self.prm.cfg.goal_idx]):
            raise RuntimeError(f"[GlobalPath] Path does not reach goal after resampling! Path length: {len(path)}")

        if choice == "infeasible":
            path = self.add_noise_to_path(path, noise_level=self.cfg.add_noise)

        path = [(x, y, z + self.cfg.z_offset / self.prm.v_scale) for (x, y, z) in path]
        self.path = path
        self.scale_path_to_world()
        if self.cfg.visualise:
            self.visualise(color=(1, 1, 1, 1))

    def euclidean(self, p1: np.ndarray, p2: np.ndarray, optimal: bool = True) -> float:
        """Distance heuristic between two points: true Euclidean distance if `optimal`, otherwise a
        blend biased towards the first sampled waypoint (to encourage non-optimal routes)."""
        assert len(p1) == len(p2) and len(p1) in (2, 3), "Points must both be 2D or 3D."
        true_distance = float(np.linalg.norm(p1 - p2))
        if optimal:
            return true_distance

        wp = self.prm.nodes[self.waypoints[0]]
        distance_waypoint = float(np.linalg.norm(p1 - wp))
        beta = 0.1
        return beta * true_distance + (1 - beta) * distance_waypoint

    def astar(self, start: int, goal: int, optimal: bool = True) -> np.ndarray:
        """A* search over the PRM roadmap, returning the path as an array of height-map coordinates."""
        num_nodes = self.prm.nodes.shape[0]

        if not hasattr(self, "_came_from") or self._came_from.shape[0] != num_nodes:
            self._came_from = np.empty(num_nodes, dtype=np.int32)
        came_from = self._came_from
        came_from[:] = -1  # -1 means "no parent"

        if not hasattr(self, "_open_mask") or self._open_mask.shape[0] != num_nodes:
            self._open_mask = np.zeros(num_nodes, dtype=bool)
        open_mask = self._open_mask
        open_mask[:] = False
        open_mask[start] = True

        if not hasattr(self, "_g_score"):
            self._g_score = np.full(num_nodes, np.inf, dtype=np.float32)
        g_score = self._g_score
        g_score[:] = np.inf
        g_score[start] = 0.0

        if not hasattr(self, "_f_score"):
            self._f_score = np.full(num_nodes, np.inf, dtype=np.float32)
        f_score = self._f_score
        f_score[:] = np.inf
        f_score[start] = self.euclidean(self.prm.nodes[start], self.prm.nodes[goal], optimal=optimal)

        while open_mask.any():
            candidates = np.where(open_mask, f_score, np.inf)
            current = np.argmin(candidates)

            if current == goal:
                path_nodes = [current]
                while came_from[current] != -1:
                    current = came_from[current]
                    path_nodes.append(current)
                path_nodes.reverse()
                return self.prm.nodes[path_nodes]

            open_mask[current] = False
            for neighbor in self.prm.roadmap[current]:
                tentative_g = g_score[current]
                if optimal:
                    tentative_g += self.euclidean(self.prm.nodes[current], self.prm.nodes[neighbor])

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.euclidean(self.prm.nodes[neighbor], self.prm.nodes[goal], optimal=optimal)
                    open_mask[neighbor] = True

        print("[GlobalPath] WARNING: A* failed to find a path.")
        return np.array([])

    def smooth_path_los(self, path: np.ndarray) -> np.ndarray:
        """Simplify a path by connecting waypoints directly whenever there's line-of-sight.

        From each waypoint, tries to jump as far ahead as possible (all the way to the end first,
        backing off until a traversable straight line is found).
        """
        if path.size == 0:
            return path
        assert path.ndim == 2 and path.shape[1] == 3, f"Expected path shape (N, 3), got {path.shape}"

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1  # try to jump straight to the end
            while j > i + 1:
                if self.prm.is_traversable(path[i], path[j], check_padding=True):
                    break
                j -= 1
            smoothed_path.append(path[j])
            i = j
        return np.array(smoothed_path)

    def smooth_path_los_for_non_optimal(self, path: np.ndarray) -> np.ndarray:
        """Line-of-sight smoothing variant that keeps the path non-optimal.

        Unlike `smooth_path_los` (which jumps as far as possible), this walks forward one waypoint
        at a time and stops advancing as soon as line-of-sight to the next point would break,
        preserving more of the original (longer, less direct) route.
        """
        if path.size == 0:
            return path
        assert path.ndim == 2 and path.shape[1] == 3, f"Expected path shape (N, 3), got {path.shape}"

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 2:
            j = i
            traversable = True
            while j < len(path) - 1:
                j += 1
                traversable = self.prm.is_traversable(path[i], path[j], check_padding=True)
                if not traversable:
                    break
            smoothed_path.append(path[j - 1])
            if (not traversable) and j - 1 == i:
                raise RuntimeError(f"Cannot progress from point {i} to {i + 1} in path smoothing!")
            i = j - 1
        smoothed_path.append(path[-1])
        return np.array(smoothed_path)

    def resample_path(self, path: np.ndarray, num_points: int) -> np.ndarray:
        """Resample a path to a fixed number of points, evenly spaced by arc length (straight-line interpolation)."""
        distances = np.cumsum(np.array([0] + [np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))]))
        total_dist = distances[-1]
        if total_dist == 0:
            first_pt = path[0]
            return np.repeat(first_pt[np.newaxis, :], num_points, axis=0)

        sample_distances = np.linspace(0, total_dist, num_points)
        resampled_path = []
        for d in sample_distances:
            idx = np.searchsorted(distances, d)
            if idx == 0:
                resampled_path.append(path[0])
            elif idx >= len(path):
                resampled_path.append(path[-1])
            else:
                t = (d - distances[idx - 1]) / (distances[idx] - distances[idx - 1])
                resampled_path.append((1 - t) * path[idx - 1] + t * path[idx])
        return np.array(resampled_path)

    def add_noise_to_path(self, path: np.ndarray, noise_level: float = 1) -> np.ndarray:
        """Add random 2D (x, y) noise to every waypoint, sampled uniformly within `noise_level` radius."""
        assert path.ndim == 2 and path.shape[1] == 3, f"Expected path shape (N, 3), got {path.shape}"

        n_points = path.shape[0]
        r = noise_level * np.sqrt(np.random.uniform(0, 1, size=n_points))
        theta = np.random.uniform(0, 2 * np.pi, size=n_points)

        noisy_path = path.copy()
        noisy_path[:, 0] += r * np.cos(theta)
        noisy_path[:, 1] += r * np.sin(theta)
        return noisy_path

    def scale_path_to_world(self) -> None:
        """Rescale `self.path` (height-map coordinates) to world coordinates, stored in `self.path_w`."""
        self.path_w = self.prm.rescale_points_to_world(torch.tensor(self.path, dtype=torch.float32))

    def visualise(self, color: tuple = (1, 1, 1, 1), print_wp: bool = False) -> None:
        from omni.isaac.debug_draw import _debug_draw

        rescaled_path = self.prm.rescale_points_to_world(torch.tensor(self.path, dtype=torch.float32)).tolist()
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.draw_lines(rescaled_path[:-1], rescaled_path[1:], [color] * len(rescaled_path[1:]), [2] * len(rescaled_path[1:]))

        if self.cfg.n_waypoints > 0 and print_wp:
            waypoint_coords = [self.prm.nodes[idx] for idx in self.waypoints]
            print("Waypoints (bin coords):", waypoint_coords)
