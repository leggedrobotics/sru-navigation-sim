# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Probabilistic roadmap (PRM) for global path planning over maze terrains.

One :class:`Prm` roadmap is built per terrain tile from the tile's height field
(`env.scene.terrain._height_field_visual`, populated by the terrain "patches" system - see
:mod:`isaaclab_nav_task.terrains.patches`). :class:`MultiPRM` owns one roadmap per tile and is
built once per environment instance (see
:class:`isaaclab_nav_task.navigation.mdp.navigation.goal_commands.RobotNavigationGoalCommand`).
"""

from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
from scipy.spatial import distance_matrix
from skimage.draw import line

from isaaclab_nav_task.terrains.terrain_constants import THRESHOLDS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PrmConfig:
    """Configuration for a single terrain tile's probabilistic roadmap."""

    def __init__(
        self,
        N: int = 3000,
        k: int = 11,
        start_idx: Optional[int] = None,
        goal_idx: Optional[int] = None,
        max_height_diff: float = 50.0,
        fpr: float = 0.0,
        fnr: float = 0.0,
        padding: int = 4,
        visualise: bool = False,
    ):
        self.N = N
        self.k = k
        self.max_height_diff = max_height_diff
        self.start_idx = start_idx
        self.goal_idx = goal_idx
        self.fpr = fpr
        self.fnr = fnr
        self.padding = padding
        self.visualise = visualise

    def copy(self) -> "PrmConfig":
        return PrmConfig(
            N=self.N,
            k=self.k,
            start_idx=self.start_idx,
            goal_idx=self.goal_idx,
            max_height_diff=self.max_height_diff,
            fpr=self.fpr,
            fnr=self.fnr,
            padding=self.padding,
            visualise=self.visualise,
        )


class MultiPRM:
    """Manager for one PRM roadmap per terrain tile."""

    def __init__(self, prm_cfg: PrmConfig, env: "ManagerBasedRLEnv"):
        height_field = getattr(env.scene.terrain, "_height_field_visual", None)
        if height_field is None:
            raise RuntimeError(
                "env.scene.terrain._height_field_visual is not set. Ensure add_goal=True on the"
                " maze sub-terrain configs and that terrain generation has run before building the PRM."
            )

        self.prm_class = Prm
        self.num_terrains = height_field.shape[0]

        terrain_gen_cfg = env.cfg.scene.terrain.terrain_generator
        self._num_rows = terrain_gen_cfg.num_rows
        self._num_cols = terrain_gen_cfg.num_cols
        self._curriculum = terrain_gen_cfg.curriculum

        self.prms: list[Prm] = []
        self.build_all(prm_cfg.copy(), env)

    def _tile_origin(self, env: "ManagerBasedRLEnv", tile_id: int) -> torch.Tensor:
        """World-frame origin of a terrain tile, matching `goal_commands._get_terrain_indices`'s indexing."""
        if self._curriculum:
            level = tile_id % self._num_rows
            terrain_type = tile_id // self._num_rows
        else:
            level = tile_id // self._num_cols
            terrain_type = tile_id % self._num_cols
        return env.scene.terrain.terrain_origins[level, terrain_type]

    def build_all(self, prm_cfg: PrmConfig, env: "ManagerBasedRLEnv"):
        """Build a roadmap for every terrain tile."""
        for tile_id in range(self.num_terrains):
            origin = self._tile_origin(env, tile_id)
            prm = self.prm_class(prm_cfg.copy(), env, tile_id, origin)
            self.prms.append(prm)
        print(f"[MultiPRM] Built roadmaps for {self.num_terrains} terrain tiles.")

    def rebuild(self, tile_ids: torch.Tensor, prm_cfg: PrmConfig):
        """Rebuild the roadmap for a subset of terrain tiles (e.g. with a different `PrmConfig`)."""
        for tile_id in tile_ids:
            self.prms[int(tile_id)].modify_cfg(**prm_cfg.__dict__)
            self.prms[int(tile_id)].build_prm()

    def get(self, tile_id: int) -> "Prm":
        """Access the roadmap for a specific terrain tile."""
        return self.prms[tile_id]


class Prm:
    """Probabilistic roadmap over a single terrain tile's height field."""

    cfg: PrmConfig
    nodes: np.ndarray  # (N, 3) nodes in height-map coordinates (x, y, height)
    free_nodes: list  # indices into `nodes` that lie in free (non-wall/pit) space
    roadmap: dict  # adjacency list: node index -> list of connected neighbor indices
    height_map: torch.Tensor
    padding_mask: torch.Tensor
    free_mask: torch.Tensor

    def __init__(
        self,
        cfg: PrmConfig,
        env: "ManagerBasedRLEnv",
        tile_id: int,
        terrain_origin: torch.Tensor,
    ) -> None:
        self.cfg = cfg
        self.tile_id = tile_id
        self.height_map = env.scene.terrain._height_field_visual[tile_id]

        self.compute_free_mask()
        self.compute_padding_mask()

        self.terrain_origin = terrain_origin
        terrain_gen_cfg = env.cfg.scene.terrain.terrain_generator
        self.h_scale = terrain_gen_cfg.horizontal_scale
        self.v_scale = terrain_gen_cfg.vertical_scale
        self.size = terrain_gen_cfg.size

        self.build_prm()
        if self.cfg.visualise:
            self.visualise()

    def modify_cfg(self, **kwargs):
        """Modify the PRM configuration."""
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
            else:
                raise AttributeError(f"PrmConfig has no attribute '{key}'")

    def compute_free_mask(self):
        """Mark cells that are valid ground or platform (i.e. safe to route through)."""
        h = self.height_map
        ground_mask = (h >= THRESHOLDS.GROUND_MIN) & (h <= THRESHOLDS.GROUND_MAX)
        platform_mask = (h >= THRESHOLDS.PLATFORM_MIN) & (h <= THRESHOLDS.PLATFORM_MAX)
        self.free_mask = ground_mask | platform_mask

    def compute_padding_mask(self):
        """Mark cells near a height "cliff" (large height jump between neighbors) as non-traversable.

        This is a PRM-local traversability margin, distinct from the terrain generator's own
        goal/spawn safety padding (`terrain_constants.PADDING`).
        """
        cliff_height_threshold = 50

        h = self.height_map
        cliff_mask = torch.zeros_like(h, dtype=torch.bool)

        v_exceeds = torch.abs(h[:-1, :] - h[1:, :]) > cliff_height_threshold
        cliff_mask[:-1, :] |= v_exceeds
        cliff_mask[1:, :] |= v_exceeds

        h_exceeds = torch.abs(h[:, :-1] - h[:, 1:]) > cliff_height_threshold
        cliff_mask[:, :-1] |= h_exceeds
        cliff_mask[:, 1:] |= h_exceeds

        # dilate the cliff mask so nodes/segments near a cliff are also excluded
        k = 2 * self.cfg.padding + 1
        mask_float = cliff_mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
        dilated = torch.nn.functional.max_pool2d(mask_float, kernel_size=k, stride=1, padding=self.cfg.padding)
        self.padding_mask = dilated.squeeze(0).squeeze(0).to(torch.bool)

    def sample_free_point(self) -> tuple[int, int, int]:
        """Randomly sample a non-padded point from the height map, returning (x, y, height)."""
        max_attempts = 10000
        rows, cols = self.height_map.shape
        for _ in range(max_attempts):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
            if not self.padding_mask[x, y]:
                return (x, y, int(self.height_map[x, y]))
        raise RuntimeError(f"Could not sample free point after {max_attempts} attempts. Check height map and padding settings.")

    def k_closest(self, points: np.ndarray, k: int) -> list[list[int]]:
        """Return the k closest neighbors (by 2D distance) for each point, excluding itself."""
        assert isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3, (
            f"points must have shape (N, 3), got {getattr(points, 'shape', None)}"
        )

        points_2d = points[:, :2]
        dist_mat = distance_matrix(points_2d, points_2d)

        neighbors = []
        for i in range(len(points)):
            idx = np.argsort(dist_mat[i])[1 : k + 1]
            neighbors.append(idx.tolist())
        return neighbors

    def reset_start_goal(self, start: np.ndarray, goal: np.ndarray):
        """Insert/replace the start and goal nodes and (re)connect them to their k-nearest neighbors."""
        assert start.shape == (3,), "Start must be a 1D array of shape (3,)."
        assert goal.shape == (3,), "Goal must be a 1D array of shape (3,)."

        if self.cfg.start_idx is None:
            self.cfg.start_idx = len(self.nodes)
            if self.cfg.start_idx in self.roadmap:
                raise RuntimeError("Start index already exists in roadmap; set start_idx explicitly in PrmConfig.")
            self.nodes = np.vstack((self.nodes, start))

        if self.cfg.goal_idx is None:
            self.cfg.goal_idx = len(self.nodes)
            if self.cfg.goal_idx in self.roadmap:
                raise RuntimeError("Goal index already exists in roadmap; set goal_idx explicitly in PrmConfig.")
            self.nodes = np.vstack((self.nodes, goal))

        self.nodes[self.cfg.start_idx] = start
        self.nodes[self.cfg.goal_idx] = goal

        idx = [self.cfg.start_idx, self.cfg.goal_idx]
        start_goal_points = np.array([start, goal])
        nodes_2d = self.nodes[:, :2]
        distances = distance_matrix(start_goal_points[:, :2], nodes_2d)
        neighbors = [np.argsort(distances[i])[1 : self.cfg.k + 1] for i in range(2)]

        # detach start/goal from any previous connections before reconnecting
        if self.cfg.start_idx in self.roadmap:
            for nb in self.roadmap[self.cfg.start_idx]:
                self.roadmap[nb].remove(self.cfg.start_idx)
        if self.cfg.goal_idx in self.roadmap:
            for nb in self.roadmap[self.cfg.goal_idx]:
                self.roadmap[nb].remove(self.cfg.goal_idx)

        self.roadmap[self.cfg.start_idx] = []
        self.roadmap[self.cfg.goal_idx] = []

        for i in range(len(idx)):
            for j in neighbors[i]:
                if self.is_traversable(self.nodes[idx[i]], self.nodes[j], check_padding=True, rev=True):
                    if j not in self.roadmap[idx[i]]:
                        self.roadmap[idx[i]].append(j)
                        self.roadmap[j].append(idx[i])

    def is_traversable(self, p1: np.ndarray, p2: np.ndarray, check_padding: bool = True, rev: bool = False) -> bool:
        """Check whether the straight line between two height-map points is traversable.

        Simulates sensor noise via `PrmConfig.fpr`/`fnr` (false positive / false negative rate).
        """
        rr, cc = line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))

        p1_padding = self.padding_mask[int(p1[0]), int(p1[1])]
        p2_padding = self.padding_mask[int(p2[0]), int(p2[1])]
        points_not_padding = (not p1_padding) and (not p2_padding)

        if rev:
            rr_rev, cc_rev = line(int(p2[1]), int(p2[0]), int(p1[1]), int(p1[0]))
            if check_padding and points_not_padding:
                if self.padding_mask[cc_rev, rr_rev].any():
                    return False
                heights = self.height_map[cc_rev, rr_rev]
                if not np.all(np.abs(np.diff(heights)) <= self.cfg.max_height_diff):
                    return False

        if check_padding and points_not_padding:
            if self.padding_mask[cc, rr].any():
                return False

        heights = self.height_map[cc, rr]
        traversable = np.all(np.abs(np.diff(heights)) <= self.cfg.max_height_diff)

        if traversable:
            return random.random() >= self.cfg.fnr  # simulate false negative
        else:
            return random.random() < self.cfg.fpr  # simulate false positive

    def build_prm(self):
        """Sample `PrmConfig.N` free points and connect each to its k-nearest traversable neighbors."""
        nodes = []
        free_nodes_mask = []
        for _ in range(self.cfg.N):
            nodes.append(self.sample_free_point())
            free_nodes_mask.append(bool(self.free_mask[nodes[-1][0], nodes[-1][1]]))
        nodes = np.array(nodes)

        self.free_nodes = [i for i in range(len(nodes)) if free_nodes_mask[i]]

        roadmap = {i: [] for i in range(len(nodes))}
        neighbors = self.k_closest(nodes, self.cfg.k)
        for i, node_neighbors in enumerate(neighbors):
            for j in node_neighbors:
                if self.is_traversable(nodes[i], nodes[j], rev=True):
                    if j not in roadmap[i]:
                        roadmap[i].append(j)
                        roadmap[j].append(i)

        self.nodes = nodes
        self.roadmap = roadmap

    def rescale_points_to_world(self, points: torch.Tensor) -> torch.Tensor:
        """Rescale points from height-map coordinates to world coordinates."""
        rescaled = points.clone().to(self.terrain_origin.device)
        rescaled[:, 0] = rescaled[:, 0] * self.h_scale - self.size[0] / 2 + self.terrain_origin[0]
        rescaled[:, 1] = rescaled[:, 1] * self.h_scale - self.size[1] / 2 + self.terrain_origin[1]
        rescaled[:, 2] = rescaled[:, 2] * self.v_scale
        return rescaled

    def rescale_points_to_heightmap(self, points: torch.Tensor) -> torch.Tensor:
        """Rescale points from world coordinates to height-map coordinates.

        If `points` only has (x, y), the height is looked up from the height map.
        """
        rescaled = points.clone()
        rescaled[:, 0] = (rescaled[:, 0] - self.terrain_origin[0] + self.size[0] / 2) / self.h_scale
        rescaled[:, 1] = (rescaled[:, 1] - self.terrain_origin[1] + self.size[1] / 2) / self.h_scale

        if points.shape[1] == 2:
            heights = torch.zeros((rescaled.shape[0], 1), device=rescaled.device)
            for i in range(rescaled.shape[0]):
                x, y = int(rescaled[i, 0].item()), int(rescaled[i, 1].item())
                heights[i, 0] = self.height_map[x, y]
            rescaled = torch.cat((rescaled[:, :2], heights), dim=1)
        elif points.shape[1] == 3:
            rescaled[:, 2] = rescaled[:, 2] / self.v_scale
        else:
            raise RuntimeError("points must have shape (N, 2) or (N, 3).")
        return rescaled

    def visualise(self):
        from omni.isaac.debug_draw import _debug_draw

        rescaled_nodes = self.rescale_points_to_world(torch.tensor(self.nodes, dtype=torch.float32)).tolist()
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.draw_points(rescaled_nodes, [(1, 0, 0, 1)] * len(rescaled_nodes), [10] * len(rescaled_nodes))

        point_list_1, point_list_2 = [], []
        for i in range(len(self.roadmap)):
            point_list_1 += [rescaled_nodes[i]] * len(self.roadmap[i])
            point_list_2 += [rescaled_nodes[j] for j in self.roadmap[i]]
        draw.draw_lines(point_list_1, point_list_2, [(0, 0, 1, 1)] * len(point_list_2), [2] * len(point_list_2))
