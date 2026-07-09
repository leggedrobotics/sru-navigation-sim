#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Standalone correctness test for `prm.py` - no Isaac Sim required.

Loads the real shipped module directly by file path (bypassing `isaaclab_nav_task`'s package
`__init__` chain, which does terrain monkey-patching that needs a real Isaac Sim environment).
Exercises `Prm`/`MultiPRM` on a small synthetic maze with a known layout (a wall with a wide gap),
so free-space/traversability results can be checked against a hand-computed expectation instead of
just "did it crash".

Run directly: `python3 tests/test_prm.py`
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
NAV_MDP_DIR = REPO_ROOT / "isaaclab_nav_task" / "navigation" / "mdp" / "navigation"
TERRAIN_CONSTANTS_PATH = REPO_ROOT / "isaaclab_nav_task" / "terrains" / "terrain_constants.py"


def _fake_namespace_package(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # marks it as a package for submodule resolution
    sys.modules[name] = mod
    return mod


def load_prm_module():
    """Load prm.py directly, without needing a real Isaac Sim install."""
    for pkg in [
        "isaaclab_nav_task",
        "isaaclab_nav_task.terrains",
        "isaaclab_nav_task.navigation",
        "isaaclab_nav_task.navigation.mdp",
        "isaaclab_nav_task.navigation.mdp.navigation",
    ]:
        _fake_namespace_package(pkg)

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("isaaclab_nav_task.terrains.terrain_constants", TERRAIN_CONSTANTS_PATH)
    return _load("isaaclab_nav_task.navigation.mdp.navigation.prm", NAV_MDP_DIR / "prm.py")


class _FakeTerrainGeneratorCfg:
    def __init__(self, horizontal_scale, vertical_scale, size, num_rows, num_cols, curriculum):
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.size = size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.curriculum = curriculum


class FakeEnv:
    """Minimal stand-in exposing exactly what Prm/MultiPRM read from a real ManagerBasedRLEnv."""

    def __init__(self, height_field: torch.Tensor, horizontal_scale: float = 0.1, vertical_scale: float = 0.005):
        num_terrains, W, H = height_field.shape
        terrain_gen_cfg = _FakeTerrainGeneratorCfg(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            size=(W * horizontal_scale, H * horizontal_scale),
            num_rows=1,
            num_cols=num_terrains,
            curriculum=False,
        )
        self.cfg = types.SimpleNamespace(scene=types.SimpleNamespace(terrain=types.SimpleNamespace(terrain_generator=terrain_gen_cfg)))
        terrain_origins = torch.zeros((1, num_terrains, 3))
        self.scene = types.SimpleNamespace(
            terrain=types.SimpleNamespace(_height_field_visual=height_field, terrain_origins=terrain_origins)
        )


def make_synthetic_height_map(size: int = 100, gap_width: int = 15) -> torch.Tensor:
    """A ground-level room (height=0) with a wall (height=300) splitting it in half, except for a
    `gap_width`-cell gap - i.e. reaching the far side requires routing through the gap. `gap_width`
    is generous (not a 1-2 cell slit): PRM connectivity through a passage depends on random samples
    actually landing in it, so a too-narrow gap tests sampling density, not correctness.
    """
    height_map = torch.zeros((size, size), dtype=torch.int16)
    wall_y = size // 2
    height_map[:, wall_y] = 300  # WALL, per terrain_constants.HEIGHTS.WALL
    height_map[0:gap_width, wall_y] = 0  # gap
    return height_map


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    prm_mod = load_prm_module()
    print("Loaded prm.py directly (no Isaac Sim needed).")

    height_map = make_synthetic_height_map(size=100, gap_width=15)
    env = FakeEnv(height_map.unsqueeze(0))  # (num_terrains=1, W, H)

    prm_cfg = prm_mod.PrmConfig(N=2000, k=12, max_height_diff=50.0, padding=2)
    multi_prm = prm_mod.MultiPRM(prm_cfg, env)
    assert multi_prm.num_terrains == 1
    prm = multi_prm.get(0)

    assert prm.free_mask.sum() > 0, "free_mask should mark the open ground cells as free"
    # height_map[x, y] convention (matches Prm.sample_free_point's own indexing): the wall sits at
    # column-index (2nd axis) 50 for x >= gap_width, so (50, 50) is a solid wall cell.
    assert prm.free_mask[50, 50].item() is False, "a wall cell (outside the gap) must not be free"
    assert prm.free_mask[10, 10].item() is True, "an open ground cell must be free"

    assert prm.is_traversable(np.array([10, 10, 0]), np.array([10, 20, 0]), check_padding=False), (
        "a segment within open ground should be traversable"
    )
    assert not prm.is_traversable(np.array([50, 45, 0]), np.array([50, 55, 0]), check_padding=False), (
        "a straight line through the solid wall must not be traversable"
    )

    n_edges = sum(len(v) for v in prm.roadmap.values())
    print(f"[Prm] nodes={len(prm.nodes)}, free_nodes={len(prm.free_nodes)}, roadmap_edges={n_edges}")
    assert n_edges > 0, "roadmap should have at least some traversable connections"

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
