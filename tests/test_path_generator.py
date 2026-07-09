#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Standalone correctness test for `path_generator.py` (builds on `prm.py`) - no Isaac Sim required.

See `test_prm.py` for the loading approach (direct file-path import, isaaclab.utils.math stubbed).
Exercises `GlobalPath`/`MultiPath` end-to-end on the same synthetic maze as `test_prm.py`: a wall
with a wide gap, so a valid path from one side to the other must route through the gap, not a
straight line - and reaching the goal at all is a direct regression test against the "path silently
zeroed out" bug this port fixes (see PR description / branch_changes_report.md item F2).

Run directly: `python3 tests/test_path_generator.py`
"""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prm import FakeEnv, load_prm_module, make_synthetic_height_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
NAV_MDP_DIR = REPO_ROOT / "isaaclab_nav_task" / "navigation" / "mdp" / "navigation"


def transform_points(points: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rigid transform: rotate `points` by quaternion `quat` (w, x, y, z), then translate by `pos`.

    Broadcasts pos/quat across an extra middle dimension if points is (B, N, 3) vs. pos/quat (B, 3)/(B, 4).
    Stands in for `isaaclab.utils.math.transform_points`, which needs a real Isaac Sim install.
    """
    if points.dim() == 3:
        pos = pos.unsqueeze(1)
        quat = quat.unsqueeze(1)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qvec = torch.stack([x, y, z], dim=-1)
    uv = torch.cross(qvec, points, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    rotated = points + 2 * (w.unsqueeze(-1) * uv + uuv)
    return rotated + pos


def load_path_generator_module():
    """Assumes load_prm_module() (or an equivalent import of prm.py) already ran."""
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    sys.modules["isaaclab"] = isaaclab
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.__path__ = []
    sys.modules["isaaclab.utils"] = isaaclab_utils
    isaaclab_utils_math = types.ModuleType("isaaclab.utils.math")
    isaaclab_utils_math.transform_points = transform_points
    sys.modules["isaaclab.utils.math"] = isaaclab_utils_math

    spec = importlib.util.spec_from_file_location(
        "isaaclab_nav_task.navigation.mdp.navigation.path_generator", str(NAV_MDP_DIR / "path_generator.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    load_prm_module()  # registers isaaclab_nav_task.navigation.mdp.navigation.prm in sys.modules
    path_gen_mod = load_path_generator_module()
    print("Loaded prm.py and path_generator.py directly (no Isaac Sim needed).")

    gap_width = 15
    height_map = make_synthetic_height_map(size=100, gap_width=gap_width)
    env = FakeEnv(height_map.unsqueeze(0))

    prm_mod = sys.modules["isaaclab_nav_task.navigation.mdp.navigation.prm"]
    prm_cfg = prm_mod.PrmConfig(N=2000, k=12, max_height_diff=50.0, padding=2)
    multi_prm = prm_mod.MultiPRM(prm_cfg, env)
    prm = multi_prm.get(0)

    # --- GlobalPath / MultiPath: force "optimal" so the result is checkable against a known layout ---
    path_cfg = path_gen_mod.GlobalPathConfig(n_waypoints=0, num_smooth_points=15, path_categories=["optimal"], category_weights=[1.0])
    multi_path = path_gen_mod.MultiPath(num_envs=4, cfg_path=path_cfg, prm_manager=multi_prm, device=torch.device("cpu"))

    tile_ids = torch.zeros(4, dtype=torch.long)

    def hm_to_world(x, y):
        # MultiPath.rebuild expects world-frame (x, y); convert from the height-map coords we want.
        return [x * 0.1 - 5.0, y * 0.1 - 5.0]

    starts = torch.tensor([hm_to_world(10, 10)] * 4)  # one side of the wall
    goals = torch.tensor([hm_to_world(10, 90)] * 4)  # other side -> must route through the gap
    multi_path.rebuild(path_cfg, starts, goals, tile_ids, env_ids=list(range(4)))

    for env_id in range(4):
        path = np.array(multi_path.get(env_id).path)
        assert path.shape == (15, 3), f"expected 15 waypoints x 3 dims, got {path.shape}"
        assert not np.allclose(path, 0.0), "REGRESSION: path is all-zero (this is exactly the F2 bug this port fixes)"

        goal_hm = prm.rescale_points_to_heightmap(goals[env_id : env_id + 1]).squeeze(0).numpy()
        dist_to_goal = np.linalg.norm(path[-1, :2] - goal_hm[:2])
        assert dist_to_goal < 2.0, f"path does not end near the goal (dist={dist_to_goal:.2f})"
        assert (path[:, 0] <= gap_width + 2).any(), "path never routes through the known gap in the wall"

    print("[GlobalPath] all 4 paths reach their goal, route through the gap, none are degenerate/zero.")

    # --- MultiPath.update() / get_all_path_metrics() smoke test ---
    inv_pos = torch.zeros(4, 3)
    inv_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)  # identity quaternion (w, x, y, z)
    mean_dist = multi_path.update(inv_pos, inv_rot, robot_pos_w=torch.zeros(4, 3), robot_quat_w=inv_rot)
    assert torch.isfinite(mean_dist)

    dists, progress = multi_path.get_all_path_metrics()
    assert dists.shape == (4,) and progress.shape == (4,)
    assert torch.isfinite(dists).all() and torch.isfinite(progress).all()
    print(f"[MultiPath] update()/get_all_path_metrics() ran cleanly: mean_dist={mean_dist:.3f}")

    # --- every path category: each hits different code (sample_waypoints, smooth_path_los vs.
    # smooth_path_los_for_non_optimal, add_noise_to_path for infeasible) ---
    for category in ["optimal", "mildly_non_optimal", "highly_non_optimal", "infeasible"]:
        cat_cfg = path_gen_mod.GlobalPathConfig(n_waypoints=1, num_smooth_points=15, path_categories=[category], category_weights=[1.0])
        for trial in range(5):
            multi_path.rebuild(cat_cfg, starts, goals, tile_ids, env_ids=list(range(4)))
            for env_id in range(4):
                path = np.array(multi_path.get(env_id).path)
                assert path.shape == (15, 3)
                assert not np.allclose(path, 0.0), f"REGRESSION: {category} trial {trial} produced an all-zero path"
                assert np.isfinite(path).all(), f"{category} trial {trial} produced non-finite path values"
        print(f"[GlobalPath] category='{category}': 5 rebuilds x 4 envs, no crashes, no degenerate paths.")

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
