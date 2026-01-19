#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a trained navigation policy (PPO/MDPO) with automatic checkpoint loading.

Usage:
    python scripts/play.py --task <task_name> [options]

Arguments:
    --task                   Task name (required, typically *-Play-v0 variant)
    --checkpoint             Path to model checkpoint (.pt file)
    --use_last_checkpoint    Use latest checkpoint from logs (default behavior)
    --num_envs              Number of parallel environments
    --video                 Enable video recording
    --video_length          Video length in steps (default: 200)

Examples:
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0 --checkpoint path/to/model.pt
    python scripts/play.py --task Isaac-Navigation-B2W-Play-v0 --video --num_envs 16

Note: Automatically finds latest checkpoint if --checkpoint not specified.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint from logs.")
parser.add_argument("--export_jit", action="store_true", default=False, help="Export policy as JIT module.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX model.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras
args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching simulation
import gymnasium as gym
import os
import re
import torch

from rsl_rl.runners import OnPolicyRunner

# Import Isaac Lab extensions
import isaaclab_tasks  # noqa: F401
import isaaclab_nav_task  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx


def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    """Find the latest checkpoint file in the log directory.

    Args:
        log_path: Base log directory path
        checkpoint_pattern: Regex pattern for checkpoint files

    Returns:
        Path to the latest checkpoint file
    """
    # Find all run directories
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    # Sort to get latest run
    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    # Find checkpoint files
    checkpoint_files = []
    for f in os.listdir(run_path):
        if re.match(checkpoint_pattern, f):
            checkpoint_files.append(f)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    # Sort to get latest checkpoint
    checkpoint_files.sort(key=lambda m: f"{m:0>15}")
    latest_checkpoint = checkpoint_files[-1]

    return os.path.join(run_path, latest_checkpoint)


def load_checkpoint_with_fallback(runner: OnPolicyRunner, checkpoint_path: str, load_optimizer: bool = True):
    """Load checkpoint with fallback for PyTorch compatibility issues.

    Args:
        runner: RSL-RL runner instance
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to load optimizer state
    """
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint to CPU first for compatibility
    loaded_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load model state - handle both standard algorithms (PPO) and MDPO
    if runner.is_mdpo:
        # MDPO uses two actor-critics, load same state into both
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        # Standard algorithms use one actor-critic
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    # Load normalizers if using empirical normalization
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    # Load optimizer if requested
    if load_optimizer:
        if runner.is_mdpo:
            runner.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
        else:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def export_policy_jit(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as JIT module to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Export using the module's export_jit method
    print(f"[INFO] Exporting JIT policy to: {export_dir}")
    actor_critic.export_jit(path=export_dir, filename="policy.pt", normalizer=normalizer)
    print(f"[INFO] JIT export complete!")


def export_policy_onnx(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as ONNX model to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    # Determine export directory (create 'export' folder in the same directory as checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    # Get the actor-critic module
    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    # Get normalizer if using empirical normalization
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    # Check if the module has export_onnx method
    if not hasattr(actor_critic, "export_onnx"):
        raise NotImplementedError(
            f"ONNX export not implemented for {type(actor_critic).__name__}. "
            "Please add an export_onnx method to this module."
        )

    # Export using the module's export_onnx method
    print(f"[INFO] Exporting ONNX policy to: {export_dir}")
    actor_critic.export_onnx(path=export_dir, filename="policy.onnx", normalizer=normalizer)
    print(f"[INFO] ONNX export complete!")


def main():
    """Play navigation policy with RSL-RL."""
    # Parse command-line arguments
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    # Instantiate the configs
    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()

    # Override config from command line
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Wrap the environment
    env = RslRlVecEnvWrapper(env)

    # Get checkpoint path
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        # Get last checkpoint from log directory
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint with compatibility handling
    load_checkpoint_with_fallback(runner, resume_path)

    # Export JIT if requested
    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)

    # Export ONNX if requested
    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    # Obtain policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset environment
    obs, _ = env.get_observations()

    # Simulate environment
    while simulation_app.is_running():
        # Run policy
        with torch.inference_mode():
            actions = policy(obs)
        # Step environment
        obs, _, _, _ = env.step(actions)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close simulation
    simulation_app.close()
