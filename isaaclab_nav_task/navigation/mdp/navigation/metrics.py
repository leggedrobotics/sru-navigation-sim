# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Small metric trackers used by navigation command terms."""

from __future__ import annotations

import torch


class RollingMetricTracker:
    """Tracks scalar episode metrics over a rolling buffer."""

    def __init__(self, num_envs: int, device: torch.device, buffer_size: int = 10):
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = torch.full((num_envs, buffer_size), float("nan"), device=device)
        self.write_index = torch.zeros(num_envs, dtype=torch.long, device=device)

    def record(self, values: torch.Tensor, env_ids: torch.Tensor):
        indices = self.write_index[env_ids] % self.buffer_size
        self.buffer[env_ids, indices] = values
        self.write_index[env_ids] += 1

    def get_mean(self) -> torch.Tensor:
        valid = ~torch.isnan(self.buffer)
        count = valid.sum(dim=1).clamp(min=1)
        total = torch.where(valid, self.buffer, torch.zeros_like(self.buffer)).sum(dim=1)
        return total / count.float()

    def get_mean_spl(self) -> torch.Tensor:
        """Compatibility alias used by existing evaluation scripts."""
        return self.get_mean()
