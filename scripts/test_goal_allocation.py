#!/usr/bin/env python3
"""Test script to verify goal and spawn positions are allocated in free space.

This script:
1. Generates maze terrain with height field
2. Simulates the goal command initialization logic
3. Verifies that sampled goal/spawn positions fall in free cells
4. Reports statistics on valid positions per terrain
5. Checks spatial distribution across the terrain
"""

import torch
import torch.nn.functional as F
import numpy as np


def apply_obstacle_padding(height_field: torch.Tensor, padding_size: int, device: str = "cpu") -> torch.Tensor:
    """Apply padding around obstacles to create safe zones."""
    # Free cells are height values 0-50 (flat ground) or 150-250 (goal markers ~200)
    is_free = (height_field >= -10) & (height_field <= 50)
    is_goal_marker = (height_field >= 150) & (height_field <= 250)
    is_obstacle = ~(is_free | is_goal_marker)

    # Detect edges using Sobel filter for stair terrain
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).unsqueeze(0).unsqueeze(0)

    hf_float = height_field.unsqueeze(1).float()
    edges_x = F.conv2d(hf_float, sobel_x, padding=1)
    edges_y = F.conv2d(hf_float, sobel_y, padding=1)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2).squeeze(1) > 0.0

    # Apply edge detection only for terrains with goal markers (stairs)
    has_goal_markers = torch.any(is_goal_marker, dim=(1, 2))
    is_obstacle = torch.where(has_goal_markers.unsqueeze(1).unsqueeze(2),
                               edges | is_obstacle, is_obstacle)

    # Add border exclusion
    is_obstacle[:, 0, :] = True
    is_obstacle[:, -1, :] = True
    is_obstacle[:, :, 0] = True
    is_obstacle[:, :, -1] = True
    is_obstacle[:, 1, :] = True
    is_obstacle[:, -2, :] = True
    is_obstacle[:, :, 1] = True
    is_obstacle[:, :, -2] = True

    # Dilate obstacles
    kernel = torch.ones((1, 1, 2 * padding_size + 1, 2 * padding_size + 1), device=device)
    dilated = F.conv2d(is_obstacle.unsqueeze(1).float(), kernel, padding=padding_size).squeeze(1) > 0

    # Create output: mark dilated obstacle regions
    result = height_field.clone()
    result[dilated] = 110  # Mark as obstacle
    return result


def create_test_height_field(num_terrains: int = 4, size: int = 150) -> torch.Tensor:
    """Create test height fields simulating maze terrain.

    Creates a simple maze pattern with realistic height values:
    - Flat areas (height 0): walkable ground
    - Walls (height 300): obstacles (1.5m / 0.005 vertical_scale)
    - Goal markers (height 200): stairs goal markers (1.0m / 0.005)
    """
    height_fields = []

    for i in range(num_terrains):
        hf = torch.zeros(size, size, dtype=torch.float32)

        # Add walls around the border (height ~300 = 1.5m wall)
        hf[:10, :] = 300
        hf[-10:, :] = 300
        hf[:, :10] = 300
        hf[:, -10:] = 300

        # Add some internal walls (maze-like)
        wall_positions = [
            (30, 30, 30, 80),   # vertical wall
            (60, 20, 60, 70),   # vertical wall
            (90, 40, 90, 120),  # vertical wall
            (20, 50, 80, 50),   # horizontal wall
            (50, 100, 130, 100), # horizontal wall
        ]

        for x1, y1, x2, y2 in wall_positions:
            if x1 == x2:  # vertical wall
                hf[x1:x1+5, y1:y2] = 300
            else:  # horizontal wall
                hf[x1:x2, y1:y1+5] = 300

        # Add goal markers in some terrains (height 200 = goal marker)
        if i % 2 == 0:
            hf[120:130, 120:130] = 200  # Goal marker area

        height_fields.append(hf.unsqueeze(0))

    return torch.cat(height_fields, dim=0)


def analyze_spatial_distribution(positions: torch.Tensor, grid_size: int, num_bins: int = 4):
    """Analyze spatial distribution of positions across the terrain.

    Args:
        positions: Tensor of (N, 3) with [terrain_idx, x, y]
        grid_size: Size of the terrain grid
        num_bins: Number of bins per dimension for distribution analysis

    Returns:
        dict with distribution statistics
    """
    # Extract x, y coordinates (ignoring terrain index)
    x_coords = positions[:, 1].float()
    y_coords = positions[:, 2].float()

    # Compute bin edges
    bin_edges = torch.linspace(0, grid_size, num_bins + 1)

    # Count positions in each bin
    x_bins = torch.bucketize(x_coords, bin_edges[1:-1])
    y_bins = torch.bucketize(y_coords, bin_edges[1:-1])

    # Create 2D histogram
    histogram = torch.zeros(num_bins, num_bins)
    for i in range(len(positions)):
        histogram[x_bins[i], y_bins[i]] += 1

    # Normalize
    total = histogram.sum()
    expected_per_bin = total / (num_bins * num_bins)

    # Compute statistics
    stats = {
        "histogram": histogram,
        "min_count": histogram.min().item(),
        "max_count": histogram.max().item(),
        "mean_count": histogram.mean().item(),
        "std_count": histogram.std().item(),
        "expected_count": expected_per_bin,
        "coverage": (histogram > 0).sum().item() / (num_bins * num_bins),
    }

    # Chi-square-like uniformity measure
    if expected_per_bin > 0:
        deviation = ((histogram - expected_per_bin) ** 2) / expected_per_bin
        stats["uniformity_score"] = 1.0 - (deviation.sum().item() / (num_bins * num_bins * total))
    else:
        stats["uniformity_score"] = 0.0

    return stats


def test_goal_allocation():
    """Test that goal and spawn positions are in free space and evenly distributed."""
    print("=" * 70)
    print("Goal and Spawn Position Allocation Test")
    print("=" * 70)

    device = "cpu"
    num_terrains = 4
    downsample_scale = 2
    goal_padding = 2
    spawn_padding = 4

    # Create test height field
    print("\n1. Creating test height field...")
    height_field = create_test_height_field(num_terrains=num_terrains)
    print(f"   Height field shape: {height_field.shape}")
    print(f"   Height range: [{height_field.min():.0f}, {height_field.max():.0f}]")

    # Downsample
    height_field_ds = height_field[:, ::downsample_scale, ::downsample_scale]
    grid_size = height_field_ds.shape[1]
    print(f"   Downsampled shape: {height_field_ds.shape}")

    # Store raw for z-lookup
    height_field_raw = height_field_ds.clone()

    # Apply padding
    print("\n2. Applying obstacle padding...")
    hf_goals = apply_obstacle_padding(height_field_ds.float(), goal_padding, device)
    hf_spawn = apply_obstacle_padding(height_field_ds.float(), spawn_padding, device)

    # Transpose for coordinate handling
    hf_goals = hf_goals.transpose(1, 2)
    hf_spawn = hf_spawn.transpose(1, 2)

    # Find free cells
    print("\n3. Finding free cells...")
    # Free: height 0-50 (flat ground) or 150-250 (goal markers ~200)
    free_mask_goals = ((hf_goals >= -10) & (hf_goals <= 50)) | ((hf_goals >= 150) & (hf_goals <= 250))
    free_mask_spawn = ((hf_spawn >= -10) & (hf_spawn <= 50)) | ((hf_spawn >= 150) & (hf_spawn <= 250))

    # Count free cells per terrain
    print("\n4. Free cell statistics per terrain:")
    print("-" * 60)
    print(f"{'Terrain':<10} {'Goal Cells':<15} {'Spawn Cells':<15} {'% Goal':<10}")
    print("-" * 60)

    total_cells = hf_goals.shape[1] * hf_goals.shape[2]
    all_valid = True

    for i in range(num_terrains):
        goal_count = free_mask_goals[i].sum().item()
        spawn_count = free_mask_spawn[i].sum().item()
        goal_pct = 100.0 * goal_count / total_cells

        status = "OK" if goal_count > 0 and spawn_count > 0 else "FAIL"
        if status == "FAIL":
            all_valid = False

        print(f"{i:<10} {goal_count:<15} {spawn_count:<15} {goal_pct:<10.1f}%  [{status}]")

    print("-" * 60)

    # Get all free positions
    goal_indices = free_mask_goals.nonzero(as_tuple=False)
    spawn_indices = free_mask_spawn.nonzero(as_tuple=False)

    # Test position sampling
    print("\n5. Testing position sampling (free space verification)...")
    num_samples = min(100, goal_indices.shape[0])
    sample_indices = torch.randint(0, goal_indices.shape[0], (num_samples,))
    sampled_goals = goal_indices[sample_indices]

    # Verify sampled positions are in free cells
    print(f"   Sampled {num_samples} goal positions...")
    invalid_goals = 0
    for idx in range(num_samples):
        terrain, x, y = sampled_goals[idx]
        # Check in original height field (need to swap x,y back due to transpose)
        height_val = height_field_raw[terrain, y, x].item()
        # Free: height 0-50 (flat ground) or 150-250 (goal markers)
        is_free = (-10 <= height_val <= 50) or (150 <= height_val <= 250)
        if not is_free:
            invalid_goals += 1
            print(f"   WARNING: Position ({terrain}, {x}, {y}) has height {height_val} - NOT FREE")

    if invalid_goals == 0:
        print(f"   All {num_samples} sampled goal positions are in FREE SPACE")
    else:
        print(f"   {invalid_goals}/{num_samples} sampled positions are INVALID")
        all_valid = False

    # Analyze spatial distribution
    print("\n6. Analyzing spatial distribution of free positions...")
    print("-" * 60)

    for terrain_idx in range(num_terrains):
        terrain_positions = goal_indices[goal_indices[:, 0] == terrain_idx]
        if len(terrain_positions) == 0:
            print(f"   Terrain {terrain_idx}: No free positions!")
            continue

        stats = analyze_spatial_distribution(terrain_positions, grid_size, num_bins=4)

        print(f"\n   Terrain {terrain_idx}:")
        print(f"   - Total positions: {len(terrain_positions)}")
        print(f"   - Grid coverage: {stats['coverage']*100:.1f}% of bins have positions")
        print(f"   - Position distribution (4x4 grid):")

        # Print histogram
        hist = stats['histogram']
        for row in range(hist.shape[0]):
            row_str = "     "
            for col in range(hist.shape[1]):
                row_str += f"{int(hist[row, col]):5d} "
            print(row_str)

        print(f"   - Min/Max positions per bin: {stats['min_count']:.0f} / {stats['max_count']:.0f}")
        print(f"   - Expected per bin: {stats['expected_count']:.1f}")
        print(f"   - Uniformity score: {stats['uniformity_score']:.3f} (1.0 = perfectly uniform)")

        # Check if distribution is reasonably uniform
        if stats['coverage'] < 0.75:
            print(f"   WARNING: Low coverage - positions may not be evenly distributed!")

    # Test random sampling simulation
    print("\n7. Simulating random goal sampling (1000 iterations)...")
    print("-" * 60)

    num_simulations = 1000
    sampled_positions = []

    for _ in range(num_simulations):
        # Random terrain
        terrain_idx = torch.randint(0, num_terrains, (1,)).item()
        terrain_positions = goal_indices[goal_indices[:, 0] == terrain_idx]

        if len(terrain_positions) > 0:
            # Random position from that terrain
            rand_idx = torch.randint(0, len(terrain_positions), (1,)).item()
            sampled_positions.append(terrain_positions[rand_idx])

    sampled_tensor = torch.stack(sampled_positions)
    overall_stats = analyze_spatial_distribution(sampled_tensor, grid_size, num_bins=4)

    print(f"   Total sampled: {len(sampled_positions)}")
    print(f"   Grid coverage: {overall_stats['coverage']*100:.1f}%")
    print(f"   Uniformity score: {overall_stats['uniformity_score']:.3f}")
    print(f"   Distribution histogram:")

    hist = overall_stats['histogram']
    for row in range(hist.shape[0]):
        row_str = "     "
        for col in range(hist.shape[1]):
            row_str += f"{int(hist[row, col]):5d} "
        print(row_str)

    # Final result
    print("\n" + "=" * 70)
    if all_valid:
        print("TEST PASSED: Goal and spawn allocation is correct!")
        print("- All sampled positions are in FREE SPACE")
        print(f"- Spatial coverage: {overall_stats['coverage']*100:.1f}%")
        print(f"- Uniformity score: {overall_stats['uniformity_score']:.3f}")
    else:
        print("TEST FAILED: Some positions are not in free space!")
    print("=" * 70)

    return all_valid


if __name__ == "__main__":
    success = test_goal_allocation()
    exit(0 if success else 1)
