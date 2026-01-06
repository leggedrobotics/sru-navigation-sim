# Terrain Generation and Goal Sampling

This document provides detailed information about the terrain generation system and goal/spawn position sampling used in the SRU navigation simulation.

## Table of Contents

- [Terrain Generation System](#terrain-generation-system)
  - [Architecture Overview](#architecture-overview)
  - [Key Files](#key-files)
  - [Mesh Optimization](#mesh-optimization)
  - [Terrain Data Flow](#terrain-data-flow)
  - [Maze Terrain Types](#maze-terrain-types)
  - [Safety Padding](#safety-padding)
  - [Terrain Configuration](#terrain-configuration)
  - [Curriculum Learning](#curriculum-learning)
- [Goal and Spawn Sampling](#goal-and-spawn-sampling)
  - [Architecture](#architecture)
  - [Key Features](#key-features)
  - [Coordinate System](#coordinate-system)
  - [Terrain Index Mapping](#terrain-index-mapping)
  - [Position Sampling](#position-sampling)
  - [Robot Spawn Height](#robot-spawn-height)

## Terrain Generation System

The extension includes custom maze terrain generators built on Isaac Lab's terrain generation system.

### Architecture Overview

```
Terrain Generation Flow:
┌────────────────────────────────────────────────────────────────────┐
│  1. HfMazeTerrainCfg                                               │
│     └─► maze_terrain() generates:                                  │
│         - heights: Height field for physics/rendering              │
│         - valid_mask: Valid goal positions (GOAL_PADDING=5 cells)  │
│         - spawn_mask: Valid spawn positions (SPAWN_PADDING=6 cells)│
│         - platform_mask: Elevated platforms for curriculum         │
│                                                                    │
│  2. TerrainGenerator (patched)                                     │
│     └─► Collects height field data from all sub-terrains           │
│     └─► Concatenates into single tensors per attribute             │
│                                                                    │
│  3. TerrainImporter (patched)                                      │
│     └─► Stores on self._height_field_* attributes                  │
│                                                                    │
│  4. RobotNavigationGoalCommand                                     │
│     └─► Reads from env.scene.terrain._height_field_*               │
│     └─► Creates PositionSampler with both masks                    │
└────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `hf_terrains_maze.py` | Terrain generation with explicit valid_mask/spawn_mask |
| `hf_terrains_maze_cfg.py` | Configuration dataclass (lines 41-51: mask storage attributes) |
| `terrain_constants.py` | Constants: PADDING (5/6 cells), HEIGHTS (0/200/300), VERTICAL_SCALE (0.005) |
| `patches.py` | Monkey-patches TerrainGenerator/TerrainImporter for mask storage |
| `maze_config.py` | MAZE_TERRAIN_CFG with sub-terrain configurations |

### Mesh Optimization

The extension includes automatic mesh optimization that significantly reduces GPU memory usage when training with many environments. This is especially important for large-scale RL training (4096+ environments).

**How it works:**
- Uses hierarchical block-based approach (20x20 → 10x10 → 5x5 blocks)
- Flat terrain regions are simplified to just 2 triangles instead of full mesh detail
- Non-flat regions recursively subdivide until 5x5 blocks, then generate detailed mesh
- Applied automatically via monkey-patching when the extension is imported

**Memory Reduction:**
| Terrain Type | Vertex Reduction |
|--------------|------------------|
| Flat terrain | ~99% |
| Maze-like | ~89% |
| Pits terrain | ~80% |
| Mixed terrain | ~79% |

This optimization is transparent - it produces visually identical terrains while dramatically reducing the mesh vertex count. The patches are applied before any terrain generation occurs, ensuring all height-field terrains benefit from the optimization.

### Terrain Data Flow

The terrain system uses **explicit boolean masks** instead of height-based classification:

```python
# During terrain generation (hf_terrains_maze.py)
terrain = TerrainData.create(width, height)  # Lines 57-63

# Mark obstacles as invalid
terrain.set_obstacle(x_start, x_end, y_start, y_end, wall_height)  # Lines 65-73

# Apply padding and create masks
terrain.apply_padding(PADDING.GOAL_PADDING)   # Lines 92-97: 5 cells = 0.5m for goals
spawn_mask = terrain.create_spawn_mask(PADDING.SPAWN_PADDING)  # Lines 99-104: 6 cells = 0.6m

# Store on config for patches to pick up
cfg.height_field_visual = heights      # For Z-lookup (terrain height)
cfg.height_field_valid_mask = valid_mask   # For goal sampling (5 cell padding)
cfg.height_field_spawn_mask = spawn_mask   # For spawn sampling (6 cell padding)
cfg.height_field_platform_mask = platform_mask  # For curriculum learning
```

These masks are set in the maze terrain function and picked up by the patches system (patches.py:45-66), which stores them on `TerrainImporter` for access via `env.scene.terrain._height_field_*`.

### Maze Terrain Types

Four terrain types are available via `HfMazeTerrainCfg`:

1. **Maze** (`non_maze_terrain=False, stairs=False`)
   - DFS-generated maze with configurable wall openings
   - Random obstacle shapes (pillars, bars, crosses, blocks)
   - Optional stairs integration (`add_stairs_to_maze=True`)

2. **Non-Maze/Random** (`non_maze_terrain=True`)
   - Random obstacle placement (~15-35% coverage based on difficulty)
   - Good for testing navigation without maze structure

3. **Stairs** (`stairs=True`)
   - 3x3 stair/platform structures with 4 cardinal stairways
   - Elevated platforms marked for curriculum learning
   - Tests robot climbing capabilities

4. **Pits** (`dynamic_obstacles=True`)
   - Pit rows with bridge crossings
   - Mix of pit (60%) and wall (40%) obstacles
   - Tests navigation over negative obstacles

### Safety Padding

Two padding levels ensure safe robot placement:

| Padding Type | Cells | Meters | Purpose |
|--------------|-------|--------|---------|
| `GOAL_PADDING` | 5 | 0.5m | Goal positions (robot just needs to reach) |
| `SPAWN_PADDING` | 6 | 0.6m | Spawn positions (accounts for robot body) |

The larger spawn padding accounts for:
- Robot body dimensions (~0.5m × 0.3m for quadrupeds)
- Random yaw orientation (diagonal ~0.58m requires ~0.3m clearance)
- Platform edge safety (prevent falling when spawning near stairs)
- Controller startup behavior

These values are defined in `terrain_constants.py` as `PADDING.GOAL_PADDING` and `PADDING.SPAWN_PADDING`.

### Terrain Configuration

The main terrain configuration is defined in [maze_config.py](../source/isaaclab_nav_task/isaaclab_nav_task/navigation/terrains/maze_config.py):

```python
MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),           # 30m × 30m per terrain tile
    border_width=30.0,            # Border around entire grid (not per-tile)
    num_rows=6,                   # 6 difficulty levels
    num_cols=30,                  # 30 terrain variations
    horizontal_scale=0.1,         # 0.1m per height field cell (300×300 cells)
    vertical_scale=0.005,         # Height conversion: value * 0.005 = meters
    curriculum=False,             # Random terrain, not progressive
    difficulty_range=(0.5, 1.0),  # Difficulty sampling range
    sub_terrains={
        "maze": HfMazeTerrainCfg(proportion=0.3, open_probability=0.9,
                                 random_wall_ratio=0.5, add_stairs_to_maze=True),
        "non_maze": HfMazeTerrainCfg(proportion=0.2, non_maze_terrain=True,
                                     random_wall_ratio=1.0),
        "stairs": HfMazeTerrainCfg(proportion=0.3, stairs=True,
                                   randomize_wall=False),
        "pits": HfMazeTerrainCfg(proportion=0.2, dynamic_obstacles=True,
                                 random_wall_ratio=1.0),
    },
)
```

**Global Parameters:**
- Total: **180 terrains** (6 rows × 30 cols), each 30m × 30m
- Height field: **300×300 cells per terrain** (0.1m resolution)
- `curriculum=False`: Random assignment across all 180 terrains

### Curriculum Learning

Terrains are organized in a grid with difficulty varying by row:
- **Rows** (`terrain_levels`): Difficulty levels (0.0 to 1.0)
- **Columns** (`terrain_types`): Different terrain types

```
Difficulty
  1.0  | [Hard Maze] [Random Obs] [Tall Stairs] [Deep Pits] ...
  0.8  | [Med Maze]  [Med Obs]    [Med Stairs]  [Med Pits]  ...
  0.5  | [Easy Maze] [Few Obs]    [Low Stairs]  [Shallow]   ...
  0.0  | [Flat]      [Flat]       [Flat]        [Flat]      ...
       └──────────────────────────────────────────────────────
           maze        non_maze     stairs        pits
```

## Goal and Spawn Sampling

The goal command generator ([RobotNavigationGoalCommand](../source/isaaclab_nav_task/isaaclab_nav_task/navigation/mdp/navigation/goal_commands.py)) handles sampling valid goal and spawn positions from maze terrains using pre-computed boolean masks.

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  RobotNavigationGoalCommand (goal_commands.py:329-766)            │
│  └─► _initialize_position_sampling() (once, lines 415-471)        │
│      └─► Creates PositionSampler with:                             │
│          - heights: Z-lookup for terrain height                    │
│          - valid_mask: Goal positions (5 cells = 0.5m padding)     │
│          - spawn_mask: Spawn positions (6 cells = 0.6m padding)    │
│          - platform_mask: Curriculum learning targets              │
│          - platform_repeat_count: 10 (default, line 74)            │
│          - border_width: 0.0 (sub-terrain default, line 457)       │
│                                                                    │
│  └─► _resample_command(env_ids) (each reset, lines 499-553)       │
│      └─► sample(): Goal from valid_mask (line 519)                 │
│      └─► sample_spawn(): Spawn from spawn_mask (line 521)          │
│      └─► Convert local → world coordinates (lines 524-547)         │
└────────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Pre-computed masks**: `valid_mask` and `spawn_mask` generated during terrain creation
- **Separate padding**: Goals (5 cells = 0.5m) vs spawns (6 cells = 0.6m) for robot body clearance
- **Platform repetition**: Stair platforms repeated 10x in sampling for curriculum learning (goal_commands.py:74, 154)
- **Efficient lookup**: Pre-built position tables enable O(1) random sampling
- **Coordinate conversion**: Handles mesh border offset and centering transform
- **Dynamic border calculation**: Border pixels computed from terrain config (goal_commands.py:107)

### Coordinate System

The terrain mesh uses a coordinate system with:
- **Border pixels**: Dynamically computed as `int(border_width / horizontal_scale) + 1`
  - Sub-terrain configs default to `border_width=0.0` (from `HfTerrainBaseCfg`)
  - With default values: `border_pixels = int(0.0 / 0.1) + 1 = 1`
- **Centering transform**: Mesh is centered at origin by `-terrain_size/2`

```python
# Converting valid_mask index to local coordinates (goal_commands.py:283-284):
local_x = (x_idx + border_pixels) * horizontal_scale - terrain_size/2
local_y = (y_idx + border_pixels) * horizontal_scale - terrain_size/2

# Example: terrain_size=30m, horizontal_scale=0.1m, border_width=0.0
# border_pixels = 1
# valid_mask[0, 0] → local position: (0.1 - 15, 0.1 - 15) = (-14.9, -14.9)
```

The `border_pixels` calculation is done in the `PositionSampler.__init__()` (goal_commands.py:107) to match the `@height_field_to_mesh` decorator behavior.

### Terrain Index Mapping

The terrain index formula depends on the generation mode:

| Mode | Formula | Description |
|------|---------|-------------|
| `curriculum=True` | `level + type * num_rows` | Column-major (iterate rows first) |
| `curriculum=False` | `level * num_cols + type` | Row-major (iterate cols first) |

```python
# In goal_commands.py:
def _get_terrain_indices(self, env_ids):
    terrain = self.env.scene.terrain
    levels = terrain.terrain_levels[env_ids]  # row
    types = terrain.terrain_types[env_ids]    # col

    if terrain_cfg.curriculum:
        return levels + types * num_rows  # column-major
    else:
        return levels * num_cols + types  # row-major
```

### Position Sampling

**PositionSampler** (goal_commands.py:51-290) provides two sampling methods:

```python
class PositionSampler:
    def sample(terrain_indices) -> (x, y, z):
        """Sample GOAL positions from valid_mask.
        Uses platform repetition for curriculum learning.
        Default platform_repeat_count=10 (goal_commands.py:74)"""

    def sample_spawn(terrain_indices) -> (x, y, z):
        """Sample SPAWN positions from spawn_mask.
        Larger padding (6 cells vs 5 cells) for robot body with random orientation."""
```

**Implementation Details:**
- Pre-computes position tables during initialization (goal_commands.py:115-201)
- Platform positions are repeated 10x in goal sampling for curriculum learning
- Uses efficient O(1) random sampling from pre-built position tables
- Falls back to `valid_mask` if `spawn_mask` is not provided (goal_commands.py:99)

**Sampling flow during episode reset:**

```
┌─────────────────────────────────────────────────────────────────┐
│  _resample_command(env_ids)  (goal_commands.py:499-553)         │
├─────────────────────────────────────────────────────────────────┤
│  1. Get terrain indices for each environment (line 516)         │
│     - terrain_levels[env_ids] → row (difficulty)                │
│     - terrain_types[env_ids] → col (terrain type)               │
│     - Apply curriculum/random index formula (lines 473-493)     │
│                                                                  │
│  2. Sample goal position from valid_mask (line 519)             │
│     - Random sample from pre-computed goal position table       │
│     - Platform positions repeated 10x for curriculum weighting  │
│     - Uses GOAL_PADDING = 5 cells = 0.5m                        │
│                                                                  │
│  3. Sample spawn position from spawn_mask (line 521)            │
│     - Random sample from pre-computed spawn position table      │
│     - Larger padding (SPAWN_PADDING = 6 cells = 0.6m)           │
│     - Ensures robot body clearance with random orientation      │
│                                                                  │
│  4. Convert to world coordinates (lines 524-547)                │
│     - Add terrain_origins[level, type] offset                   │
│     - Goal: Add random height offset (0.2-0.8m) for marker      │
│     - Spawn: Add spawn_offset = 0.05m (NOT 0.5m!)               │
│       (robot's default_root_state already has standing height)  │
│                                                                  │
│  5. Update environment origins (lines 540-542)                  │
│     - env.scene.terrain.env_origins[env_ids] = spawn position   │
│     - Robot will be reset to this position                      │
└─────────────────────────────────────────────────────────────────┘
```

### Robot Spawn Height

The spawn height offset accounts for the robot's standing height:

```python
# In _resample_command() (goal_commands.py:537-542):
spawn_offset = 0.05  # Small offset to prevent clipping into terrain

# Note: robot's default_root_state already includes standing height (~0.5m)
terrain.env_origins[env_ids, 0] = terrain_origins[:, 0] + spawn_x
terrain.env_origins[env_ids, 1] = terrain_origins[:, 1] + spawn_y
terrain.env_origins[env_ids, 2] = spawn_z + spawn_offset  # Just 5cm above terrain
```

**Key Change:** The implementation uses only a small 5cm offset because the robot's `default_root_state` configuration already includes the proper standing height (~0.5m). This ensures the robot spawns at the correct height without double-counting the base height.
