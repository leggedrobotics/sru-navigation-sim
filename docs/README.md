# IsaacLab Navigation Extension - SRU Project

[![Paper](https://img.shields.io/badge/IJRR-2025-blue)](https://journals.sagepub.com/home/ijr)
[![Website](https://img.shields.io/badge/Project-Website-green)](https://michaelfyang.github.io/sru-project-website/)

> **📌 Important Note**: This repository contains the **IsaacLab task extension** for the SRU project, providing diverse navigation environments with dynamic obstacle configurations and terrain variations. This repository does **not** include the `rsl_rl` learning module (network architectures, PPO/MDPO training algorithms). See the [project website](https://michaelfyang.github.io/sru-project-website/) for the complete navigation system.

## Overview

A standalone, self-contained IsaacLab task extension for visual navigation in Isaac Lab v2.1.1 (Isaac Sim 4.5). This repository provides:

- **Environment**: Diverse navigation environments in IsaacLab with dynamic obstacle configurations and terrain variations
- **Task Definition**: Hierarchical control architecture interface for visual navigation with reinforcement learning
- **Simulation**: High-fidelity physics simulation with realistic depth sensor noise

**Note**: This repository focuses on the simulation environment and task definition. The RL training infrastructure (neural network architectures, PPO/MDPO algorithms) is provided by the separate `rsl_rl` learning module.

This extension implements a hierarchical control architecture for visual navigation:
- **High-level policy**: Learns to output SE2 velocity commands (vx, vy, omega) at 5Hz
- **Low-level policy**: Pre-trained locomotion policy that converts velocity commands to joint actions at 50Hz

The extension is fully self-contained with all necessary robot models, materials, and pre-trained locomotion policies included.

### What's Included

- ✅ IsaacLab task extension for visual navigation environments
- ✅ Maze terrain generation with curriculum learning
- ✅ Self-contained assets: Robot models (USD), locomotion policies, depth encoders
- ✅ Multiple robot platforms: B2W (bipedal wheeled) and AoW-D (Anymal on Wheels)
- ✅ Observation definitions: Depth images, proprioception, goal commands
- ✅ Reward functions: Goal reaching, action smoothing, movement penalties
- ✅ Hierarchical action interface: SE2 velocity commands to low-level controllers
- ✅ Domain randomization: Camera pose, action scaling, low-pass filters, sensor delays
- ✅ Training scripts compatible with RSL-RL (PPO/MDPO algorithms)

### What's NOT Included

- ❌ `rsl_rl` learning module (network architectures, PPO/MDPO training algorithms)
- ❌ Neural network structures for high-level navigation policy
- ❌ On-policy RL training algorithms (PPO/MDPO implementations)

**Note**: The `rsl_rl` package must be installed separately to train navigation policies. See the Installation section below.

### Related Projects

- [sru-pytorch-spatial-learning](https://github.com/michaelfyang/sru-pytorch-spatial-learning) - Core SRU architecture
- [SRU Project Website](https://michaelfyang.github.io/sru-project-website/) - Complete navigation system

## Features

- **Visual navigation** using depth cameras with realistic noise simulation
- **Maze terrain generation** with curriculum learning
- **Self-contained assets**: All robot models and locomotion policies included
- **Multiple robot platforms**:
  - **B2W**: Bipedal wheeled robot (with ZedX camera)
  - **AoW-D**: Anymal on Wheels (with ZedX camera)
- **Asymmetric actor-critic** with privileged critic observations
- **Curriculum learning** for terrain difficulty progression
- **Multiple algorithms**: MDPO and PPO support via RSL-RL
- **Domain randomization**: Camera pose, action scaling, low-pass filters, sensor delays

## Installation

First, install the extension in development mode:

```bash
cd source/isaaclab_nav_task
pip install -e .
```

## Available Tasks

### B2W
| Task ID | Description |
|---------|-------------|
| `Isaac-Nav-MDPO-B2W-v0` | MDPO training |
| `Isaac-Nav-PPO-B2W-v0` | PPO training |
| `Isaac-Nav-MDPO-B2W-Play-v0` | MDPO playback |
| `Isaac-Nav-PPO-B2W-Play-v0` | PPO playback |
| `Isaac-Nav-MDPO-B2W-Dev-v0` | MDPO development |
| `Isaac-Nav-PPO-B2W-Dev-v0` | PPO development |

### AoW-D
| Task ID | Description |
|---------|-------------|
| `Isaac-Nav-MDPO-AoW-D-v0` | MDPO training |
| `Isaac-Nav-PPO-AoW-D-v0` | PPO training |
| `Isaac-Nav-MDPO-AoW-D-Play-v0` | MDPO playback |
| `Isaac-Nav-PPO-AoW-D-Play-v0` | PPO playback |
| `Isaac-Nav-MDPO-AoW-D-Dev-v0` | MDPO development |
| `Isaac-Nav-PPO-AoW-D-Dev-v0` | PPO development |

## Training

### Using the standalone training script

```bash
# Train B2W with PPO
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 --num_envs 4096 --headless

# Train AoW-D with PPO
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-AoW-D-v0 --num_envs 4096 --headless

# Train with custom wandb run name
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-MDPO-B2W-v0 --num_envs 4096 --headless \
    --run_name "experiment_v1_with_curriculum"

# Train with multiple custom parameters
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 --num_envs 2048 --headless \
    --run_name "large_training_run" --seed 42 --max_iterations 20000
```

### Development/Testing (smaller config with tensorboard)

The `-Dev-v0` variants use tensorboard logging instead of wandb and have reduced iterations (300 vs 15000) for quick testing:

```bash
# Quick test with small environment count
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-Dev-v0 --num_envs 32 --headless
```

### Using the standard RSL-RL workflow

```bash
# Train with RSL-RL
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Nav-MDPO-B2W-v0 --num_envs 4096
```

## Playing Trained Policies

```bash
# Play using standalone script
./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-MDPO-B2W-Play-v0 --num_envs 16

# Play with specific checkpoint
./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-MDPO-B2W-Play-v0 \
    --checkpoint /path/to/model.pt
```

## Architecture

```
isaaclab_nav_task/
├── config/
│   └── extension.toml            # Extension metadata
├── docs/
│   └── README.md                 # This file
├── scripts/
│   ├── train.py                  # Training script
│   └── play.py                   # Playback script
├── setup.py                      # Installation script
├── pyproject.toml                # Build configuration
└── isaaclab_nav_task/
    ├── __init__.py               # Extension entry point
    └── navigation/
        ├── __init__.py
        ├── navigation_env_cfg.py         # Base environment config
        ├── assets/                       # Robot configurations and data
        │   ├── __init__.py
        │   ├── b2w.py                    # B2W robot config
        │   ├── aow_d.py                  # AoW-D robot config
        │   └── data/                     # Self-contained asset directory
        │       ├── Robots/               # Robot USD models and materials
        │       │   └── AoW-D/            # AoW-D robot assets
        │       │       ├── aow_d.usd     # Robot USD model
        │       │       └── Props/        # Materials and textures
        │       └── Policies/             # Pre-trained models
        │           ├── depth_encoder/    # VAE depth encoders
        │           │   └── vae_pretrain_new.pth  (ZedX)
        │           └── locomotion/       # Low-level locomotion policies
        │               ├── aow_d/        # policy_blind_3_1.pt (1.7 MB)
        │               └── b2w/          # policy_b2w_new_2.pt (2.0 MB)
        ├── config/
        │   ├── rl_cfg.py                 # Base RL configurations
        │   ├── b2w/
        │   │   ├── __init__.py           # Task registration
        │   │   ├── navigation_env_cfg.py
        │   │   └── agents/
        │   │       └── rsl_rl_cfg.py
        │   └── aow_d/
        │       ├── __init__.py
        │       ├── navigation_env_cfg.py
        │       └── agents/
        │           └── rsl_rl_cfg.py
        ├── mdp/
        │   ├── observations.py       # Observation functions (13 functions)
        │   ├── rewards.py            # Reward functions (5 functions)
        │   ├── terminations.py       # Termination conditions (4 functions)
        │   ├── curriculums.py        # Curriculum terms (1 function)
        │   ├── events.py             # Domain randomization events (5 functions)
        │   ├── depth_utils/          # Depth processing utilities
        │   │   ├── __init__.py
        │   │   ├── camera_config.py      # Camera configurations (ZedX)
        │   │   └── depth_noise_encoder.py # VAE-based depth encoder
        │   └── navigation/
        │       ├── goal_commands.py
        │       ├── goal_commands_cfg.py
        │       └── actions/
        │           ├── __init__.py
        │           ├── navigation_se2_actions.py
        │           └── navigation_se2_actions_cfg.py
        └── terrains/                # Custom terrain generators
            ├── __init__.py
            ├── hf_terrains_maze.py      # Maze terrain generation
            ├── hf_terrains_maze_cfg.py  # Maze terrain configs
            ├── maze_config.py           # Maze parameters
            └── patches.py               # TerrainImporter patches
```

## Compatibility

- **Isaac Lab**: v2.1.1
- **Isaac Sim**: 4.5.0
- **Python**: 3.10
- **PyTorch**: >= 2.5.1

## Self-Contained Assets

The extension includes all necessary assets and does not depend on external asset repositories:

### Robot Models (`assets/data/Robots/`)
- **AoW-D**: Complete USD model with materials and textures
  - Used when AoW-D robots are not available in the base `isaaclab_assets`
  - Includes all necessary Props and material textures (11 baked textures)

### Pre-trained Policies (`assets/data/Policies/`)

**Depth Encoders** (`depth_encoder/`):
- `vae_pretrain_new.pth`: ZedX camera encoder for B2W and AoW-D
- VAE architecture with RegNet backbone + Feature Pyramid Network

**Locomotion Policies** (`locomotion/`):
- `aow_d/policy_blind_3_1.pt` (1.7 MB): AoW-D wheeled locomotion
- `b2w/policy_b2w_new_2.pt` (2.0 MB): B2W bipedal wheeled locomotion

All locomotion policies are pre-trained and loaded by the hierarchical action controller.

## Key Components

### Navigation Environment (`navigation_env_cfg.py`)
- Defines the scene with terrain, robot, and sensors
- Configures observation groups for policy and critic
- Sets up reward terms for goal reaching and movement penalties
- Configures curriculum for terrain difficulty

### MDP Components (`mdp/`)

**Cleaned and optimized** - removed unused functions to improve maintainability:

- **observations.py** (13 functions): Depth image processing, proprioception, goal direction, delay buffers
- **rewards.py** (5 functions): Goal reaching, action smoothing, movement penalties
- **terminations.py** (4 functions): Timeout, collision detection, angle limits, goal reaching
- **curriculums.py** (1 function): Backward movement penalty scheduling
- **events.py** (5 functions): Camera randomization, action scaling, delay buffer management

### Navigation Actions (`mdp/navigation/actions/`)
- Hierarchical action space with SE2 velocity commands
- Integration with pre-trained low-level locomotion policies

### Terrain Generation (`terrains/`)

The extension includes custom maze terrain generators built on Isaac Lab's terrain generation system.

#### Architecture Overview

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

#### Key Files

| File | Purpose |
|------|---------|
| `hf_terrains_maze.py` | Terrain generation with explicit valid_mask/spawn_mask |
| `hf_terrains_maze_cfg.py` | Configuration dataclass with mask storage attributes |
| `terrain_constants.py` | PADDING, HEIGHTS, THRESHOLDS constants |
| `patches.py` | Monkey-patches for TerrainGenerator/TerrainImporter |
| `maze_config.py` | MAZE_TERRAIN_CFG with sub-terrain configurations |

#### Mesh Optimization (`patches.py`)

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

#### Terrain Data Flow

The terrain system uses **explicit boolean masks** instead of height-based classification:

```python
# During terrain generation (hf_terrains_maze.py)
terrain = TerrainData.create(width, height)

# Mark obstacles as invalid
terrain.set_obstacle(x_start, x_end, y_start, y_end, wall_height)

# Apply padding and create masks
terrain.apply_padding(PADDING.GOAL_PADDING)   # 5 cells = 0.5m for goals
spawn_mask = terrain.create_spawn_mask(PADDING.SPAWN_PADDING)  # 6 cells = 0.6m for spawns

# Store on config for patches to pick up
cfg.height_field_visual = heights      # For Z-lookup
cfg.height_field_valid_mask = valid_mask   # For goal sampling
cfg.height_field_spawn_mask = spawn_mask   # For spawn sampling
cfg.height_field_platform_mask = platform_mask  # For curriculum
```

#### Maze Terrain Types (`hf_terrains_maze.py`)

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

#### Safety Padding

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

#### Terrain Configuration (`maze_config.py`)

```python
MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),           # 30m × 30m per terrain
    num_rows=6,                   # 6 difficulty levels
    num_cols=30,                  # 30 terrain types
    horizontal_scale=0.1,         # 0.1m per height field cell
    vertical_scale=0.005,         # Height scaling factor
    curriculum=False,             # Random terrain assignment
    sub_terrains={
        "maze": HfMazeTerrainCfg(proportion=0.3, ...),
        "non_maze": HfMazeTerrainCfg(proportion=0.2, non_maze_terrain=True, ...),
        "stairs": HfMazeTerrainCfg(proportion=0.3, stairs=True, ...),
        "pits": HfMazeTerrainCfg(proportion=0.2, dynamic_obstacles=True, ...),
    },
)
```

Key parameters:
- **grid_size**: Maze grid cells (default: 15×15)
- **cell_size**: Size per maze cell (default: 2.0m)
- **wall_height**: Obstacle height (default: 1.5m)
- **open_probability**: Controls maze complexity (0.9 = more open)
- **random_wall_ratio**: Mix of random vs standard walls

#### Curriculum Learning

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

### Goal Commands (`mdp/navigation/goal_commands.py`)

The goal command generator (`RobotNavigationGoalCommand`) handles sampling valid goal and spawn positions from maze terrains using pre-computed boolean masks.

#### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  RobotNavigationGoalCommand                                        │
│  └─► _initialize_position_sampling() (once)                        │
│      └─► Creates PositionSampler with:                             │
│          - heights: Z-lookup for terrain height                    │
│          - valid_mask: Goal positions (0.5m padding)               │
│          - spawn_mask: Spawn positions (0.6m padding)              │
│          - platform_mask: Curriculum learning targets              │
│                                                                    │
│  └─► _resample_command(env_ids) (each reset)                       │
│      └─► sample(): Goal from valid_mask                            │
│      └─► sample_spawn(): Spawn from spawn_mask                     │
│      └─► Convert local → world coordinates                         │
└────────────────────────────────────────────────────────────────────┘
```

#### Key Features

- **Pre-computed masks**: `valid_mask` and `spawn_mask` generated during terrain creation
- **Separate padding**: Goals (0.3m) vs spawns (0.8m) for robot body clearance
- **Platform repetition**: Stair platforms repeated in sampling for curriculum learning
- **Efficient lookup**: Pre-built position tables enable O(1) random sampling
- **Coordinate conversion**: Handles mesh border offset and centering transform

#### Coordinate System

The terrain mesh uses a coordinate system with:
- **Border pixels**: 1 extra pixel on each edge from `@height_field_to_mesh` decorator
- **Centering transform**: Mesh is centered at origin by `-terrain_size/2`

```python
# Converting valid_mask index to local coordinates:
local_x = (x_idx + border_pixels) * horizontal_scale - terrain_size/2
local_y = (y_idx + border_pixels) * horizontal_scale - terrain_size/2

# Example: terrain_size=30m, horizontal_scale=0.1m
# valid_mask[0, 0] → local position: (0.1 - 15, 0.1 - 15) = (-14.9, -14.9)
```

#### Terrain Index Mapping

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

#### Position Sampling

**PositionSampler** provides two sampling methods:

```python
class PositionSampler:
    def sample(terrain_indices) -> (x, y, z):
        """Sample GOAL positions from valid_mask.
        Uses platform repetition for curriculum learning."""

    def sample_spawn(terrain_indices) -> (x, y, z):
        """Sample SPAWN positions from spawn_mask.
        Larger padding for robot body with random orientation."""
```

**Sampling flow during episode reset:**

```
┌─────────────────────────────────────────────────────────────────┐
│  _resample_command(env_ids)                                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Get terrain indices for each environment                    │
│     - terrain_levels[env_ids] → row (difficulty)                │
│     - terrain_types[env_ids] → col (terrain type)               │
│     - Apply curriculum/random index formula                     │
│                                                                  │
│  2. Sample goal position (from valid_mask)                      │
│     - Random sample from pre-computed goal position table       │
│     - Platform positions repeated for curriculum weighting      │
│                                                                  │
│  3. Sample spawn position (from spawn_mask)                     │
│     - Random sample from pre-computed spawn position table      │
│     - Larger padding ensures robot body clearance               │
│                                                                  │
│  4. Convert to world coordinates                                │
│     - Add terrain_origins[level, type] offset                   │
│     - Goal: Add random height offset (0.2-0.8m) for marker      │
│     - Spawn: Add robot base height (0.5m) for standing          │
│                                                                  │
│  5. Update environment origins                                  │
│     - env.scene.terrain.env_origins[env_ids] = spawn position   │
│     - Robot will be reset to this position                      │
└─────────────────────────────────────────────────────────────────┘
```

#### Robot Spawn Height

The spawn height offset accounts for the robot's standing height:

```python
# In _resample_command():
robot_base_height = 0.5  # Quadruped standing height

terrain.env_origins[env_ids, 2] = spawn_z + robot_base_height
```

This ensures the robot spawns at proper standing height above the terrain surface, rather than with its base at ground level.

### Depth Processing (`mdp/depth_utils/`)
- **DepthNoise**: Simulates realistic stereo camera noise using disparity-based filtering
- **DepthNoiseEncoder**: VAE-based depth encoder using RegNet backbone with Feature Pyramid Network
- **Camera Configurations**: Pre-defined configs for different camera types:

| Camera | Robots | Resolution | Depth Range | Encoder |
|--------|--------|------------|-------------|---------|
| ZedX | B2W, AoW-D | 64x40 | 0.25-10.0m | `vae_pretrain_new.pth` |

### Custom Robot Assets (`assets/`)

Robot configuration modules define robot-specific parameters:

**B2W** (`b2w.py`):
- Actuator configurations (position/velocity control)
- Initial joint states
- USD asset path (from base `isaaclab_assets`)

**AoW-D** (`aow_d.py`):
- Actuator configurations for wheeled quadruped
- Initial joint states
- USD asset path (from local `assets/data/Robots/AoW-D/`)
- Uses local robot model when not available in base assets

Both configurations integrate seamlessly with the hierarchical navigation controller and pre-trained locomotion policies.

## Docker and Cluster Setup

### Docker Modifications

The Dockerfile includes:
1. **Custom RSL-RL**: Installs custom `rsl_rl` package in editable mode
2. **Git safe directories**: Prevents ownership errors in containers

### Quick Start Workflow

```bash
# 1. Build Docker image
./docker/container.sh start --suffix nav

# 2. Push to cluster (converts to Singularity automatically)
./docker/cluster/cluster_interface.sh push base-nav

# 3. Submit training job
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-PPO-B2W-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--headless"

# 4. Monitor job
squeue -u $USER
```

### Configuration

**Step 1**: Create `.env.base-nav` profile in `docker/` directory:
```bash
cp docker/.env.base docker/.env.base-nav
```

**Step 2**: Configure `docker/cluster/.env.cluster` before deployment:
- Set `CLUSTER_PYTHON_EXECUTABLE=source/isaaclab_nav_task/scripts/train.py`
- Add cluster credentials and paths

**Step 3**: Add cluster-specific module loads in `docker/cluster/submit_job_slurm.sh`:
```bash
module load eth_proxy  # Required for network access on ETH cluster
```

See the [IsaacLab cluster guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html#cluster-guide) for details.

### Training Examples

```bash
# B2W with MDPO training (10k iterations)
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-MDPO-B2W-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--headless"

# B2W with custom run name
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-MDPO-B2W-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--run_name experiment_v1_b2w" \
    "--headless"

# AoW-D with MDPO training (10k iterations)
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-MDPO-AoW-D-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--headless"

# Quick dev test with PPO training (300 iters, tensorboard)
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-PPO-B2W-Dev-v0" \
    "--num_envs 32" \
    "--headless"
```

### Troubleshooting

**Git ownership errors**: Rebuild Docker image (includes fix) or run in container:
```bash
git config --global --add safe.directory '*'
```

**Memory issues**: Reduce `--num_envs` or increase `#SBATCH --mem-per-cpu`

## License

MIT License - See [LICENSE](../LICENSE) file for details

Copyright (c) 2025 Fan Yang, Per Frivik, Robotic Systems Lab, ETH Zurich

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{yang2025sru,
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  title = {Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  doi = {10.1177/02783649251401926},
  url = {https://doi.org/10.1177/02783649251401926}
}
```

## Contact

**Authors**:
- Fan Yang (fanyang1@ethz.ch)
- Per Frivik (pfrivik@ethz.ch)

**Affiliation**: Robotic Systems Lab, ETH Zurich
