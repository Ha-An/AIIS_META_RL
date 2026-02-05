# DRL (Deep Reinforcement Learning) Module

Standard RL algorithms without meta-learning optimization. Designed for comparison with Meta-RL approaches.

## Overview

This module contains standard RL algorithms that reuse the Meta-RL framework's components (environment, sampler, baseline, etc.) for **fair comparison** with meta-learning algorithms.

### Key Features

- **PPO (Proximal Policy Optimization)**: Standard policy gradient algorithm with PPO clipping
- **Fixed Task Mode**: Train on a single deterministic task
- **Randomized Task Mode**: Train with new random tasks sampled each epoch
- **Meta-RL Compatible**: Uses same hyperparameters, environment, and sampling infrastructure
- **TensorBoard Logging**: Built-in visualization support

## Algorithms

### PPO (Proximal Policy Optimization)

A standard policy gradient algorithm that doesn't involve any meta-learning.

**Key differences from ProMP:**
- ❌ No inner-loop task adaptation
- ❌ No outer-loop meta-optimization
- ✅ Simple per-task PPO updates
- ✅ Optional task randomization

**PPO Loss:**
```
L(θ) = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
  where r = π(a|s) / π_old(a|s)  (importance weight)
        A = advantage estimate
        ε = clip_eps (default: 0.3)
```

## Usage

### Fixed Task Training

Train PPO on a single fixed task (deterministic):

```bash
# Task 0: Low demand, low leadtime (easy)
python DRL/train_ppo_example.py --mode fixed_task --task-id 0 --epochs 100

# Task 2: High demand, high leadtime (hard)
python DRL/train_ppo_example.py --mode fixed_task --task-id 2 --epochs 100

# Task 3: Extreme demand, variable leadtime
python DRL/train_ppo_example.py --mode fixed_task --task-id 3 --epochs 100
```

### Randomized Task Training

Train PPO with new random tasks sampled each epoch:

```bash
python DRL/train_ppo_example.py --mode randomized_task --epochs 100
```

### Custom Hyperparameters

All hyperparameters are configurable:

```bash
python DRL/train_ppo_example.py \
    --mode fixed_task \
    --task-id 1 \
    --epochs 150 \
    --beta 0.001 \
    --clip-eps 0.2 \
    --outer-iters 10
```

## Available Tasks

| Task ID | Name | Demand Range | Leadtime Range | Difficulty |
|---------|------|--------------|----------------|-----------|
| 0 | Low Demand, Low Leadtime | (3, 8) | (1, 2) | Easy |
| 1 | Medium Demand, Medium Leadtime | (10, 15) | (2, 3) | Medium |
| 2 | High Demand, High Leadtime | (18, 25) | (3, 5) | Hard |
| 3 | Extreme Demand, Variable Leadtime | (20, 30) | (1, 5) | Extreme |

## Hyperparameter Configuration

All hyperparameters are compatible with Meta-RL experiments for fair comparison:

```python
params = {
    # Algorithm Selection
    "mode": "fixed_task",  # "fixed_task" or "randomized_task"
    
    # Task Configuration
    "task_id": 0,  # For fixed_task mode
    "num_tasks": 1,  # For fixed_task (or 5 for randomized_task)
    "rollout_per_task": 20,  # Trajectories per task per epoch
    "max_path_length": 200,  # Episode length (from SIM_TIME)
    
    # Policy Optimization
    "beta": 0.0005,  # Learning rate (same as meta-RL beta)
    "clip_eps": 0.3,  # PPO clipping epsilon
    "outer_iters": 5,  # PPO update steps per epoch
    
    # Advantage Estimation
    "discount": 0.99,  # Discount factor (γ)
    "gae_lambda": 1.0,  # GAE lambda parameter
    "normalize_adv": True,  # Normalize advantages
    
    # Architecture
    "Layers": [64, 64, 64],  # MLP hidden layers
    "learn_std": True,  # Learn policy std deviation
}
```

## Implementation Details

### Fixed Task Mode
```
Epoch 0: Set fixed task → Collect trajectories → PPO updates
Epoch 1: Use same task → Collect trajectories → PPO updates
...
Epoch N: Use same task → Collect trajectories → PPO updates
```

### Randomized Task Mode
```
Epoch 0: Sample task 1-5 → Collect trajectories → PPO updates
Epoch 1: Sample new tasks 1-5 → Collect trajectories → PPO updates
...
Epoch N: Sample new tasks 1-5 → Collect trajectories → PPO updates
```

## TensorBoard Visualization

View training progress:

```bash
tensorboard --logdir=Tensorboard_logs/PPO_Fixed_Task_0
tensorboard --logdir=Tensorboard_logs/PPO_Randomized_Tasks
```

Logged metrics:
- **Scalar metrics**: AverageReward, AverageReturn, cost values
- **PPO/Loss**: Loss per update iteration
- **Epoch/PPO_Loss**: Mean loss per epoch

## File Structure

```
DRL/
├── __init__.py           # Package initialization
├── PPO.py               # PPO algorithm implementation
├── train_ppo_example.py # Training script with examples
└── README.md            # This file
```

## Comparison with Meta-RL

### ProMP (Meta-RL)
- Inner-loop adaptation per task
- Outer-loop meta-optimization
- Learns task-adaptive initial policies
- Better generalization to new tasks

### PPO (Standard RL)
- No task adaptation
- Simple policy optimization
- Fixed policy for each task
- Good for single-task or curriculum learning

## Examples

### Train on Easy Task for 100 Epochs
```python
python DRL/train_ppo_example.py --mode fixed_task --task-id 0 --epochs 100
```

### Train with Random Tasks for 50 Epochs
```python
python DRL/train_ppo_example.py --mode randomized_task --epochs 50 --beta 0.001
```

### High Learning Rate, Many Updates
```python
python DRL/train_ppo_example.py --mode fixed_task --task-id 2 --beta 0.002 --outer-iters 20
```

## Notes

- **Compatibility**: All components (environment, agent, sampler, baseline) are identical to Meta-RL module
- **Hyperparameters**: `beta` (learning rate), `clip_eps`, `outer_iters` use same values as Meta-RL `alpha`, `clip_eps`, `outer_iters`
- **Task Distribution**: Fixed and randomized tasks use the same distribution as Meta-RL experiments
- **Device**: Automatically detects and uses GPU if available

## See Also

- [Meta-RL README](../AIIS_META/README.md)
- [ProMP Algorithm](../AIIS_META/Algos/MAML/promp.py)
- [VPG_MAML Algorithm](../AIIS_META/Algos/MAML/maml.py)
