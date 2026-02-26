# train_ppo_example.py
# -*- coding: utf-8 -*-
"""
Example: Training with PPO (Standard RL without Meta-Learning)
==============================================================
This script demonstrates how to use the PPO algorithm with:
1. Fixed task scenarios (deterministic training)
2. Randomized task scenarios (variable training)

It reuses the same hyperparameters and task configurations as Meta-RL algorithms
for fair comparison.

Usage:
    # Fixed task example
    python train_ppo_example.py --mode fixed_task --task-seed 42
    
    # Randomized task example
    python train_ppo_example.py --mode randomized_task
    
    # Custom learning rate
    python train_ppo_example.py --mode fixed_task --beta 0.001
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRL.PPO import PPO
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Agents.Categorical.Meta_Categorical import MetaCategoricalAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
from envs.promp_env import MetaEnv
import envs.scenarios as scenarios
from envs.config_SimPy import *
from DRL.config import (
    TENSORBOARD_LOG_PATH,
    SAVED_MODEL_PATH_DRL,
    META_RL_DEFAULTS,
    PPO_RUN_CONFIG,
)

# Scenario distribution parameters (same family as Meta-RL)
SCENARIO_DIST_CONFIG = {
    # Keep this minimal; sampling defaults are centralized in envs/scenarios.py
    # so DRL and non-DRL experiments stay aligned.
    "seed": None,
}


def _next_run_dir(base_dir: Path, prefix: str = "Train_") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix):]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = max(existing, default=0) + 1
    return base_dir / f"{prefix}{next_idx}"


def create_fixed_task(task_id: int = 0) -> Dict:
    """
    Create a fixed task configuration in MetaEnv format.
    
    Task variations:
    - Task 0: Low demand, low leadtime (easy)
    - Task 1: Medium demand, medium leadtime
    - Task 2: High demand, high leadtime (hard)
    - Task 3: Extreme demand, variable leadtime
    
    Args:
        task_id: Task identifier (0-3)
    
    Returns:
        Task configuration dict with DEMAND and LEADTIME in MetaEnv format
    """
    import random
    
    task_configs = {
        0: {
            "name": "Low Demand, Low Leadtime",
            "demand": {"Dist_Type": "UNIFORM", "min": 3, "max": 8},
            "leadtime": {"Dist_Type": "UNIFORM", "min": 1, "max": 2},
            "description": "Easy scenario - low demand and short leadtime"
        },
        1: {
            "name": "Medium Demand, Medium Leadtime",
            "demand": {"Dist_Type": "UNIFORM", "min": 10, "max": 15},
            "leadtime": {"Dist_Type": "UNIFORM", "min": 2, "max": 3},
            "description": "Medium scenario - balanced demand and leadtime"
        },
        2: {
            "name": "High Demand, High Leadtime",
            "demand": {"Dist_Type": "UNIFORM", "min": 18, "max": 25},
            "leadtime": {"Dist_Type": "UNIFORM", "min": 3, "max": 5},
            "description": "Hard scenario - high demand and long leadtime"
        },
        3: {
            "name": "Extreme Demand, Variable Leadtime",
            "demand": {"Dist_Type": "UNIFORM", "min": 20, "max": 30},
            "leadtime": {"Dist_Type": "UNIFORM", "min": 1, "max": 5},
            "description": "Extreme scenario - very high demand and uncertain leadtime"
        }
    }
    
    if task_id not in task_configs:
        raise ValueError(f"task_id must be in [0, 1, 2, 3], got {task_id}")
    
    config = task_configs[task_id]
    
    # Create task in MetaEnv format
    from envs.config_SimPy import MAT_COUNT
    task = {
        "DEMAND": config["demand"],
        "LEADTIME": [config["leadtime"] for _ in range(MAT_COUNT)]
    }
    
    # Store metadata for display
    task["_name"] = config["name"]
    task["_description"] = config["description"]
    
    return task


def create_ppo_agent(device: torch.device, num_tasks: int, policy_dist: str = "gaussian") -> Tuple[object, SimpleMLP]:
    """
    Create PPO agent (compatible with meta-RL agents).
    
    Args:
        device: Device to place agent on
    
    Returns:
        (agent, mlp)
    """
    env = MetaEnv()
    
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    policy_dist = policy_dist.lower()
    if policy_dist == "categorical":
        from envs.config_RL import ACTION_SPACE
        num_actions = len(ACTION_SPACE)
        mlp = SimpleMLP(
            input_dim=obs_dim,
            output_dim=act_dim * num_actions,
            hidden_layers=[64, 64, 64]
        )
        agent = MetaCategoricalAgent(
            mlp=mlp,
            num_tasks=num_tasks,
            action_dim=act_dim,
            num_actions=num_actions,
        )
    else:
        mlp = SimpleMLP(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_layers=[64, 64, 64]
        )
        agent = MetaGaussianAgent(
            mlp=mlp,
            num_tasks=num_tasks,
            learn_std=True
        )
    
    agent.to(device)
    return agent, mlp


def run_fixed_task_example(task_id: int = 0,
                           epochs: int = 100,
                           beta: float = 0.0005,
                           clip_eps: float = 0.3,
                           outer_iters: int = 5,
                           policy_dist: str = "gaussian"):
    """
    Train PPO on a fixed task.
    
    Args:
        task_id: Fixed task to use (0-3)
        epochs: Number of training epochs
        beta: Learning rate
        clip_eps: PPO clipping epsilon
        outer_iters: PPO update iterations per epoch
    """
    print("\n" + "="*70)
    print("PPO TRAINING: FIXED TASK MODE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment and agent
    env = MetaEnv()
    env.create_scenarios = lambda: scenarios.create_scenarios(**SCENARIO_DIST_CONFIG)
    agent, _ = create_ppo_agent(device, num_tasks=1, policy_dist=policy_dist)
    baseline = LinearFeatureBaseline()
    
    # Create fixed task
    fixed_task = create_fixed_task(task_id)
    print(f"\n✓ Task: {fixed_task.get('_name', 'Unknown Task')}")
    print(f"  Description: {fixed_task.get('_description', 'N/A')}")
    print(f"  Demand: {fixed_task['DEMAND']}")
    print(f"  Leadtime: {fixed_task['LEADTIME'][0]}")  # All are same in fixed task
    
    # Create log directory
    run_dir = _next_run_dir(TENSORBOARD_LOG_PATH)
    log_dir = run_dir / f"PPO_Fixed_Task_{task_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cost subdirectories for TensorBoard logging
    cost_dirs = [
        log_dir / "cost_Delivery cost",
        log_dir / "cost_Holding cost",
        log_dir / "cost_Order cost",
        log_dir / "cost_Process cost",
        log_dir / "cost_Shortage cost"
    ]
    for cost_dir in cost_dirs:
        cost_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PPO trainer
    ppo = PPO(
        env=env,
        max_path_length=SIM_TIME,
        agent=agent,
        baseline=baseline,
        tensor_log=str(log_dir),
        num_tasks=1,
        rollout_per_task=META_RL_DEFAULTS["rollout_per_task"],
        beta=beta,
        clip_eps=clip_eps,
        outer_iters=outer_iters,
        fixed_task=fixed_task,
        randomize_tasks=False,
        device=device,
        action_log_interval=META_RL_DEFAULTS["action_log_interval"]
    )
    
    # Train
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate (β): {beta}")
    print(f"  Clip epsilon: {clip_eps}")
    print(f"  Outer iterations: {outer_iters}")
    print(f"  Log directory: {log_dir}")
    print(f"{'='*70}\n")
    
    ppo.learn(epochs)
    
    # Save model
    model_run_dir = _next_run_dir(SAVED_MODEL_PATH_DRL)
    model_save_path = model_run_dir / "PPO_Fixed" / "saved_model"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    ppo.save_model(str(model_save_path))
    
    ppo.close()


def run_randomized_task_example(epochs: int = 100,
                                beta: float = 0.0005,
                                clip_eps: float = 0.3,
                                outer_iters: int = 5,
                                policy_dist: str = "gaussian"):
    """
    Train PPO with randomized tasks (new task each epoch).
    
    Args:
        epochs: Number of training epochs
        beta: Learning rate
        clip_eps: PPO clipping epsilon
        outer_iters: PPO update iterations per epoch
    """
    print("\n" + "="*70)
    print("PPO TRAINING: RANDOMIZED TASK MODE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment and agent
    env = MetaEnv()
    env.create_scenarios = lambda: scenarios.create_scenarios(**SCENARIO_DIST_CONFIG)
    agent, _ = create_ppo_agent(device, num_tasks=META_RL_DEFAULTS["num_task"], policy_dist=policy_dist)
    baseline = LinearFeatureBaseline()
    
    print(f"\n✓ Task Randomization: ENABLED")
    print(f"  New random task sampled each epoch")
    print(f"  Task distribution: Same as Meta-RL framework")
    
    # Create log directory
    run_dir = _next_run_dir(TENSORBOARD_LOG_PATH)
    log_dir = run_dir / "PPO_Randomized_Tasks"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PPO trainer (randomize_tasks=True by default)
    ppo = PPO(
        env=env,
        max_path_length=SIM_TIME,
        agent=agent,
        baseline=baseline,
        tensor_log=str(log_dir),
        num_tasks=META_RL_DEFAULTS["num_task"],
        rollout_per_task=META_RL_DEFAULTS["rollout_per_task"],
        beta=beta,
        clip_eps=clip_eps,
        outer_iters=outer_iters,
        randomize_tasks=True,
        device=device,
        action_log_interval=META_RL_DEFAULTS["action_log_interval"]
    )
    
    # Train
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Tasks per epoch: {META_RL_DEFAULTS['num_task']}")
    print(f"  Learning rate (β): {beta}")
    print(f"  Clip epsilon: {clip_eps}")
    print(f"  Outer iterations: {outer_iters}")
    print(f"  Log directory: {log_dir}")
    print(f"{'='*70}\n")
    
    ppo.learn(epochs)
    
    # Save model
    model_run_dir = _next_run_dir(SAVED_MODEL_PATH_DRL)
    model_save_path = model_run_dir / "PPO_Randomized" / "saved_model"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    ppo.save_model(str(model_save_path))
    
    ppo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO Training Example (Non-Meta-Learning)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=PPO_RUN_CONFIG["mode"],
        choices=["fixed_task", "randomized_task"],
        help="Training mode: fixed_task (deterministic) or randomized_task (variable)"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=PPO_RUN_CONFIG["task_id"],
        choices=[0, 1, 2, 3],
        help="Fixed task ID (0: easy, 1: medium, 2: hard, 3: extreme)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=PPO_RUN_CONFIG["epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=PPO_RUN_CONFIG["beta"],
        help="Learning rate (meta-RL compatible)"
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=PPO_RUN_CONFIG["clip_eps"],
        help="PPO clipping epsilon"
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=PPO_RUN_CONFIG["outer_iters"],
        help="PPO optimization iterations per epoch"
    )
    parser.add_argument(
        "--policy-dist",
        type=str,
        default=PPO_RUN_CONFIG["policy_dist"],
        choices=["gaussian", "categorical"],
        help="Policy distribution type"
    )
    
    args = parser.parse_args()
    
    # Run selected training mode
    if args.mode == "fixed_task":
        run_fixed_task_example(
            task_id=args.task_id,
            epochs=args.epochs,
            beta=args.beta,
            clip_eps=args.clip_eps,
            outer_iters=args.outer_iters,
            policy_dist=args.policy_dist
        )
    else:  # randomized_task
        run_randomized_task_example(
            epochs=args.epochs,
            beta=args.beta,
            clip_eps=args.clip_eps,
            outer_iters=args.outer_iters,
            policy_dist=args.policy_dist
        )
