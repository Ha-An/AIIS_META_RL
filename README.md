# AIIS_META_RL

Meta-reinforcement learning, standard deep reinforcement learning, and few-shot evaluation for an inventory control simulator built on SimPy.

## Overview

This repository contains three main experiment tracks:

- `AIIS_META/`: meta-RL training, including `ProMP` and `VPG_MAML`
- `DRL/`: PPO training without meta-learning
- `Few_shot_learning/`: few-shot adaptation experiments comparing pretrained `PPO` and `ProMP`
- `Evaluation/`: policy comparison against heuristic baselines under stationary and nonstationary settings

The environment is defined under `envs/` and is shared across all experiment types so that policy comparisons are made on the same simulator and scenario-generation pipeline.

## Repository Layout

```text
AIIS_META_RL/
├─ AIIS_META/            # Meta-RL training code and saved checkpoints
├─ DRL/                  # PPO training and PPO evaluation
├─ Few_shot_learning/    # Few-shot adaptation experiments
├─ Evaluation/           # Policy comparison and paper-ready plots/tables
├─ envs/                 # Shared SimPy inventory environment and scenario generator
├─ requirements.txt
└─ README.md
```

## Environment Setup

The repository is typically used with the Conda environment `aiis_meta_rl` and Python `3.9`.

```bash
conda create -n aiis_meta_rl python=3.9
conda activate aiis_meta_rl
pip install -r requirements.txt
```

On this workspace, the environment path is:

```text
C:\Users\User\anaconda3\envs\aiis_meta_rl\python.exe
```

## Common Entry Points

### 1. Meta-RL Training

Main training entry point:

```bash
python AIIS_META/main.py
```

This script is used to train `ProMP` or `VPG_MAML`, depending on the configuration inside `AIIS_META/main.py`.

Outputs are typically written to:

- TensorBoard logs: `AIIS_META/Tensorboard_logs/`
- Checkpoints: `AIIS_META/Saved_model/`

### 2. PPO Training

Standard PPO training entry point:

```bash
python DRL/train_ppo_example.py --mode randomized_task
```

Other supported modes and hyperparameters are documented in `DRL/README.md`.

Outputs are typically written to:

- TensorBoard logs: `DRL/Tensorboard_logs/`
- Checkpoints: `DRL/Saved_Model/`

### 3. Few-Shot Adaptation Experiments

Few-shot experiment entry point:

```bash
python -m Few_shot_learning.run
```

Useful CLI overrides:

```bash
python -m Few_shot_learning.run --scenario-mode case_randomized --shots 1,2,3
python -m Few_shot_learning.run --smoke
```

Outputs are written to timestamped folders under:

- `Few_shot_learning/runs/`

Each run directory typically contains:

- `raw_results.csv`
- `summary_by_mode_model_shot.csv`
- `summary_by_case_model_shot.csv`
- `progress_latest.txt`
- box plots and final report JSON

### 4. Evaluation Against Heuristics

Policy comparison entry point:

```bash
python -m Evaluation.run_policy_comparison
```

This generates:

- box plots for stationary and nonstationary environments
- significance tables comparing `ProMP` against `PPO` and heuristics
- cost-composition plots

Outputs are written to:

- `Evaluation/outputs/`

## TensorBoard

Typical log directories:

- Meta-RL: `AIIS_META/Tensorboard_logs`
- DRL: `DRL/Tensorboard_logs`

Example:

```bash
tensorboard --logdir AIIS_META/Tensorboard_logs
```

## Scenario Generation

All training and evaluation pipelines share the scenario generator in:

- `envs/scenarios.py`

This is the central place for:

- demand and lead-time distribution sampling
- fixed vs randomized scenario generation
- case-restricted scenario families used in few-shot evaluation

## Notes on Reproducibility

- Meta-RL training uses an explicit global seed in `AIIS_META/main.py`.
- Few-shot and evaluation pipelines derive scenario/support/query seeds deterministically from a master seed.
- DRL evaluation uses explicit seeds.
- DRL training should be treated separately and checked against `DRL/train_ppo_example.py` and `DRL/PPO.py` when strict reproducibility matters.

## Main Output Folders

- `AIIS_META/Saved_model/`
- `DRL/Saved_Model/`
- `Few_shot_learning/runs/`
- `Evaluation/outputs/`

## References

- ProMP / MAML-style meta-RL formulation: see `AIIS_META/`
- PyTorch: https://pytorch.org/
- SimPy: https://simpy.readthedocs.io/

## Contact

Primary contact:

- Yosep Oh (Ha-An)
- yosepoh@hanyang.ac.kr
- Department of Industrial and Management Engineering
- Hanyang University ERICA, South Korea

Research group:

- AIIS Lab (Artificial Intelligence for Industrial Systems)
- https://sites.google.com/view/ha-an/
