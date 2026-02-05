# AIIS_META_RL

## Overview

This project implements ProMP (Proximal Meta-Policy Search) and related meta-RL algorithms using modern libraries (PyTorch, TensorBoard). It uses a SimPy-based inventory management simulator as the environment.

### Key Features
- Meta-Reinforcement Learning: ProMP and MAML-style algorithms
- Modern Library Support: PyTorch 2.x and TensorBoard
- Inventory Management Simulation: SimPy-based inventory management environment

---

## Quick Start

### 1) Environment Setup
```bash
conda create -n aiis_meta_rl python=3.9
conda activate aiis_meta_rl
pip install -r requirements.txt
```

### 2) Meta-RL Training (ProMP)
```bash
python AIIS_META/main.py
```

### 3) PPO Training (DRL)
```bash
python DRL/train_ppo_example.py --mode fixed_task --task-id 0 --epochs 100
```

### 4) Few-shot Evaluation
```bash
python Few_shot_learning/eval_few_shot.py
```

### 5) Batch Few-shot Experiments (multi-model)
```bash
python Few_shot_learning/experiment_setup.py
```

---

## TensorBoard Logs

- Meta-RL (ProMP/MAML): `AIIS_META/Tensorboard_logs`
- DRL (PPO): `DRL/Tensorboard_logs`
- Few-shot runs: `Few_shot_learning/Tensorboard_logs`

Example:
```bash
tensorboard --logdir=AIIS_META/Tensorboard_logs
```

---

## Project Structure

```
AIIS_META_RL/
  AIIS_META/main.py                # Meta-RL training entrypoint
  DRL/train_ppo_example.py         # PPO training entrypoint
  Few_shot_learning/               # Few-shot evaluation
  requirements.txt                 # Python dependencies
  AIIS_META/                       # Meta-RL algorithms and agents
    Agents/                        # Policy networks (MLP, Gaussian, Categorical)
    Algos/                         # ProMP, MAML
    Baselines/                     # Advantage baselines
    Sampler/                       # Samplers and processors
    Utils/                         # Utilities
  envs/                            # Inventory management environment
    config_SimPy.py                # SimPy configuration
    config_RL.py                   # RL config
    environment.py                 # Environment definitions
    promp_env.py                   # MetaEnv
    scenarios.py                   # Scenario generation
  Tensorboard_logs/                # Legacy/global logs (not used by current entrypoints)
  Saved_Model/                     # Trained model storage
  README.md                        # This document
```

---

## References

- ProMP: https://promp.readthedocs.io/en/latest/index.html
- Environment: https://github.com/Ha-An/SimPy_IMS
- PyTorch: https://pytorch.org/
- SimPy: https://simpy.readthedocs.io/

---

## License

This project is based on ProMP and updates it with modern deep learning libraries.

---

## Contact

Primary Contact
- Name: Yosep Oh (Ha-An)
- Email: yosepoh@hanyang.ac.kr
- Department: Department of Industrial and Management Engineering
- Institution: Hanyang University ERICA, South Korea

Research Group
- Lab: AIIS Lab (Artificial Intelligence for Industrial Systems)
- Website: https://sites.google.com/view/ha-an/
