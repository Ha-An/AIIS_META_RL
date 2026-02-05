Few-shot experiment setup
=========================

This script runs a fixed, reproducible few-shot evaluation across multiple
pretrained checkpoints (ProMP, VPG_MAML, PPO) under the same evaluation settings.

What it does
------------
- Uses `Few_shot_learning/experiment_setup.py`
- Forces a common `EVAL_CONFIG` (same tasks/rollouts/steps)
- Switches model paths per experiment
- Runs a single seed (fixed)
- Evaluates four scenario cases:
  - LowDemand_ShortLead
  - LowDemand_LongLead
  - HighDemand_ShortLead
  - HighDemand_LongLead
- Writes TensorBoard logs under `Few_shot_learning/Tensorboard_logs/<run_name>`

How to run
----------
```powershell
python Few_shot_learning/experiment_setup.py
```

Where logs go
------------
- `Few_shot_learning/Tensorboard_logs/ProMP_LowDemand_ShortLead_seed0`
- `Few_shot_learning/Tensorboard_logs/ProMP_HighDemand_LongLead_seed0`
- `Few_shot_learning/Tensorboard_logs/VPG_MAML_LowDemand_ShortLead_seed0`
- `Few_shot_learning/Tensorboard_logs/PPO_HighDemand_ShortLead_seed0`
- etc.

What to edit
------------
In `Few_shot_learning/experiment_setup.py`:
- `EXPERIMENTS`: checkpoint paths and model config overrides
- `SEEDS`: list of random seeds (currently fixed to one)
- `CASE_SCENARIOS`: four demand/leadtime cases
- `EVAL_CONFIG_OVERRIDE`: evaluation settings (fixed across runs)

Notes
-----
- Ensure `MODEL_CONFIG` matches the checkpoint (policy_dist, layers, learn_std).
- PPO checkpoints are loaded as agent-only; ProMP/VPG_MAML use full state_dict.
