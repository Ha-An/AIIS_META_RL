# DRL

Standard PPO training and evaluation code used as the non-meta-learning baseline.

## What is in this folder

- `PPO.py`: PPO implementation
- `train_ppo_example.py`: PPO training entry point
- `eval_ppo.py`: PPO evaluation utilities
- `config.py`: PPO defaults, model paths, and evaluation settings
- `Saved_Model/`: saved PPO checkpoints
- `Tensorboard_logs/`: PPO training logs
- `Eval_results/`: PPO evaluation outputs

## Main training entry point

Run from the repository root:

```bash
python DRL/train_ppo_example.py --mode randomized_task
```

Supported modes:

- `fixed_task`: train on one handcrafted scenario
- `randomized_task`: resample randomized tasks during training

Examples:

```bash
python DRL/train_ppo_example.py --mode fixed_task --task-id 0 --epochs 100
python DRL/train_ppo_example.py --mode randomized_task --epochs 500
```

## Main configuration

Training defaults are defined in `DRL/config.py`:

- `PPO_RUN_CONFIG`
- `META_RL_DEFAULTS`
- `EVAL_MODEL_SPECS`
- `EVAL_RUN_CONFIG`

Important fields:

- `beta`: PPO learning rate
- `clip_eps`: PPO clipping parameter
- `outer_iters`: number of PPO optimization passes per epoch
- `policy_dist`: `categorical` or `gaussian`

## Outputs

- TensorBoard logs: `DRL/Tensorboard_logs/`
- Checkpoints: `DRL/Saved_Model/`
- Evaluation outputs: `DRL/Eval_results/`

## Evaluation

To evaluate saved PPO models:

```bash
python -m DRL.eval_ppo
```

Evaluation settings and model lists are defined in `DRL/config.py`.

## Notes

- This module reuses the same environment family as the meta-RL code for fair comparison.
- PPO training is separate from `AIIS_META/`; there is no inner-loop adaptation in this folder.
- If reproducibility matters, check how seeds are handled in `train_ppo_example.py`, `PPO.py`, and `eval_ppo.py`.
