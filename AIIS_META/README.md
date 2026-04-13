# AIIS_META

Meta-reinforcement learning training code for the inventory-control environment.

## What is in this folder

- `main.py`: main training entry point
- `Algos/`: meta-RL algorithms such as `ProMP` and `VPG_MAML`
- `Agents/`: policy networks and distributions
- `Baselines/`: baseline estimators
- `Sampler/`: trajectory sampling and sample processing
- `Utils/`: utility modules
- `Saved_model/`: trained meta-RL checkpoints
- `Tensorboard_logs/`: TensorBoard logs for meta-RL runs

## Main entry point

Run meta-RL training from the repository root:

```bash
python AIIS_META/main.py
```

The algorithm and training settings are selected inside `AIIS_META/main.py`.

## Current training flow

The training script:

1. fixes the global training seed
2. builds the shared SimPy-based environment
3. constructs the policy network and meta-RL adapter
4. trains the selected algorithm
5. writes TensorBoard logs and checkpoints

## Outputs

- TensorBoard logs: `AIIS_META/Tensorboard_logs/`
- Saved checkpoints: `AIIS_META/Saved_model/`

The training script can save:

- the final model
- the best few-shot checkpoint
- the best checkpoint by total cost within the final epoch window

## Notes

- `ProMP` and `VPG_MAML` share the same environment and much of the same sampling infrastructure.
- Few-shot validation during training is configured inside `AIIS_META/main.py`.
- If you are comparing against PPO, use the separate `DRL/` module rather than this folder.
