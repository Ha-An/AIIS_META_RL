# Few_shot_learning

This folder contains a randomized few-shot evaluator for pretrained `PPO` and `ProMP` models.

What this code does:
- rebuilds the policy architecture and loads the pretrained checkpoints from `config.py`
- generates randomized stationary tasks from `envs/scenarios.py`
- generates randomized nonstationary 3-segment tasks from `envs/scenarios.py`
- applies the exact same generated task to both models for fair comparison
- runs K-shot adaptation using support trajectories
- measures query total cost after adaptation
- saves stationary and nonstationary total-cost boxplots for every run

Task modes:
- `stationary`: one generated scenario distribution is used for the full 200-day episode
- `nonstationary`: three generated scenario distributions are used over days `1-100`, `101-150`, `151-200`

Few-shot protocol:
- `K` is the number of support trajectories sampled for adaptation
- `QUERY_ROLLOUT_PER_TASK` is the number of query trajectories used to estimate performance on the same generated task
- `EVAL_ADAPT_UPDATES` is the number of inner updates applied while reusing the same support batch

Run:
```bash
python -m Few_shot_learning.run
```

Quick smoke test:
```bash
python -m Few_shot_learning.run --smoke
```

Useful overrides:
```bash
python -m Few_shot_learning.run --shots 0,1,2,3 --stationary-scenarios 10 --nonstationary-sequences 10 --query-rollouts 3
```

Outputs are stored under:
- `Few_shot_learning/runs/run_YYYYMMDD_HHMMSS/`
