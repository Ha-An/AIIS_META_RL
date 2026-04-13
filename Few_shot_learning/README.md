# Few_shot_learning

Few-shot adaptation experiments for pretrained `PPO` and `ProMP` policies.

## What is in this folder

- `run.py`: command-line entry point
- `core.py`: experiment logic
- `config.py`: checkpoints, scenario modes, shot settings, and runtime defaults
- `runs/`: timestamped experiment outputs
- `Tensorboard_logs/`: auxiliary logs if generated

## Experiment modes

Supported scenario modes:

- `randomized`
  - can evaluate `stationary` and `nonstationary`
- `case_randomized`
  - evaluates stationary scenarios only
  - uses four narrowed scenario families:
    - high demand / long lead time
    - high demand / short lead time
    - low demand / long lead time
    - low demand / short lead time

## Few-shot protocol

- `k`-shot means `k` support trajectories are collected from the target scenario
- the same support set is reused for a fixed number of inner updates
- performance is measured on a separate query set
- the main metric is total cost

The main defaults are defined in `Few_shot_learning/config.py`:

- `SCENARIO_MODE`
- `ENVIRONMENT_MODES`
- `EVAL_SHOTS`
- `QUERY_ROLLOUT_PER_TASK`
- `EVAL_ADAPT_UPDATES`
- `CASE_RANDOMIZED_SCENARIO_COUNT_PER_CASE`

## Run

From the repository root:

```bash
python -m Few_shot_learning.run
```

Examples:

```bash
python -m Few_shot_learning.run --smoke
python -m Few_shot_learning.run --scenario-mode case_randomized --shots 1,2,3
python -m Few_shot_learning.run --scenario-mode randomized --modes stationary,nonstationary
```

## Outputs

Each run is written to:

- `Few_shot_learning/runs/run_YYYYMMDD_HHMMSS/`

Typical outputs:

- `raw_results.csv`
- `summary_by_mode_model_shot.csv`
- `summary_by_task_model_shot.csv`
- `summary_by_case_model_shot.csv`
- `progress_latest.txt`
- `final_report.json`
- box plots for overall results and case-wise results

## Notes

- PPO and ProMP are evaluated on identical scenario realizations for fair comparison.
- Scenario, support, and query seeds are deterministically derived from a master seed.
- If `case_randomized` is selected, only `stationary` evaluation is supported.
