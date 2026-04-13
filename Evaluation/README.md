# Evaluation

Policy-comparison experiments for heuristic baselines and pretrained RL policies.

## What is in this folder

- `run_policy_comparison.py`: main evaluation entry point
- `config.py`: model paths, seeds, scenario counts, and evaluation settings
- `outputs/`: timestamped evaluation outputs

## Compared methods

The evaluation compares:

- heuristic baselines: `R1`, `R3`, `R5`
- pretrained RL policies: `PPO`, `ProMP`

## Main entry point

Run from the repository root:

```bash
python -m Evaluation.run_policy_comparison
```

Examples:

```bash
python -m Evaluation.run_policy_comparison --days 200 --stationary-num-scenarios-per-seed 100 --nonstationary-num-sequences-per-seed 100
python -m Evaluation.run_policy_comparison --seeds 2026 2027 2028 2029 2030
python -m Evaluation.run_policy_comparison --scenario-mode fixed
```

## Main configuration

Key settings are defined in `Evaluation/config.py`:

- `PRETRAINED_PROMP_MODEL_PATH`
- `PRETRAINED_PPO_MODEL_PATH`
- `EVALUATION_SEEDS`
- `SCENARIO_MODE`
- `STATIONARY_NUM_SCENARIOS_PER_SEED`
- `NONSTATIONARY_NUM_SEQUENCES_PER_SEED`
- `NONSTATIONARY_SEGMENTS`
- `SCENARIO_SAMPLING_OVERRIDES`
- `METHODS`

## Outputs

Each run is written to:

- `Evaluation/outputs/run_YYYYMMDD_HHMMSS/`

Typical outputs include:

- `episode_results.csv`
- `summary_by_environment_method.csv`
- `stationary_boxplot_total_cost.png`
- `nonstationary_boxplot_total_cost.png`
- `significance_promp_vs_others_by_environment.csv`
- `significance_promp_vs_others_by_scenario.csv`
- `significance_promp_vs_others_heatmap.png`
- `stationary_cost_composition.png`
- `nonstationary_cost_composition.png`

## Notes

- Paired evaluations are used so that all methods see identical scenario realizations.
- The evaluation pipeline can generate box plots, significance tables, and cost-composition figures.
- Significance testing is implemented for `ProMP` against the comparator methods.
