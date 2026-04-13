# envs

Shared inventory simulation environment used by meta-RL, PPO, few-shot evaluation, and policy-comparison experiments.

## What is in this folder

- `scenarios.py`: randomized scenario generation
- `promp_env.py`: main RL environment wrapper
- `environment.py`: simulator components
- `config_SimPy.py`: SimPy-side environment constants
- `config_RL.py`: RL-side action and state settings
- `simpy_diagnostics.py`: diagnostic runner for environment sanity checks
- `reorder_point_experiment.py`: heuristic-policy utilities

## Main role

This folder provides the shared simulator and scenario-generation pipeline for:

- `AIIS_META/`
- `DRL/`
- `Few_shot_learning/`
- `Evaluation/`

Using one shared environment definition keeps comparisons consistent across all experiments.

## Scenario generation

Scenario pools are generated through:

```bash
python -c "import envs.scenarios as s; print(s.create_scenarios(num_scenarios=3, seed=2026))"
```

In practice, the higher-level experiment modules call `envs/scenarios.py` directly.

## Diagnostics

To inspect simulator behavior:

```bash
python -m envs.simpy_diagnostics --days 60 --lot-size 5 --policy fixed_lot --cust-order-cycle 7
```

Heuristic standalone mode:

```bash
python -m envs.simpy_diagnostics --policy heuristic_rop --reorder-point 1 --days 200 --cust-order-cycle 7
```

Typical outputs:

- `envs/diagnostics_outputs/simpy_diagnostics.png`
- `envs/diagnostics_outputs/simpy_daily_metrics.csv`

## Notes

- The environment is SimPy-based.
- Cost components include holding, process, delivery, order, and shortage costs.
- Heuristic-policy experiments and RL experiments both depend on this folder.
