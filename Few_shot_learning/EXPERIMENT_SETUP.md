Few-shot experiment setup
=========================

All settings are centralized in `Few_shot_learning/config.py`.

Entry points
------------
- `Few_shot_learning/run_batch_random_scenarios.py`
  - Existing random-scenario batch experiment (5 case families).
- `Few_shot_learning/run_batch_fixed_scenarios.py`
  - New fixed-scenario batch experiment (4 fixed scenarios).
- `Few_shot_learning/experiment_setup.py`
  - Compatibility wrapper to `run_batch_random_scenarios.py`.

How to run
----------
```powershell
# Random-scenario batch
python Few_shot_learning/run_batch_random_scenarios.py

# Fixed-scenario batch (4 fixed cases x 30 seeds by default)
python Few_shot_learning/run_batch_fixed_scenarios.py
```

What to edit
------------
Edit only `Few_shot_learning/config.py`:
- `EXPERIMENTS`, `SEEDS`
- `SCENARIO_CASES` and `build_scenario_dist_config(...)` for random-scenario mode
- `FIXED_SCENARIO_CASES`, `FIXED_SEEDS`, and `build_fixed_scenario_dist_config(...)` for fixed-scenario mode
- `MODEL_CONFIG`, `EVAL_CONFIG`, `EVAL_CONFIG_OVERRIDE`
- `COST_BOXPLOT_CONFIG`
