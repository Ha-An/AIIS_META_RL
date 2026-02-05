import os
import copy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import Few_shot_learning.config as cfg
import Few_shot_learning.eval_few_shot as eval_few_shot


# =========================
# Experiment definitions
# =========================
EXPERIMENTS = [
    {
        "name": "ProMP",
        "model_path": r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_2\saved_model",
        "model_config": {
            # Must match pretraininㄴg
            "policy_dist": "categorical",
            "layers": [64, 64, 64],
            "learn_std": True,
        },
    },
    # {
    #     "name": "VPG_MAML",
    #     "model_path": r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_2\saved_model",
    #     "model_config": {
    #         "policy_dist": "categorical",
    #         "layers": [64, 64, 64],
    #         "learn_std": True,
    #     },
    # },
    # {
    #     "name": "PPO_Fixed_Task0",
    #     "model_path": r"C:\Github\AIIS_META_RL\DRL\Saved_Model\Train_4\PPO_Fixed\saved_model",
    #     "model_config": {
    #         "policy_dist": "categorical",
    #         "layers": [64, 64, 64],
    #         "learn_std": True,
    #     },
    # },
    # {
    #     "name": "PPO_Fixed_Task1",
    #     "model_path": r"C:\Github\AIIS_META_RL\DRL\Saved_Model\Train_6\PPO_Fixed\saved_model",
    #     "model_config": {
    #         "policy_dist": "categorical",
    #         "layers": [64, 64, 64],
    #         "learn_std": True,
    #     },
    # },
    # {
    #     "name": "PPO_Fixed_Task2",
    #     "model_path": r"C:\Github\AIIS_META_RL\DRL\Saved_Model\Train_7\PPO_Fixed\saved_model",
    #     "model_config": {
    #         "policy_dist": "categorical",
    #         "layers": [64, 64, 64],
    #         "learn_std": True,
    #     },
    # },
    # {
    #     "name": "PPO_Fixed_Task3",
    #     "model_path": r"C:\Github\AIIS_META_RL\DRL\Saved_Model\Train_8\PPO_Fixed\saved_model",
    #     "model_config": {
    #         "policy_dist": "categorical",
    #         "layers": [64, 64, 64],
    #         "learn_std": True,
    #     },
    # },
    # {
    #     "name": "PPO_randomized",
    #     "model_path": r"C:\Github\AIIS_META_RL\DRL\Saved_Model\Train_5\PPO_Randomized\saved_model",
    #     "model_config": {
    #         "policy_dist": "categorical",
    #         "layers": [64, 64, 64],
    #         "learn_std": True,
    #     },
    # },
]

SEEDS = [0]

# Scenario cases for few-shot evaluation
CASE_SCENARIOS = [
    {
        "case": "LowDemand_ShortLead",
        "demand_min_range": (5, 8),
        "demand_max_range": (5, 8),
        "lead_min_range": (1, 2),
        "lead_max_range": (1, 2),
    },
    {
        "case": "LowDemand_LongLead",
        "demand_min_range": (5, 8),
        "demand_max_range": (5, 8),
        "lead_min_range": (4, 5),
        "lead_max_range": (4, 5),
    },
    {
        "case": "HighDemand_ShortLead",
        "demand_min_range": (16, 20),
        "demand_max_range": (16, 20),
        "lead_min_range": (1, 2),
        "lead_max_range": (1, 2),
    },
    {
        "case": "HighDemand_LongLead",
        "demand_min_range": (16, 20),
        "demand_max_range": (16, 20),
        "lead_min_range": (4, 5),
        "lead_max_range": (4, 5),
    },
]

# Global evaluation settings (apply to all runs)
EVAL_CONFIG_OVERRIDE = {
    "eval_rounds": 10,
    "max_adapt_steps": 10,
    "seed": None,
    "parallel": False,
    "envs_per_task": 1,
    "num_task": 5,
    "rollout_per_task": 10,
    "max_path_length": cfg.MODEL_CONFIG["max_path_length"],
    "max_total_env_steps": 5_000_000,
    "auto_clamp": True,
    "action_log_interval": 10,
}


def _apply_overrides(base_cfg, overrides):
    for k, v in overrides.items():
        base_cfg[k] = v


def _build_uniform_dists(min_range, max_range, step=1):
    dists = []
    for min_val in range(min_range[0], min_range[1] + 1, step):
        for max_val in range(max_range[0], max_range[1] + 1, step):
            if min_val <= max_val:
                dists.append({"Dist_Type": "UNIFORM", "min": min_val, "max": max_val})
    return dists


def _scenario_config_for_case(case_def):
    return {
        "demand_dists": _build_uniform_dists(
            case_def["demand_min_range"],
            case_def["demand_max_range"],
            step=1,
        ),
        "leadtime_dists": _build_uniform_dists(
            case_def["lead_min_range"],
            case_def["lead_max_range"],
            step=1,
        ),
        "leadtime_mode": "per_material_random",
        "leadtime_profiles_count": 1,
        "seed": None,
    }


def main():
    # Snapshot original configs so we can restore between runs
    orig_model = copy.deepcopy(cfg.MODEL_CONFIG)
    orig_eval = copy.deepcopy(cfg.EVAL_CONFIG)
    orig_scenario = copy.deepcopy(cfg.SCENARIO_DIST_CONFIG)
    orig_model_path = cfg.PRETRAINED_MODEL_PATH

    try:
        for exp in EXPERIMENTS:
            for case_def in CASE_SCENARIOS:
                for seed in SEEDS:
                    # Reset configs each run
                    cfg.MODEL_CONFIG = copy.deepcopy(orig_model)
                    cfg.EVAL_CONFIG = copy.deepcopy(orig_eval)
                    cfg.SCENARIO_DIST_CONFIG = copy.deepcopy(orig_scenario)

                    # Apply overrides
                    cfg.PRETRAINED_MODEL_PATH = exp["model_path"]
                    _apply_overrides(cfg.MODEL_CONFIG, exp.get("model_config", {}))
                    _apply_overrides(cfg.EVAL_CONFIG, EVAL_CONFIG_OVERRIDE)
                    cfg.EVAL_CONFIG["seed"] = seed
                    cfg.SCENARIO_DIST_CONFIG = _scenario_config_for_case(case_def)

                    run_name = f"{exp['name']}_{case_def['case']}_seed{seed}"
                    os.environ["FEWSHOT_RUN_NAME"] = run_name
                    print(f"\n[FEWSHOT] Running: {run_name}")
                    print(f"  model_path: {cfg.PRETRAINED_MODEL_PATH}")

                    eval_few_shot.main()
    finally:
        # Restore original configs
        cfg.MODEL_CONFIG = orig_model
        cfg.EVAL_CONFIG = orig_eval
        cfg.SCENARIO_DIST_CONFIG = orig_scenario
        cfg.PRETRAINED_MODEL_PATH = orig_model_path
        os.environ.pop("FEWSHOT_RUN_NAME", None)


if __name__ == "__main__":
    main()
