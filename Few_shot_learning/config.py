import torch
from envs.config_SimPy import SIM_TIME

# Path to pretrained ProMP model (state_dict saved from AIIS_META/main.py)
PRETRAINED_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_2\saved_model"
#PRETRAINED_MODEL_PATH = r"C:\Github\AIIS_META_RL\DRL\Saved_model\Train_1\PPO_Randomized\saved_model"

# Scenario distribution config (same keys as envs.scenarios.create_scenarios)
# You can control UNIFORM distributions by specifying min/max ranges below.
# DEMAND_MIN_RANGE = (5, 15)   # inclusive range of possible demand mins
# DEMAND_MAX_RANGE = (5, 15)   # inclusive range of possible demand maxs
# LEADTIME_MIN_RANGE = (1, 4)  # inclusive range of possible leadtime mins
# LEADTIME_MAX_RANGE = (1, 4)  # inclusive range of possible leadtime maxs
DEMAND_MIN_RANGE = (16, 18)
DEMAND_MAX_RANGE = (16, 20)
LEADTIME_MIN_RANGE = (2, 5)
LEADTIME_MAX_RANGE = (2, 5)


# Use a larger step to reduce the number of scenarios
DEMAND_STEP = 1
LEADTIME_STEP = 1


def _build_uniform_dists(min_range, max_range, step=1):
    dists = []
    for min_val in range(min_range[0], min_range[1] + 1, step):
        for max_val in range(max_range[0], max_range[1] + 1, step):
            if min_val <= max_val:
                dists.append({"Dist_Type": "UNIFORM", "min": min_val, "max": max_val})
    return dists


SCENARIO_DIST_CONFIG = {
    "demand_dists": _build_uniform_dists(DEMAND_MIN_RANGE, DEMAND_MAX_RANGE, DEMAND_STEP),
    "leadtime_dists": _build_uniform_dists(LEADTIME_MIN_RANGE, LEADTIME_MAX_RANGE, LEADTIME_STEP),
    "leadtime_mode": "per_material_random",
    "leadtime_profiles_count": 1,
    "seed": None,
}

# Meta-RL compatible model settings (must match pretraining for safe load)
MODEL_CONFIG = {
    "algorithm": "ProMP",
    "policy_dist": "categorical",  # "gaussian" or "categorical"
    "layers": [64, 64, 64],
    "num_task": 10,
    "rollout_per_task": 5,
    "max_path_length": SIM_TIME,
    "alpha": 0.002,
    "beta": 0.0005,
    "num_inner_grad": 3,
    "outer_iters": 5,
    "clip_eps": 0.2,
    "parallel": True,
    "learn_std": True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Few-shot evaluation settings
EVAL_CONFIG = {
    "eval_rounds": 10,      # number of task sets to evaluate
    "max_adapt_steps": 20,  # 0..N adaptation steps for curve
    "seed": 42,
    # Safety defaults to prevent overload during evaluation
    "parallel": False,
    "envs_per_task": 1,
    # Optional overrides (None = use MODEL_CONFIG)
    "num_task": None,
    "rollout_per_task": None,
    "max_path_length": None,
    # Hard cap on total simulated env steps
    "max_total_env_steps": 5_000_000,
    "auto_clamp": True,
    # Action logging (TensorBoard)
    "action_log_interval": 10,  # log every N adaptation steps
}
