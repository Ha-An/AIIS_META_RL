from pathlib import Path

# Paths
DRL_DIR = Path(__file__).parent
TENSORBOARD_LOG_PATH = DRL_DIR / "Tensorboard_logs"
SAVED_MODEL_PATH_DRL = DRL_DIR / "Saved_Model"

# PPO runtime defaults
PPO_RUN_CONFIG = {
    "mode": "randomized_task",        # "fixed_task" or "randomized_task"
    # 0: Low Demand, Low Leadtime
    # 1: Medium Demand, Medium Leadtime
    # 2: High Demand, High Leadtime
    # 3: Extreme Demand, Variable Leadtime
    "task_id": 0,                # 0-3 # only for fixed_task mode
    "epochs": 500,
    "beta": 0.0005,
    "clip_eps": 0.3,
    "outer_iters": 5,
    "policy_dist": "categorical",   # "gaussian" or "categorical"
}

# Meta-RL compatible sampling defaults
META_RL_DEFAULTS = {
    "num_task": 5,
    "rollout_per_task": 20,
    "clip_eps": 0.3,
    "outer_iters": 5,
    "action_log_interval": 100,
}
