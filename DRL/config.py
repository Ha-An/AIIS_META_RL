from pathlib import Path

# Paths
DRL_DIR = Path(__file__).parent
TENSORBOARD_LOG_PATH = DRL_DIR / "Tensorboard_logs"
SAVED_MODEL_PATH_DRL = DRL_DIR / "Saved_Model"
EVAL_RESULT_PATH = DRL_DIR / "Eval_results"

# PPO runtime defaults
PPO_RUN_CONFIG = {
    "mode": "randomized_task",        # "fixed_task" or "randomized_task"
    # 0: Low Demand, Low Leadtime
    # 1: Medium Demand, Medium Leadtime
    # 2: High Demand, High Leadtime
    # 3: Extreme Demand, Variable Leadtime
    "task_id": 0,                # 0-3 # only for fixed_task mode
    "epochs": 500,
    "beta": 0.0003,
    "clip_eps": 0.15,
    "outer_iters": 3,
    "policy_dist": "categorical",   # "gaussian" or "categorical"
}

# Meta-RL compatible sampling defaults
META_RL_DEFAULTS = {
    "num_task": 5,
    "rollout_per_task": 20,
    "clip_eps": 0.15,
    "outer_iters": 3,
    "action_log_interval": 100,
}

# PPO evaluation defaults
# model_specs:
#   - Used by DRL/eval_ppo.py for multi-model comparison.
#   - Put every checkpoint you want to compare in this list.
#   - name: label shown in plots and summaries
#   - model_path: checkpoint file to load
#   - policy_dist: action distribution expected by that checkpoint
# mode:
#   - fixed_task: evaluate the same handcrafted scenario on every repetition
#   - randomized_task: sample randomized scenarios and compare all models on the exact same sampled tasks
EVAL_MODEL_SPECS = [
    {
        "name": "Train_7",
        "model_path": SAVED_MODEL_PATH_DRL / "Train_7" / "PPO_Randomized" / "saved_model",
        "policy_dist": "categorical",
    },
    {
        "name": "Train_8",
        "model_path": SAVED_MODEL_PATH_DRL / "Train_8" / "PPO_Randomized" / "saved_model",
        "policy_dist": "categorical",
    },
    {
        "name": "Train_9",
        "model_path": SAVED_MODEL_PATH_DRL / "Train_9" / "PPO_Randomized" / "saved_model",
        "policy_dist": "categorical",
    },
]

EVAL_RUN_CONFIG = {
    # Legacy single-model defaults kept for eval_ppo_fixed.py compatibility.
    "model_path": EVAL_MODEL_SPECS[0]["model_path"],
    "mode": "randomized_task",
    "task_id": 1,
    "reps": 200,
    "policy_dist": EVAL_MODEL_SPECS[0]["policy_dist"],
    "seed": 42,
    "output_dir": EVAL_RESULT_PATH,
    "model_specs": EVAL_MODEL_SPECS,
}
