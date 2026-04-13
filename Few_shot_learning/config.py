from pathlib import Path

import torch


# -----------------------------------------------------------------------------
# Paths
# ROOT_DIR is the repository root. RESULTS_ROOT is where every run writes its
# own timestamped output folder.
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT_DIR / "Few_shot_learning" / "runs"


# -----------------------------------------------------------------------------
# Pretrained checkpoints
# These are the exact paths requested by the user.
# -----------------------------------------------------------------------------
# PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_21\saved_model_final"
PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_23\saved_model_best_totalcost_last_window"
# PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_21\saved_model_best_fewshot"
PRETRAINED_PPO_MODEL_PATH = r"C:\Github\AIIS_META_RL\DRL\Saved_model\Train_8\PPO_Randomized\saved_model"

MODEL_SPECS = [
    {"name": "PPO", "checkpoint_path": PRETRAINED_PPO_MODEL_PATH},
    {"name": "ProMP", "checkpoint_path": PRETRAINED_PROMP_MODEL_PATH},
]

# -----------------------------------------------------------------------------
# Reproducibility / episode horizon
# A single master seed is used everywhere. Scenario generation seeds, support
# sampling seeds, and query sampling seeds are all deterministically derived
# from this value.
# -----------------------------------------------------------------------------
RANDOM_SEED = 2026
DAYS = 200

# -----------------------------------------------------------------------------
# Evaluation modes
# -----------------------------------------------------------------------------
# SCENARIO_MODE:
# - randomized: use the original full-range scenario generator
# - case_randomized: generate stationary-only tasks from one of four narrowed
#   demand/leadtime range cases
SCENARIO_MODE = "case_randomized"

#ENVIRONMENT_MODES = ["stationary", "nonstationary"]
ENVIRONMENT_MODES = ["stationary"]
NONSTATIONARY_SEGMENTS = [
    (1, 50),
    (51, 100),
    (101, 150),
    (151, 200),
]

# -----------------------------------------------------------------------------
# Few-shot protocol
# EVAL_SHOTS:
#   K-shot means K support trajectories are sampled for inner adaptation.
# QUERY_ROLLOUT_PER_TASK:
#   Number of query trajectories used to estimate performance on one generated
#   scenario / one generated nonstationary sequence after adaptation.
# EVAL_ADAPT_UPDATES:
#   Number of inner updates applied while reusing the same K support trajectories.
# -----------------------------------------------------------------------------
EVAL_SHOTS = [1, 2, 3] 
QUERY_ROLLOUT_PER_TASK = 20
EVAL_ADAPT_UPDATES = 3


# -----------------------------------------------------------------------------
# Randomized workload
# RANDOMIZED_STATIONARY_SCENARIO_COUNT:
#   Number of stationary tasks to generate.
# RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT:
#   Number of nonstationary 3-segment task sequences to generate.
# The same generated task is evaluated for both PPO and ProMP.
# -----------------------------------------------------------------------------
RANDOMIZED_STATIONARY_SCENARIO_COUNT = 20
RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT = 20

# -----------------------------------------------------------------------------
# Case-randomized workload
# This mode ignores nonstationary evaluation. Each case narrows the demand and
# leadtime sampling range before scenarios are generated.
# -----------------------------------------------------------------------------
CASE_RANDOMIZED_SCENARIO_COUNT_PER_CASE = 50
CASE_RANDOMIZED_CASES = [
    {
        "name": "HighDemand_LongLead",
        "demand_min": 11,
        "demand_max": 20,
        "leadtime_min": 4,
        "leadtime_max": 7,
    },
    {
        "name": "HighDemand_ShortLead",
        "demand_min": 11,
        "demand_max": 20,
        "leadtime_min": 1,
        "leadtime_max": 4,
    },
    {
        "name": "LowDemand_LongLead",
        "demand_min": 1,
        "demand_max": 10,
        "leadtime_min": 4,
        "leadtime_max": 7,
    },
    {
        "name": "LowDemand_ShortLead",
        "demand_min": 1,
        "demand_max": 10,
        "leadtime_min": 1,
        "leadtime_max": 4,
    },
]

# ----------------------------------------------------------------------------
# Model reconstruction settings for evaluation.
# The policy architecture must match the pretrained checkpoints.
# alpha is the default inner step size used when a checkpoint does not carry
# learned inner_step_sizes (for example PPO).
# -----------------------------------------------------------------------------
BASE_MODEL_CONFIG = {
    "policy_dist": "categorical",
    "layers": [64, 64, 64],
    "alpha": 0.01, #0.003
    "learn_std": True,
    "trainable_learning_rate": True,
    "inner_step_size_max": 0.05,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# -----------------------------------------------------------------------------
# Sampler / runtime settings
# num_tasks stays 1 because each evaluation cell is one generated task.
# envs_per_task is also 1 for stability. parallel is kept False for the same
# reason.
# -----------------------------------------------------------------------------
NUM_TASKS = 1
ENVS_PER_TASK = 1
PARALLEL = False
FAIL_FAST = False
MAX_EPISODE_RETRIES = 5


# -----------------------------------------------------------------------------
# Optional overrides passed into envs.scenarios.create_scenarios(...).
# Leave empty to use the defaults defined in envs/scenarios.py.
# Example:
# RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES = {"demand_min": 5, "demand_max": 15}
# -----------------------------------------------------------------------------
RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES = {}

# -----------------------------------------------------------------------------
# Checkpoint loading
# - auto: if checkpoint has ProMP-style keys (agent.*, inner_step_sizes.*),
#   load the adapter state; otherwise load agent weights only.
# - full: force full adapter load
# - agent_only: load only policy weights and reset inner step sizes to alpha
# -----------------------------------------------------------------------------
CHECKPOINT_LOAD_MODE = "auto"
RESET_INNER_STEP_SIZES_ON_AGENT_ONLY_LOAD = False


# -----------------------------------------------------------------------------
# Plot ordering
# -----------------------------------------------------------------------------
PLOT_MODEL_ORDER = ["PPO", "ProMP"]
