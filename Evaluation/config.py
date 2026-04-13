from pathlib import Path

import torch

from envs.scenarios import SCENARIO_SAMPLING_DEFAULTS

# PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_23\saved_model_final"
PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_23\saved_model_best_totalcost_last_window"
# PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_23\saved_model_best_fewshot"
PRETRAINED_PPO_MODEL_PATH = r"C:\Github\AIIS_META_RL\DRL\Saved_model\Train_8\PPO_Randomized\saved_model"

RANDOM_SEED = 2026
DAYS = 200

# -----------------------------------------------------------------------------
# Scenario selection mode
# SCENARIO_MODE = "fixed": Use the five manually defined cases in FIXED_SCENARIO_CASES. Only stationary evaluation is meaningful in this mode.
# SCENARIO_MODE = "randomized":Sample new task/scenario distributions from envs/scenarios.py.Both stationary and nonstationary evaluation can be executed.
# ENVIRONMENT_MODES: Which environment types to run. For randomized mode, use one or both of ["stationary", "nonstationary"]. In fixed mode, nonstationary is ignored.
# -----------------------------------------------------------------------------
SCENARIO_MODE = "randomized"  # "fixed" or "randomized"
ENVIRONMENT_MODES = ["stationary", "nonstationary"]

# -----------------------------------------------------------------------------
# Fixed-scenario evaluation settings
# FIXED_EPISODES_PER_SCENARIO:For each fixed scenario case, run this many independent 200-day episodes. The scenario distribution stays the same; only stochastic realizations. within that scenario vary across repetitions.
# -----------------------------------------------------------------------------
FIXED_EPISODES_PER_SCENARIO = 50

# -----------------------------------------------------------------------------
# Randomized stationary evaluation settings
# RANDOMIZED_STATIONARY_SCENARIO_COUNT: Number of distinct stationary scenarios to generate. Each generated scenario defines a demand distribution and lead-timedistribution(s). The same generated scenario is reused for all policies.
# RANDOMIZED_STATIONARY_EPISODES_PER_SCENARIO: Number of episode rollouts to sample from each generated stationary scenario. Increasing this repeats simulation under the same distribution.
# -----------------------------------------------------------------------------
RANDOMIZED_STATIONARY_SCENARIO_COUNT = 50
RANDOMIZED_STATIONARY_EPISODES_PER_SCENARIO = 5

# -----------------------------------------------------------------------------
# Randomized nonstationary evaluation settings
# RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT:Number of nonstationary scenario sequences to generate.One sequence consists of three scenario distributions, one per segment.
# RANDOMIZED_NONSTATIONARY_EPISODES_PER_SEQUENCE: Number of episode rollouts to sample from each generated 3-segment sequence.
# -----------------------------------------------------------------------------
RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT = 50
RANDOMIZED_NONSTATIONARY_EPISODES_PER_SEQUENCE = 5

# -----------------------------------------------------------------------------
# Nonstationary segment definition
# Each tuple is (start_day, end_day), inclusive.
# The current evaluation design uses exactly three segments:
#   1-100, 101-150, 151-200
# A new scenario distribution is applied at the start of each segment.
# -----------------------------------------------------------------------------
NONSTATIONARY_SEGMENTS = [
    (1, 50),
    (51, 100),
    (101, 150),
    (151, 200),
]


# -----------------------------------------------------------------------------
# Randomized scenario sampling overrides
# Leave this empty to use envs.scenarios.SCENARIO_SAMPLING_DEFAULTS as-is.
# You can override keys such as demand/lead-time range options here when you
# want randomized evaluation to sample from a narrower or broader task family.
# Example:
# RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES = {
#     "demand_min": 5,
#     "demand_max": 25,
# }
# -----------------------------------------------------------------------------
RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES = {}


# -----------------------------------------------------------------------------
# Fixed stationary cases
# Each case defines one stationary task distribution.
# - demand_dist: distribution for customer demand size
# - leadtime_dist: shared lead-time distribution applied to all materials
#
# The first case mirrors the default randomized train-range settings from
# envs.scenarios.SCENARIO_SAMPLING_DEFAULTS.
# The remaining four cases explicitly split the space into low/high demand and
# short/long lead-time combinations.
# -----------------------------------------------------------------------------
FIXED_SCENARIO_CASES = [
    {
        "case": "Fixed_Default_TrainRange_Uniform",
        "demand_dist": {
            "Dist_Type": "UNIFORM",
            "min": SCENARIO_SAMPLING_DEFAULTS["demand_min"],
            "max": SCENARIO_SAMPLING_DEFAULTS["demand_max"],
        },
        "leadtime_dist": {
            "Dist_Type": "UNIFORM",
            "min": SCENARIO_SAMPLING_DEFAULTS["leadtime_min"],
            "max": SCENARIO_SAMPLING_DEFAULTS["leadtime_max"],
        },
    },
    {
        "case": "Fixed_LowDemand_ShortLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 10},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 3},
    },
    {
        "case": "Fixed_LowDemand_LongLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 10},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 4, "max": 7},
    },
    {
        "case": "Fixed_HighDemand_ShortLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 11, "max": 20},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 3},
    },
    {
        "case": "Fixed_HighDemand_LongLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 11, "max": 20},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 4, "max": 7},
    },
]


# -----------------------------------------------------------------------------
# Heuristic baselines
# These are simple reorder-point rules. For example, R3 means "place a material
# order when inventory reaches reorder point 3" according to the heuristic
# simulator logic referenced from envs/reorder_point_experiment.py.
# -----------------------------------------------------------------------------
HEURISTIC_REORDER_POINTS = [1, 3, 5]


# -----------------------------------------------------------------------------
# Pretrained policy definitions
# name:
#   Label used in result tables and plots.
# checkpoint_path:
#   Filesystem path to the saved model.
# policy_dist:
#   Action distribution family expected by the saved policy network.
#   Current checkpoints are categorical discrete-action policies.
# -----------------------------------------------------------------------------
PRETRAINED_MODELS = [
    {
        "name": "PPO",
        "checkpoint_path": PRETRAINED_PPO_MODEL_PATH,
        "policy_dist": "categorical",
    },
    {
        "name": "ProMP",
        "checkpoint_path": PRETRAINED_PROMP_MODEL_PATH,
        "policy_dist": "categorical",
    },
]


# -----------------------------------------------------------------------------
# Policy-network reconstruction settings
# These values are used only to rebuild the policy network architecture before
# loading checkpoint weights for PPO/ProMP evaluation.
# They are not training hyperparameters for a new run.
# -----------------------------------------------------------------------------
AGENT_HIDDEN_LAYERS = [64, 64, 64]
LEARN_STD = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Robustness / output settings
# MAX_EPISODE_RETRIES:
#   Retry count for a single episode if the external env code hits one of the
#   known intermittent state-corruption failures.
# OUTPUT_ROOT:
#   Root directory under which each run creates
#   Evaluation/outputs/run_YYYYMMDD_HHMMSS/.
# PLOT_METHOD_ORDER:
#   Fixed display order for box plots and summary tables.
# -----------------------------------------------------------------------------
MAX_EPISODE_RETRIES = 5

OUTPUT_ROOT = Path("Evaluation") / "outputs"
PLOT_METHOD_ORDER = ["R1", "R3", "R5", "PPO", "ProMP"]
