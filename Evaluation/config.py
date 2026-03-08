from pathlib import Path

import torch


# ===== Pretrained models (update paths when models change) =====
PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_13\saved_model_final"
#PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_3\saved_model"
PRETRAINED_PPO_MODEL_PATH = r"C:\Github\AIIS_META_RL\DRL\Saved_model\Train_4\PPO_Randomized\saved_model"


# ===== Global experiment settings =====
RANDOM_SEED = 2026
DAYS = 200
SCENARIO_MODE = "fixed"  # "randomized" or "fixed"

# Final comparison seeds (paired design across methods per seed/case).
EVALUATION_SEEDS = [RANDOM_SEED + i for i in range(10)]

# Number of sampled stationary scenarios per seed.
STATIONARY_NUM_SCENARIOS_PER_SEED = 10

# Number of sampled nonstationary sequences per seed.
# (One sequence = scenario1 + scenario2 + scenario3 by segment)
NONSTATIONARY_NUM_SEQUENCES_PER_SEED = 10

# Nonstationary day segments (1-based inclusive).
# 1~100: scenario 1, 101~150: scenario 2, 151~200: scenario 3
NONSTATIONARY_SEGMENTS = [
    (1, 100),
    (101, 150),
    (151, 200),
]


# ===== Scenario sampling overrides (uses envs/scenarios.py) =====
# Empty dict means using envs.scenarios.SCENARIO_SAMPLING_DEFAULTS.
SCENARIO_SAMPLING_OVERRIDES = {}


# ===== Methods to compare =====
# Supported kinds: "heuristic_rop", "pretrained_rl"
METHODS = [
    {"name": "R1", "kind": "heuristic_rop", "reorder_point": 1},
    {"name": "R3", "kind": "heuristic_rop", "reorder_point": 3},
    {"name": "R5", "kind": "heuristic_rop", "reorder_point": 5},
    {
        "name": "PPO",
        "kind": "pretrained_rl",
        "checkpoint_path": PRETRAINED_PPO_MODEL_PATH,
        "policy_dist": "categorical",
    },
    {
        "name": "ProMP",
        "kind": "pretrained_rl",
        "checkpoint_path": PRETRAINED_PROMP_MODEL_PATH,
        "policy_dist": "categorical",
    },
]


# ===== Agent architecture settings =====
AGENT_HIDDEN_LAYERS = [64, 64, 64]
LEARN_STD = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Output =====
OUTPUT_ROOT = Path("Evaluation") / "outputs"
METHOD_PLOT_ORDER = ["R1", "R3", "R5", "PPO", "ProMP"]

# Save one integrated boxplot using all seeds/cases combined.
SAVE_SINGLE_INTEGRATED_BOXPLOT = True
