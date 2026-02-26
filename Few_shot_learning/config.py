import torch
from envs.config_SimPy import SIM_TIME, MAT_COUNT
from envs.scenarios import SCENARIO_SAMPLING_DEFAULTS

# ------------------------------------------------------------------
# Global random seed (single source of truth for few-shot runs)
# ------------------------------------------------------------------
RANDOM_SEED = 42
NUM_SEEDS = 5

# ------------------------------------------------------------------
# Pretrained model path (default for single-run eval_few_shot.py)
# ------------------------------------------------------------------
PRETRAINED_PROMP_MODEL_PATH = r"C:\Github\AIIS_META_RL\AIIS_META\Saved_model\Train_4\saved_model"
PRETRAINED_PPO_MODEL_PATH = r"C:\Github\AIIS_META_RL\DRL\Saved_model\Train_4\PPO_Randomized\saved_model"

# Default model path for single-run eval_few_shot.py
PRETRAINED_MODEL_PATH = PRETRAINED_PROMP_MODEL_PATH

# ------------------------------------------------------------------
# Fixed-scenario quick setup (default mode)
# Edit this block first for fixed-scenario experiments.
# ------------------------------------------------------------------
DEFAULT_BATCH_MODE = "fixed"  # "fixed" or "random"

# Fixed batch evaluation budget / sampling
FIXED_NUM_SEEDS = 10
FIXED_SEED_START = 42
FIXED_SEEDS = list(range(FIXED_SEED_START, FIXED_SEED_START + FIXED_NUM_SEEDS))
FIXED_SCENARIO_POOL_SIZE = 1

# Fixed batch runtime overrides (applied by run_batch_fixed_scenarios.py)
FIXED_EVAL_OVERRIDES = {
    # Number of adaptation-evaluation repeats per run (higher = slower, lower variance).
    "eval_rounds": 1,
    # Maximum few-shot adaptation step index (10 means internal evaluation over 0..10).
    "max_adapt_steps": 10,
    # Base random seed (overwritten per run by FIXED_SEEDS in fixed batch mode).
    "seed": RANDOM_SEED,
    # Use multiprocessing sampler workers (False is usually faster when num_task=1).
    "parallel": False,
    # Number of env workers per task (effective when parallel=True; must be <= rollout_per_task).
    "envs_per_task": 1,
    # Number of tasks sampled per iteration (fixed mode uses 1 for a single scenario).
    "num_task": 1,
    # Rollouts collected per task at each step (main speed vs stability knob).
    "rollout_per_task": 10,
    # Rollout horizon in days (runtime scales roughly linearly with this).
    "max_path_length": SIM_TIME,
    # Hard cap on total env steps per run to avoid runaway evaluation time.
    "max_total_env_steps": 5_000_000,
    # Auto-reduce eval parameters if estimated steps exceed max_total_env_steps.
    "auto_clamp": True,
    # Logging interval for action stats when adaptation-curve logging is enabled.
    "action_log_interval": 10,
}

# Fixed batch boxplot settings
FIXED_COST_BOXPLOT_ADAPT_STEPS = [0, 1, 2, 3]
FIXED_COST_BOXPLOT_REPETITIONS = 1
FIXED_SKIP_ADAPTATION_CURVE = True

# Fixed scenarios for deterministic case-wise evaluation.
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
        "leadtime_mode": "per_material_random",
        "scenario_seed": 100,
    },
    {
        "case": "Fixed_LowDemand_ShortLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 5},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 3},
        "leadtime_mode": "per_material_random",
        "scenario_seed": 101,
    },
    {
        "case": "Fixed_LowDemand_LongLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 5},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 3, "max": 5},
        "leadtime_mode": "per_material_random",
        "scenario_seed": 102,
    },
    {
        "case": "Fixed_HighDemand_ShortLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 12, "max": 18},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 1, "max": 3},
        "leadtime_mode": "per_material_random",
        "scenario_seed": 103,
    },
    {
        "case": "Fixed_HighDemand_LongLead",
        "demand_dist": {"Dist_Type": "UNIFORM", "min": 12, "max": 18},
        "leadtime_dist": {"Dist_Type": "UNIFORM", "min": 3, "max": 5},
        "leadtime_mode": "per_material_random",
        "scenario_seed": 104,
    },
]

# If non-empty, fixed batch runs only these case names.
FIXED_ACTIVE_CASES = []

# ------------------------------------------------------------------
# Checkpoint load policy for fair few-shot comparison
# ------------------------------------------------------------------
# "agent_only": always load only policy-network weights
# "auto": load full ProMP state for Meta-RL checkpoints, agent-only for PPO checkpoints
# "full": always try loading full ProMP state_dict
CHECKPOINT_LOAD_MODE = "auto"

# When using agent_only mode, reinitialize inner-step sizes to alpha for fairness.
RESET_INNER_STEP_SIZES_ON_LOAD = False

# ------------------------------------------------------------------
# Model config (must match checkpoint architecture)
# ------------------------------------------------------------------
MODEL_CONFIG = {
    "algorithm": "ProMP",
    "policy_dist": "categorical",
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


# ------------------------------------------------------------------
# Few-shot evaluation runtime config
# ------------------------------------------------------------------
EVAL_CONFIG = {
    "eval_rounds": 10,      # adaptation curve repeats
    "max_adapt_steps": 10,  # curve from 0..10
    "seed": RANDOM_SEED,
    "parallel": False,
    "envs_per_task": 1,
    "num_task": None,       # None -> use MODEL_CONFIG
    "rollout_per_task": None,
    "max_path_length": None,
    "max_total_env_steps": 5_000_000,
    "auto_clamp": True,
    "action_log_interval": 10,
}


# ------------------------------------------------------------------
# Boxplot config:
# run 30 repetitions for adaptation steps 0, 5, 10 and draw total-cost boxplot
# ------------------------------------------------------------------
COST_BOXPLOT_CONFIG = {
    "enabled": True,
    "adapt_steps": list(FIXED_COST_BOXPLOT_ADAPT_STEPS)
    if DEFAULT_BATCH_MODE == "fixed" else [0, 5, 10],
    "repetitions": FIXED_COST_BOXPLOT_REPETITIONS
    if DEFAULT_BATCH_MODE == "fixed" else 30,
}

# TensorBoard logging level for few-shot evaluation:
# - "off": disable TB event writing
# - "summary": compact logs (recommended default)
# - "full": detailed logs (actions histogram + per-repetition costs)
FEWSHOT_TB_MODE = "summary"


# ------------------------------------------------------------------
# Scenario helper builders
# ------------------------------------------------------------------
def _build_uniform_dists(min_val, max_val):
    return [
        {"Dist_Type": "UNIFORM", "min": i, "max": j}
        for i in range(min_val, max_val + 1)
        for j in range(i, max_val + 1)
    ]


def _build_normal_dists_for_demand(min_val, max_val):
    return [
        {"Dist_Type": "NORMAL", "mean": mean, "std": std}
        for mean in range(min_val, max_val + 1)
        for std in [1, 2, 3]
    ]


def _build_poisson_dists_for_demand(min_val, max_val):
    return [{"Dist_Type": "POISSON", "lam": lam} for lam in range(min_val, max_val + 1)]


def _build_normal_dists_for_leadtime(min_val, max_val):
    return [
        {"Dist_Type": "NORMAL", "mean": x / 2, "std": std}
        for x in range(min_val * 2, max_val * 2 + 1)
        for std in [0.4, 0.8]
    ]


def _build_poisson_dists_for_leadtime(min_val, max_val):
    return [{"Dist_Type": "POISSON", "lam": x / 2} for x in range(min_val * 2, max_val * 2 + 1)]


def build_scenario_dist_config(case_cfg):
    """
    Build scenario config dict passed to envs.scenarios.create_scenarios(...).
    """
    if case_cfg.get("use_meta_default", False):
        # Use the same default range/mode as MetaRL training.
        return {
            "demand_min": SCENARIO_SAMPLING_DEFAULTS["demand_min"],
            "demand_max": SCENARIO_SAMPLING_DEFAULTS["demand_max"],
            "leadtime_min": SCENARIO_SAMPLING_DEFAULTS["leadtime_min"],
            "leadtime_max": SCENARIO_SAMPLING_DEFAULTS["leadtime_max"],
            "leadtime_mode": SCENARIO_SAMPLING_DEFAULTS["leadtime_mode"],
            "num_scenarios": SCENARIO_SAMPLING_DEFAULTS["num_scenarios"],
            "seed": case_cfg.get("seed", RANDOM_SEED),
        }

    d_min, d_max = case_cfg["demand_range"]
    l_min, l_max = case_cfg["leadtime_range"]
    dist_families = case_cfg.get("dist_families", ["UNIFORM", "NORMAL", "POISSON"])

    demand_dists = []
    leadtime_dists = []

    if "UNIFORM" in dist_families:
        demand_dists.extend(_build_uniform_dists(d_min, d_max))
        leadtime_dists.extend(_build_uniform_dists(l_min, l_max))
    if "NORMAL" in dist_families:
        demand_dists.extend(_build_normal_dists_for_demand(d_min, d_max))
        leadtime_dists.extend(_build_normal_dists_for_leadtime(l_min, l_max))
    if "POISSON" in dist_families:
        demand_dists.extend(_build_poisson_dists_for_demand(d_min, d_max))
        leadtime_dists.extend(_build_poisson_dists_for_leadtime(l_min, l_max))

    return {
        "demand_dists": demand_dists,
        "leadtime_dists": leadtime_dists,
        "leadtime_mode": case_cfg.get("leadtime_mode", "per_material_random"),
        "num_scenarios": case_cfg.get("num_scenarios", 200),
        "seed": case_cfg.get("seed", RANDOM_SEED),
    }


# ------------------------------------------------------------------
# Few-shot batch experiment settings
# ------------------------------------------------------------------
EXPERIMENTS = [
    {
        "name": "ProMP",
        "pretrained_model_path": PRETRAINED_PROMP_MODEL_PATH,
        "model_config": {
            "policy_dist": "categorical",
            "layers": [64, 64, 64],
            "learn_std": True,
        },
    },
    {
        "name": "PPO",
        "pretrained_model_path": PRETRAINED_PPO_MODEL_PATH,
        "model_config": {
            "policy_dist": "categorical",
            "layers": [64, 64, 64],
            "learn_std": True,
        },
    },
]

SEEDS = [RANDOM_SEED + i for i in range(NUM_SEEDS)]

# Five scenario cases:
# 1) MetaRL-default range (baseline)
# 2~5) Low/High demand x Short/Long leadtime
SCENARIO_CASES = [
    {
        "case": "MetaDefault_Mixed",
        "use_meta_default": True,
    },
    {
        "case": "LowDemand_ShortLead",
        "demand_range": (1, 10),
        "leadtime_range": (1, 3),
        "dist_families": ["UNIFORM", "NORMAL", "POISSON"],
    },
    {
        "case": "LowDemand_LongLead",
        "demand_range": (1, 10),
        "leadtime_range": (5, 7),
        "dist_families": ["UNIFORM", "NORMAL", "POISSON"],
    },
    {
        "case": "HighDemand_ShortLead",
        "demand_range": (11, 20),
        "leadtime_range": (1, 3),
        "dist_families": ["UNIFORM", "NORMAL", "POISSON"],
    },
    {
        "case": "HighDemand_LongLead",
        "demand_range": (11, 20),
        "leadtime_range": (5, 7),
        "dist_families": ["UNIFORM", "NORMAL", "POISSON"],
    },
]

# Config used by random-scenario batch runner.
EVAL_CONFIG_OVERRIDE_RANDOM = {
    "eval_rounds": 10,
    "max_adapt_steps": 10,
    "seed": RANDOM_SEED,
    "parallel": True,
    # Use vectorized rollout workers per task to speed up few-shot evaluation.
    "envs_per_task": 10,
    "num_task": 5,
    "rollout_per_task": 10,
    "max_path_length": MODEL_CONFIG["max_path_length"],
    "max_total_env_steps": 5_000_000,
    "auto_clamp": True,
    "action_log_interval": 10,
}

# Config used by fixed-scenario batch runner.
EVAL_CONFIG_OVERRIDE_FIXED = dict(FIXED_EVAL_OVERRIDES)

# Backward-compatible alias used by random runner.
EVAL_CONFIG_OVERRIDE = (
    dict(EVAL_CONFIG_OVERRIDE_FIXED)
    if DEFAULT_BATCH_MODE == "fixed"
    else dict(EVAL_CONFIG_OVERRIDE_RANDOM)
)

# Default single-run scenario config (used when running eval_few_shot.py directly).
SCENARIO_DIST_CONFIG = build_scenario_dist_config(SCENARIO_CASES[0])


def build_fixed_scenario_dist_config(case_cfg):
    """
    Build a fixed scenario pool config from explicitly configured scenario
    definitions in this file (no distribution-family sampling).
    """
    demand = dict(case_cfg["demand_dist"])

    if "leadtime_by_material" in case_cfg:
        if len(case_cfg["leadtime_by_material"]) != MAT_COUNT:
            raise ValueError(
                f"leadtime_by_material length must be MAT_COUNT({MAT_COUNT}), "
                f"got {len(case_cfg['leadtime_by_material'])}"
            )
        leadtime_profile = [dict(x) for x in case_cfg["leadtime_by_material"]]
    else:
        base_lt = dict(case_cfg["leadtime_dist"])
        leadtime_profile = [dict(base_lt) for _ in range(MAT_COUNT)]

    scenario_template = {"DEMAND": demand, "LEADTIME": leadtime_profile}
    n = int(case_cfg.get("num_scenarios", FIXED_SCENARIO_POOL_SIZE))
    fixed_pool = [
        {"DEMAND": dict(scenario_template["DEMAND"]), "LEADTIME": [dict(x) for x in scenario_template["LEADTIME"]]}
        for _ in range(n)
    ]

    return {
        # eval_few_shot._create_scenarios() will detect this key and bypass
        # envs.scenarios.create_scenarios() sampling.
        "fixed_scenarios": fixed_pool,
        "seed": int(case_cfg.get("scenario_seed", RANDOM_SEED)),
    }
