from envs.config_SimPy import *
import random

SCENARIO_SAMPLING_DEFAULTS = {
    "demand_min": 1,
    "demand_max": 20,
    "leadtime_min": 1,
    "leadtime_max": 7,
    "leadtime_mode": "per_material_random",
    "num_scenarios": 200,
}


def _default_demand_dists(demand_min=6, demand_max=12):
    uniform_dists = [
        {"Dist_Type": "UNIFORM", "min": i, "max": j}
        for i in range(demand_min, demand_max + 1)
        for j in range(i, demand_max + 1)
    ]
    normal_dists = [
        {"Dist_Type": "NORMAL", "mean": mean, "std": std}
        for mean in range(demand_min, demand_max + 1)
        for std in [1, 2, 3]
    ]
    poisson_dists = [
        {"Dist_Type": "POISSON", "lam": lam}
        for lam in range(demand_min, demand_max + 1)
    ]
    return uniform_dists + normal_dists + poisson_dists


def _default_leadtime_dists(leadtime_min=1, leadtime_max=3):
    uniform_dists = [
        {"Dist_Type": "UNIFORM", "min": i, "max": j}
        for i in range(leadtime_min, leadtime_max + 1)
        for j in range(i, leadtime_max + 1)
    ]
    normal_dists = [
        {"Dist_Type": "NORMAL", "mean": mean, "std": std}
        for mean in [x / 2 for x in range(leadtime_min * 2, leadtime_max * 2 + 1)]
        for std in [0.4, 0.8]
    ]
    poisson_dists = [
        {"Dist_Type": "POISSON", "lam": x / 2}
        for x in range(leadtime_min * 2, leadtime_max * 2 + 1)
    ]
    return uniform_dists + normal_dists + poisson_dists


def create_scenarios(
    demand_dists=None,
    leadtime_dists=None,
    demand_min=None,
    demand_max=None,
    leadtime_min=None,
    leadtime_max=None,
    leadtime_mode=None,
    leadtime_profiles_count=None,  # Kept for backward compatibility (unused)
    num_scenarios=None,
    seed=None,
):
    """
    Create a scenario pool by random sampling.

    Each scenario samples:
      - one demand distribution (type + parameters)
      - one leadtime profile for all suppliers
        * shared: one distribution shared by all suppliers
        * per_material_random: each supplier sampled independently
    """
    demand_min = SCENARIO_SAMPLING_DEFAULTS["demand_min"] if demand_min is None else demand_min
    demand_max = SCENARIO_SAMPLING_DEFAULTS["demand_max"] if demand_max is None else demand_max
    leadtime_min = SCENARIO_SAMPLING_DEFAULTS["leadtime_min"] if leadtime_min is None else leadtime_min
    leadtime_max = SCENARIO_SAMPLING_DEFAULTS["leadtime_max"] if leadtime_max is None else leadtime_max
    leadtime_mode = SCENARIO_SAMPLING_DEFAULTS["leadtime_mode"] if leadtime_mode is None else leadtime_mode
    num_scenarios = SCENARIO_SAMPLING_DEFAULTS["num_scenarios"] if num_scenarios is None else num_scenarios

    rng = random.Random(seed)
    demand_candidates = demand_dists or _default_demand_dists(demand_min=demand_min, demand_max=demand_max)
    leadtime_candidates = leadtime_dists or _default_leadtime_dists(
        leadtime_min=leadtime_min, leadtime_max=leadtime_max
    )

    scenarios = []
    for _ in range(num_scenarios):
        demand = dict(rng.choice(demand_candidates))

        if leadtime_mode == "shared":
            dist = dict(rng.choice(leadtime_candidates))
            leadtime_profile = [dict(dist) for _ in range(MAT_COUNT)]
        elif leadtime_mode == "per_material_random":
            leadtime_profile = [dict(rng.choice(leadtime_candidates)) for _ in range(MAT_COUNT)]
        else:
            raise ValueError(f"Unknown leadtime_mode: {leadtime_mode}")

        scenarios.append({"DEMAND": demand, "LEADTIME": leadtime_profile})

    return scenarios
