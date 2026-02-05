from envs.config_SimPy import *
from envs.config_RL import *
import itertools
import random
import numpy as np
from collections import defaultdict

# 1. Task (Scenario) Sampling Related

def _default_demand_dists():
    demand_uniform_range = [
        (i, j)
        for i in range(5, 16)  # Range for demand min values
        for j in range(i, 16)  # Range for demand max values
        if i <= j
    ]
    return [
        {"Dist_Type": "UNIFORM", "min": min_val, "max": max_val}
        for min_val, max_val in demand_uniform_range
    ]


def _default_leadtime_dists():
    leadtime_uniform_range = [
        (i, j)
        for i in range(1, 5)  # Range for lead time min values
        for j in range(i, 5)  # Range for lead time max values
        if i <= j
    ]
    return [
        {"Dist_Type": "UNIFORM", "min": min_val, "max": max_val}
        for min_val, max_val in leadtime_uniform_range
    ]


def _build_leadtime_profiles(leadtime_dists, leadtime_mode, leadtime_profiles_count, rng):
    if leadtime_mode == "shared":
        return [[dist for _ in range(MAT_COUNT)] for dist in leadtime_dists]

    if leadtime_mode == "per_material_random":
        count = leadtime_profiles_count or len(leadtime_dists)
        return [
            [rng.choice(leadtime_dists) for _ in range(MAT_COUNT)]
            for _ in range(count)
        ]

    raise ValueError(f"Unknown leadtime_mode: {leadtime_mode}")


def create_scenarios(
    demand_dists=None,
    leadtime_dists=None,
    leadtime_mode="per_material_random",
    leadtime_profiles_count=1,
    seed=None,
):
    """
    Creates a set of tasks (scenario definitions).

    A "scenario/task" is defined by:
      - customer demand distribution
      - supplier leadtime distributions (per material)

    Args:
        demand_dists: list of demand distribution dicts.
        leadtime_dists: list of leadtime distribution dicts.
        leadtime_mode: "shared" (all materials use same dist) or
                       "per_material_random" (sample per material).
        leadtime_profiles_count: number of leadtime profiles to sample for
                                 "per_material_random" mode.
        seed: optional seed for reproducible scenario generation.

    Returns:
        scenarios (list): list of scenario dicts in MetaEnv format.
    """
    rng = random.Random(seed)
    demand_dists = demand_dists or _default_demand_dists()
    leadtime_dists = leadtime_dists or _default_leadtime_dists()

    leadtime_profiles = _build_leadtime_profiles(
        leadtime_dists=leadtime_dists,
        leadtime_mode=leadtime_mode,
        leadtime_profiles_count=leadtime_profiles_count,
        rng=rng,
    )

    scenarios = [
        {"DEMAND": demand, "LEADTIME": leadtime_profile}
        for demand in demand_dists
        for leadtime_profile in leadtime_profiles
    ]

    return scenarios
