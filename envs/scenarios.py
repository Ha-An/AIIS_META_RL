from envs.config_SimPy import *
from envs.config_RL import *
import itertools
import random
import numpy as np
from collections import defaultdict

# 1. Task Sampling Related

def create_scenarios():
    """
    Creates a set of tasks (e.g., environment or scenario definitions).
    
    This function defines the demand and lead time ranges and creates all possible 
    combinations of these parameters for the tasks in the environment.
    
    Returns:
        scenarios (list): A list of dictionaries representing all possible scenarios 
                          combining demand and leadtime configurations.
    """
    # DEMAND
    demand_uniform_range = [
        (i, j)
        for i in range(5, 16)  # Range for demand min values
        for j in range(i, 16)  # Range for demand max values
        if i <= j
    ]
    
    # Define the uniform demand distribution
    demand_uniform = [
        {"Dist_Type": "UNIFORM", "min": min_val, "max": max_val}
        for min_val, max_val in demand_uniform_range
    ]

    # LEADTIME
    leadtime_uniform_range = [
        (i, j)
        for i in range(1, 5)  # Range for lead time min values
        for j in range(i, 5)  # Range for lead time max values
        if i <= j
    ]
    
    # Define the uniform lead time distribution
    leadtime_uniform = [
        {"Dist_Type": "UNIFORM", "min": min_val, "max": max_val}
        for min_val, max_val in leadtime_uniform_range
    ]

    # Create all combinations of demand and leadtime
    scenarios = [{"DEMAND": demand, "LEADTIME": [random.sample(leadtime_uniform, 1)[0] for _ in range(MAT_COUNT)]} for demand in demand_uniform]

    return scenarios
