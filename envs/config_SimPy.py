import random  # For random number generation
import os
import numpy as np
#### Items #####################################################################
# ID: Index of the element in the dictionary
# TYPE: Product, Material, WIP;
# NAME: Item's name or model;
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to suppliers [days]
# INIT_LEVEL: Initial inventory level [units]
# DEMAND_QUANTITY: Demand quantity for the final product [units] -> THIS IS UPDATED EVERY 24 HOURS (Default: 0)
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# REMOVE## LOT_SIZE_ORDER: Lot-size for the order of materials (Q) [units] -> THIS IS AN AGENT ACTION THAT IS UPDATED EVERY 24 HOURS
# HOLD_COST: Holding cost of the items [$/unit*day]
# PURCHASE_COST: Purchase cost of the materials [$/unit]
# SETUP_COST_PRO: Setup cost for the delivery of the products to the customer [$/delivery]
# ORDER_COST_TO_SUP: Ordering cost for the materials to a supplier [$/order]
# DELIVERY_COST: Delivery cost of the products [$/unit]
# DUE_DATE: Term of customer order to delivered [days]
# SHORTAGE_COST: Backorder cost of products [$/unit]

#### Processes #####################################################################
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day] (Production rate must be a positive number between 1 and 24.)
# INPUT_TYPE_LIST: List of types of input materials or WIPs
# QNTY_FOR_INPUT_ITEM: Quantity of input materials or WIPs [units]
# OUTPUT: Output WIP or Product
# PROCESS_COST: Processing cost of the process [$/unit]
# PROCESS_STOP_COST: Penalty cost for stopping the process [$/unit]

#### Dictionaries #####################################################################
# I: Basic information by inventory according to AP
# P: Process basic information according to AP
I = {
    "AP1": {
        0: {
            "ID": 0,
            "TYPE": "Product",
            "NAME": "PRODUCT",
            "CUST_ORDER_CYCLE": 7,
            "INIT_LEVEL": 0,
            "DEMAND_QUANTITY": 0,
            "HOLD_COST": 1,
            "SETUP_COST_PRO": 1,
            "DELIVERY_COST": 1,
            "DUE_DATE": 7,
            "SHORTAGE_COST_PRO": 50
        },
        1: {
            "ID": 1,
            "TYPE": "Material",
            "NAME": "MAT 1",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 1,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        }
    },
    "AP2": {
        0: {
            "ID": 0,
            "TYPE": "Product",
            "NAME": "PROD",
            "CUST_ORDER_CYCLE": 7,
            "INIT_LEVEL": 0,
            "DEMAND_QUANTITY": 0,
            "HOLD_COST": 1,
            "SETUP_COST_PRO": 1,
            "DELIVERY_COST": 1,
            "DUE_DATE": 7,
            "SHORTAGE_COST_PRO": 50
        },
        1: {
            "ID": 1,
            "TYPE": "Material",
            "NAME": "MAT 1",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        2: {
            "ID": 2,
            "TYPE": "Material",
            "NAME": "MAT 2",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 4,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        3: {
            "ID": 3,
            "TYPE": "Material",
            "NAME": "MAT 3",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        4: {
            "ID": 4,
            "TYPE": "WIP",
            "NAME": "WIP 1",
            "INIT_LEVEL": 1,
            "HOLD_COST": 1
        }
    },
    "AP3": {
        0: {
            "ID": 0,
            "TYPE": "Product",
            "NAME": "PROD",
            "CUST_ORDER_CYCLE": 7,
            "INIT_LEVEL": 0,
            "DEMAND_QUANTITY": 0,
            "HOLD_COST": 1,
            "SETUP_COST_PRO": 1,
            "DELIVERY_COST": 1,
            "DUE_DATE": 7,
            "SHORTAGE_COST_PRO": 50
        },
        1: {
            "ID": 1,
            "TYPE": "Material",
            "NAME": "MAT 1",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        2: {
            "ID": 2,
            "TYPE": "Material",
            "NAME": "MAT 2",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 4,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        3: {
            "ID": 3,
            "TYPE": "Material",
            "NAME": "MAT 3",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        4: {
            "ID": 4,
            "TYPE": "Material",
            "NAME": "MAT 4",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        5: {
            "ID": 5,
            "TYPE": "Material",
            "NAME": "MAT 5",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        6: {
            "ID": 6,
            "TYPE": "WIP",
            "NAME": "WIP 1",
            "INIT_LEVEL": 1,
            "HOLD_COST": 1
        },
        7: {
            "ID": 7,
            "TYPE": "WIP",
            "NAME": "WIP 2",
            "INIT_LEVEL": 1,
            "HOLD_COST": 1
        }
    },
    "AP4": {
        0: {
            "ID": 0,
            "TYPE": "Product",
            "NAME": "PROD",
            "CUST_ORDER_CYCLE": 7,
            "INIT_LEVEL": 0,
            "DEMAND_QUANTITY": 0,
            "HOLD_COST": 1,
            "SETUP_COST_PRO": 1,
            "DELIVERY_COST": 1,
            "DUE_DATE": 7,
            "SHORTAGE_COST_PRO": 200 # default 100
        },
        1: {
            "ID": 1,
            "TYPE": "Material",
            "NAME": "MAT 1",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        2: {
            "ID": 2,
            "TYPE": "Material",
            "NAME": "MAT 2",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        3: {
            "ID": 3,
            "TYPE": "Material",
            "NAME": "MAT 3",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        4: {
            "ID": 4,
            "TYPE": "Material",
            "NAME": "MAT 4",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        5: {
            "ID": 5,
            "TYPE": "Material",
            "NAME": "MAT 5",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        6: {
            "ID": 6,
            "TYPE": "Material",
            "NAME": "MAT 6",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        7: {
            "ID": 7,
            "TYPE": "Material",
            "NAME": "MAT 7",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        8: {
            "ID": 8,
            "TYPE": "Material",
            "NAME": "MAT 8",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 2,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        9: {
            "ID": 9,
            "TYPE": "Material",
            "NAME": "MAT 9",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        10: {
            "ID": 10,
            "TYPE": "Material",
            "NAME": "MAT 10",
            "MANU_ORDER_CYCLE": 1,
            "INIT_LEVEL": 1,
            "SUP_LEAD_TIME": 2,
            "HOLD_COST": 1,
            "PURCHASE_COST": 2,
            "ORDER_COST_TO_SUP": 1,
            "LOT_SIZE_ORDER": 0
        },
        11: {
            "ID": 11,
            "TYPE": "WIP",
            "NAME": "WIP 1",
            "INIT_LEVEL": 1,
            "HOLD_COST": 1
        },
        12: {
            "ID": 12,
            "TYPE": "WIP",
            "NAME": "WIP 2",
            "INIT_LEVEL": 1,
            "HOLD_COST": 1
        }
        ,
        13: {
            "ID": 13,
            "TYPE": "WIP",
            "NAME": "WIP 3",
            "INIT_LEVEL": 1,
            "HOLD_COST": 1
        },
        14: {
            "ID": 14,
            "TYPE": "WIP",
            "NAME": "WIP 4",
            "INIT_LEVEL": 1,
            "HOLD_COST": 1
        }
    }
}

P = {
    "AP1": {
        0: {
            "ID": 0,
            "PRODUCTION_RATE": 2,
            "INPUT_TYPE_LIST": [I["AP1"][1]],
            "QNTY_FOR_INPUT_ITEM": [1],
            "OUTPUT": I["AP1"][0],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        }
    },
    "AP2": {
        0: {
            "ID": 0,
            "PRODUCTION_RATE": 2,
            "INPUT_TYPE_LIST": [I["AP2"][1], I["AP2"][2]],
            "QNTY_FOR_INPUT_ITEM": [1, 1],
            "OUTPUT": I["AP2"][4],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        },
        1: {
            "ID": 1,
            "PRODUCTION_RATE": 2,
            "INPUT_TYPE_LIST": [I["AP2"][2], I["AP2"][3], I["AP2"][4]],
            "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
            "OUTPUT": I["AP2"][0],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        }
    },
    "AP3": {
        0: {
            "ID": 0,
            "PRODUCTION_RATE": 2,
            "INPUT_TYPE_LIST": [I["AP3"][1], I["AP3"][2]],
            "QNTY_FOR_INPUT_ITEM": [1, 1],
            "OUTPUT": I["AP3"][6],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        },
        1: {
            "ID": 1,
            "PRODUCTION_RATE": 2,
            "INPUT_TYPE_LIST": [I["AP3"][2], I["AP3"][3], I["AP3"][6]],
            "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
            "OUTPUT": I["AP3"][7],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        },
        2: {
            "ID": 2,
            "PRODUCTION_RATE": 2,
            "INPUT_TYPE_LIST": [I["AP3"][4], I["AP3"][5], I["AP3"][7]],
            "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
            "OUTPUT": I["AP3"][0],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        }
    },
     "AP4": {
        0: {
            "ID": 0,
            "PRODUCTION_RATE": 12,
            "INPUT_TYPE_LIST": [I["AP4"][1], I["AP4"][2]],
            "QNTY_FOR_INPUT_ITEM": [1, 1],
            "OUTPUT": I["AP4"][11],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        },
        1: {
            "ID": 1,
            "PRODUCTION_RATE": 12,
            "INPUT_TYPE_LIST": [I["AP4"][2], I["AP4"][3], I["AP4"][11]],
            "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
            "OUTPUT": I["AP4"][12],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        },
        2: {
            "ID": 2,
            "PRODUCTION_RATE": 12,
            "INPUT_TYPE_LIST": [I["AP4"][4], I["AP4"][5], I["AP4"][12]],
            "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
            "OUTPUT": I["AP4"][13],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        },
        3: {
            "ID": 3,
            "PRODUCTION_RATE": 12,
            "INPUT_TYPE_LIST": [I["AP4"][6], I["AP4"][7], I["AP4"][8], I["AP4"][13]],
            "QNTY_FOR_INPUT_ITEM": [1, 1, 1, 1],
            "OUTPUT": I["AP4"][14],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        },
        4: {
            "ID": 4,
            "PRODUCTION_RATE": 12,
            "INPUT_TYPE_LIST": [I["AP4"][8], I["AP4"][9], I["AP4"][10], I["AP4"][14]],
            "QNTY_FOR_INPUT_ITEM": [1, 1, 1, 1],
            "OUTPUT": I["AP4"][0],
            "PROCESS_COST": 1,
            "PROCESS_STOP_COST": 2
        }
    }
}


# Options for RL states
DAILY_CHANGE = 0  # 0: False / 1: True
INTRANSIT = 1  # 0: False / 1: True

# Create demand
def DEMAND_QTY_FUNC(scenario):
    # Uniform distribution
    if scenario["Dist_Type"] == "UNIFORM":
        return random.randint(scenario['min'], scenario["max"])
    # Gaussian distribution
    elif scenario["Dist_Type"] == "GAUSSIAN":
        # Gaussian distribution
        demand = round(np.random.normal(scenario['mean'], scenario['std']))
        if demand < 0:
            return 1
        elif demand > INVEN_LEVEL_MAX:
            return INVEN_LEVEL_MAX
        else:
            return demand

def SUP_LEAD_TIME_FUNC(lead_time_dict):
    if lead_time_dict["Dist_Type"] == "UNIFORM":
        # Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
        # SUP_LEAD_TIME must be an integer
        return random.randint(lead_time_dict['min'], lead_time_dict['max'])
    elif lead_time_dict["Dist_Type"] == "GAUSSIAN":
        mean = lead_time_dict['mean']
        std = lead_time_dict['std']
        # Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
        lead_time = np.random.normal(mean, std)
        if lead_time < 0:
            lead_time = 0
        elif lead_time > 7:
            lead_time = 7
        # SUP_LEAD_TIME must be an integer
        return int(round(lead_time))

# Validation
# State space
# if this is not 0, the length of state space of demand quantity is not identical to INVEN_LEVEL_MAX
INVEN_LEVEL_MIN = 0
INVEN_LEVEL_MAX = 50  # Capacity limit of the inventory [units]

# Simulation
SIM_TIME = 200 # Default: 200 [days] per episode

# Print logs
PRINT_SIM = False
ASSEMBLY_PROCESS = "AP4"
SSPOLICY = False
STATIONARY = True
NUM_OF_TEST = 1
if ASSEMBLY_PROCESS == "AP1":
    MAT_COUNT = 1
elif ASSEMBLY_PROCESS == "AP2":
    MAT_COUNT = 3
elif ASSEMBLY_PROCESS == "AP3":
    MAT_COUNT = 5
elif ASSEMBLY_PROCESS == "AP4":
    MAT_COUNT = 10
# PRINT_LOG_TIMESTEP = True
# PRINT_LOG_DAILY_REPORT = True

TIME_CORRECTION = 0.0001