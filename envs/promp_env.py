from gym.core import Env
import numpy as np
import random
from envs.config_SimPy import *
from gym import spaces
import numpy as np
from envs.config_RL import * 
# import environment as env # for Meta-learning
import envs.environment as env #for baseline_test
import torch
from envs.log_SimPy import *
from envs.log_RL import *
import matplotlib.pyplot as plt
from envs.scenarios import *
from collections import Counter
import pandas as pd

class MetaEnv(Env):
    """
    Wrapper around OpenAI gym environments, interface for meta learning
    """
    def __init__(self):
        self.actions = []

        self.all_tasks = create_scenarios()
        #print("Tensorboard Directory: :", TENSORFLOW_LOGS)
        super(MetaEnv, self).__init__()

        self.scenario = None

        self.shortages = 0
        self.total_reward = 0

        # Record the cumulative value of each cost
        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        
        os = []
        # Define action space
        self.action_space = spaces.Box(low = 0, high = 10, shape = (MAT_COUNT, 1), dtype = int)
        # if self.scenario["Dist_Type"] == "UNIFORM":
        #    k = INVEN_LEVEL_MAX*2+(self.scenario["max"]+1)

        # DAILY_CHANGE + INTRANSIT + REMAINING_DEMAND
        os = spaces.Box(low = 0, high = INVEN_LEVEL_MAX*2+1, shape=(len(I[ASSEMBLY_PROCESS])*(1+DAILY_CHANGE)+MAT_COUNT*INTRANSIT+1,1), dtype=int)
        
        '''
        - Inventory Level of Product
        - Daily Change of Product
        - Inventory Level of WIP
        - Daily Change of WIP
        - Inventory Level of Material
        - Daily Change of Material
        - Demand - Inventory Level of Product
        '''
        self.observation_space = os
        self.task = None
    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        print("all tasks reset")
        self.tasks = random.sample(self.all_tasks, n_tasks)
        self.all_tasks = create_scenarios()
        return self.tasks

    def set_task(self, task):
        #print("Task_Setted: ", task)
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        self.scenario = task

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return self.scenario

    def reset(self):
        # Initialize the total reward for the episode
        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        # Initialize the simulation environment
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
            I, P, DAILY_EVENTS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                                  self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I, self.scenario)
        env.update_daily_report(self.inventoryList)

        state_real = self.get_current_state()
        STATE_DICT.clear()
        DAILY_REPORTS.clear()
        self.actions = []
        return state_real

    def step(self, action):
        self.actions.append(action)
        
        # Update the action of the agent
        test_actions = []
        # Update the action of the agent
        if RL_ALGORITHM == "PPO":
            i = 0
            for _ in range(len(I[ASSEMBLY_PROCESS])):
                if I[ASSEMBLY_PROCESS][_]["TYPE"] == "Material":
                    # Set action as predicted value
                    I[ASSEMBLY_PROCESS][_]["LOT_SIZE_ORDER"] = min(max(np.round(action[i]),0),10) # 양수 상한
                    i += 1
                    test_actions.append(I[ASSEMBLY_PROCESS][_]['LOT_SIZE_ORDER'])

        elif RL_ALGORITHM == "DQN":
            pass

        # Capture the current state of the environment
        # current_state = env.cap_current_state(self.inventoryList)
        # Run the simulation for 24 hours (until the next day)
        # Action append

        self.simpy_env.run(until=self.simpy_env.now + 24)
        env.update_daily_report(self.inventoryList)
        # Capture the next state of the environment
        state_real = self.get_current_state()
        # Set the next state
        next_state = state_real
        # Calculate the total cost of the day
        cost = env.Cost.update_cost_log(self.inventoryList)
        # Cost Dict update
        for key in DAILY_COST_REPORT.keys():
            self.cost_dict[key] += DAILY_COST_REPORT[key]

        env.Cost.clear_cost()
        reward = -cost
        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0

        if PRINT_SIM:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "PPO":
                i = 0
                for _ in range(len(I)):
                    if I[ASSEMBLY_PROCESS][_]["TYPE"] == "Raw Material":
                        print(
                            f"[Order Quantity for {I[ASSEMBLY_PROCESS][_]['NAME']}] ", action[i])
                        i += 1
            # SimPy simulation print
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", -reward)
            print("Total cost: ", -self.total_reward)
            print("[REAL_STATE for the next round] ",  [
                    item for item in next_state])

        self.daily_events.clear()

        # Check if the simulation is done
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료

        info = {}  # 추가 정보 (필요에 따라 사용)

        return next_state, reward, done, self.cost_dict
    
    def get_current_state(self):
        # Make State for RL
        state = []
        # Update STATE_ACTION_REPORT_REAL
        for id in range(len(I[ASSEMBLY_PROCESS])):
            # ID means Item_ID, 7 means to the length of the report for one item
            # append On_Hand_inventory
            state.append(
                STATE_DICT[-1][f"On_Hand_{I[ASSEMBLY_PROCESS][id]['NAME']}"]+INVEN_LEVEL_MAX)
            # append changes in inventory
            if DAILY_CHANGE == 1:
                # append changes in inventory
                state.append(
                    STATE_DICT[-1][f"Daily_Change_{I[ASSEMBLY_PROCESS][id]['NAME']}"]+INVEN_LEVEL_MAX)
            if INTRANSIT == 1:
                if I[ASSEMBLY_PROCESS][id]["TYPE"] == "Material":
                    # append Intransition inventory
                    state.append(
                        STATE_DICT[-1][f"In_Transit_{I[ASSEMBLY_PROCESS][id]['NAME']}"])
        '''
        # Append remaining demand
        print(I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"] -
                     self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)
        '''
        state.append(I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"] -
                     self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)
        
        return state
    def report_scalar(self, paths):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        print("="*20)
        rewards = 0
        for task_idx in range(len(paths)):
            for idx in range(len(paths[task_idx]['rewards'])):
                rewards += sum(paths[task_idx]['rewards'][idx])
        average = rewards/(len(paths)*len(paths[0]["rewards"]))
        #print("Reward:", average)
        print("="*20)

        '''
        if now+1 == goal:
            bins = np.arange(-2, 6, 0.1)  # 11까지 해야 10이 포함됨
            plt.hist(hist_path, bins=bins, edgecolor='black')
            plt.title('Histogram (-2 to 6, bin=0.1)')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()
        '''
        cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        
        for task_idx in range(len(paths)):
            for idx in range(len(paths[task_idx]['env_infos'])):
                for key in self.cost_dict.keys():
                    cost_dict[key] = cost_dict[key] + paths[task_idx]['env_infos'][idx][key][-1]/(len(paths)*len(paths[0]["rewards"]))
        '''
        action_datas = {
            "Mat1-I": [min(max(0,np.round(action[0])),5) for action in paths[0]['actions']],
            "Mat1-R":[action[0] for action in paths[0]['actions']],
            "Mat2-I": [min(max(0,np.round(action[1])),5) for action in paths[0]['actions']],
            "Mat2-R":[action[1] for action in paths[0]['actions']],
            "Mat3-I": [min(max(0,np.round(action[2])),5) for action in paths[0]['actions']],
            "Mat3-R":[action[2] for action in paths[0]['actions']],
            "Mat4-I": [min(max(0,np.round(action[3])),5) for action in paths[0]['actions']],
            "Mat4-R":[action[3] for action in paths[0]['actions']],
            "Mat5-I": [min(max(0,np.round(action[4])),5) for action in paths[0]['actions']],
            "Mat5-R":[action[4] for action in paths[0]['actions']],
        }
        '''
        return average, cost_dict, None
       
 
    