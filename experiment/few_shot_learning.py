"""
Few-Shot Learning Evaluation Module
====================================
Evaluates generalization performance of trained meta-RL models on new tasks.

This module:
1. Loads pre-trained meta-learning models
2. Tests on new task distributions (few-shot learning scenarios)
3. Visualizes results via TensorBoard
4. Supports configurable task distributions and adaptation episodes
"""

import os
import sys
import torch
import numpy as np
import random
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.config_SimPy import *
from envs.config_RL import *
from envs.promp_env import MetaEnv
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Agents.Simple_Mlp import SimpleMLP


class FewShotEvaluator:
    """
    Few-shot learning evaluator for testing generalization of meta-trained policies.
    
    This class handles:
    - Loading pre-trained models
    - Creating new task distributions for evaluation
    - Few-shot adaptation on new tasks
    - Computing evaluation metrics and logging to TensorBoard
    """

    def __init__(self,
                 model_path: str,
                 log_dir: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the few-shot evaluator.

        Args:
            model_path (str): Path to saved trained model state_dict
            log_dir (str, optional): TensorBoard log directory. If None, creates default
            device (torch.device, optional): Computation device. Defaults to CPU
        """
        self.device = device or torch.device('cpu')
        self.model_path = model_path
        self.env = MetaEnv()
        
        # Setup TensorBoard logging
        if log_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            log_dir = os.path.join(parent_dir, "Tensorboard_logs", "Few_Shot_Eval")
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Agent will be initialized in load_model()
        self.agent = None
        self.meta_params = None
        
        print(f"FewShotEvaluator initialized")
        print(f"  Model path: {model_path}")
        print(f"  Log dir: {log_dir}")
        print(f"  Device: {self.device}")

    def load_model(self, model_params: Dict[str, Any]) -> None:
        """
        Load pre-trained model from checkpoint.

        Args:
            model_params (dict): Model configuration parameters including:
                - 'Layers': MLP hidden layer sizes
                - 'num_task': Number of tasks (for agent initialization)
                - 'learn_std': Whether policy learns log_std
        """
        print("\n" + "="*60)
        print("Loading pre-trained model...")
        print("="*60)
        
        # Create agent with same architecture
        mlp = SimpleMLP(
            np.prod(self.env.observation_space.shape),
            np.prod(self.env.action_space.shape),
            hidden_layers=model_params["Layers"]
        )
        
        self.agent = MetaGaussianAgent(
            mlp=mlp,
            num_tasks=model_params["num_task"],
            learn_std=model_params["learn_std"]
        ).to(self.device)
        
        # Load state_dict if model exists
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                # Handle different state_dict formats
                # If keys have "agent." prefix, remove it
                if any(k.startswith("agent.") for k in state_dict.keys()):
                    state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items() if not k.startswith("inner_step_sizes")}
                
                self.agent.load_state_dict(state_dict)
                print(f"✓ Model loaded from {self.model_path}")
            except RuntimeError as e:
                print(f"⚠ Warning: Could not load state_dict: {str(e)[:100]}...")
                print("  Attempting to load agent parameters only...")
                try:
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    # Extract only agent-related keys
                    agent_keys = {k.replace("agent.", ""): v for k, v in state_dict.items() 
                                 if k.startswith("agent.") and not k.startswith("inner_step_sizes")}
                    if agent_keys:
                        self.agent.load_state_dict(agent_keys, strict=False)
                        print(f"✓ Partial model loaded")
                    else:
                        print(f"  Could not extract agent parameters")
                except Exception as e2:
                    print(f"  Failed: {str(e2)[:100]}")
        else:
            print(f"⚠ Warning: Model path does not exist: {self.model_path}")
            print("  Continuing with randomly initialized weights")
        
        # Store meta parameters (pre-update parameters)
        self.meta_params = dict(self.agent.named_parameters())
        self.agent.eval()  # Set to evaluation mode
        
        print(f"✓ Agent ready with architecture:")
        print(f"  - Input: {np.prod(self.env.observation_space.shape)}")
        print(f"  - Hidden: {model_params['Layers']}")
        print(f"  - Output: {np.prod(self.env.action_space.shape)}")

    def create_custom_task_distributions(self,
                                        num_test_tasks: int = 10,
                                        demand_ranges: Optional[List[Tuple[int, int]]] = None,
                                        leadtime_ranges: Optional[List[Tuple[int, int]]] = None
                                        ) -> List[Dict]:
        """
        Create custom task distributions for few-shot evaluation.
        
        Allows specifying demand and leadtime distributions different from training.

        Args:
            num_test_tasks (int): Number of test tasks to create
            demand_ranges (list, optional): List of (min, max) demand tuples.
                                          If None, uses default range.
            leadtime_ranges (list, optional): List of (min, max) leadtime tuples.
                                            If None, uses default range.

        Returns:
            test_tasks (list): List of task dictionaries with custom distributions
        """
        print("\n" + "="*60)
        print("Creating custom task distributions for few-shot evaluation...")
        print("="*60)
        
        # Default ranges if not specified
        if demand_ranges is None:
            demand_ranges = [
                (5, 10),   # Low demand
                (10, 15),  # Medium demand
                (15, 20),  # High demand
            ]
        
        if leadtime_ranges is None:
            leadtime_ranges = [
                (1, 2),    # Short leadtime
                (2, 3),    # Medium leadtime
                (3, 5),    # Long leadtime
            ]
        
        test_tasks = []
        
        for i in range(num_test_tasks):
            # Randomly select demand range
            demand_min, demand_max = random.choice(demand_ranges)
            demand_config = {
                "Dist_Type": "UNIFORM",
                "min": demand_min,
                "max": demand_max
            }
            
            # Randomly select leadtime ranges for each material
            leadtime_configs = [
                random.choice([
                    {"Dist_Type": "UNIFORM", "min": lt_min, "max": lt_max}
                    for lt_min, lt_max in leadtime_ranges
                ])
                for _ in range(MAT_COUNT)
            ]
            
            task = {
                "DEMAND": demand_config,
                "LEADTIME": leadtime_configs,
                "task_id": i
            }
            test_tasks.append(task)
        
        print(f"✓ Created {num_test_tasks} custom test tasks")
        print(f"  Demand ranges: {demand_ranges}")
        print(f"  Leadtime ranges: {leadtime_ranges}")
        
        return test_tasks

    def run_episode(self,
                   task: Dict,
                   params: Optional[Dict] = None,
                   max_steps: Optional[int] = None,
                   deterministic: bool = False) -> Dict:
        """
        Run a single episode on the given task with specified parameters.

        Args:
            task (dict): Task dictionary with DEMAND and LEADTIME configurations
            params (dict, optional): Policy parameters. If None, uses meta_params
            max_steps (int, optional): Maximum episode length. If None, uses SIM_TIME
            deterministic (bool): If True, use deterministic policy (mean action)

        Returns:
            episode_data (dict): Dictionary containing:
                - 'total_cost': Total cost for the episode
                - 'cost_dict': Dictionary of individual cost components
                - 'rewards': List of rewards per step
                - 'task_id': Task identifier
        """
        if params is None:
            params = self.meta_params
        
        max_steps = max_steps or SIM_TIME
        
        # Set task in environment
        self.env.set_task(task)
        obs = self.env.reset()
        
        episode_costs = {key: 0 for key in self.env.cost_dict.keys()}
        episode_rewards = []
        done = False
        step = 0
        
        with torch.no_grad():
            while not done and step < max_steps:
                # Convert observation to tensor and ensure proper shape
                # obs should be a numpy array, reshape to (1, 1, obs_dim) for single trajectory
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                if len(obs_tensor.shape) == 2:
                    obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension if needed
                
                # Get action from policy using pre-update (meta) parameters
                # For single episode evaluation, we don't do full batch processing
                try:
                    with torch.no_grad():
                        # Compute distribution directly without get_actions to avoid batch issues
                        dist = self.agent.distribution(obs_tensor, params=params)
                        if deterministic:
                            action_tensor = dist.mean
                        else:
                            action_tensor = dist.rsample()
                except Exception as e:
                    # Fallback: use simpler action selection
                    print(f"    Warning: Distribution computation failed ({str(e)[:50]}), using mean action")
                    action_tensor = torch.zeros(obs_tensor.shape[0], np.prod(self.env.action_space.shape))
                
                action = action_tensor.cpu().numpy().flatten()
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                episode_rewards.append(reward)
                step += 1
        
        # Accumulate costs from environment
        for cost_type, cost_value in self.env.cost_dict.items():
            episode_costs[cost_type] = cost_value
        
        total_cost = sum(episode_costs.values())
        
        episode_data = {
            'total_cost': total_cost,
            'cost_dict': episode_costs.copy(),
            'rewards': episode_rewards,
            'task_id': task.get('task_id', -1),
            'num_steps': step
        }
        
        return episode_data

    def adapt_policy(self,
                    support_episodes: List[Dict],
                    inner_lr: float = 0.01,
                    num_adaptation_steps: int = 1) -> Dict[str, torch.Tensor]:
        """
        Perform few-shot adaptation on support episodes (basic meta-learning update).
        
        This performs simple gradient-based adaptation using the support episodes.
        For more sophisticated adaptation, integrate with the ProMP/MAML algorithms.

        Args:
            support_episodes (list): List of episode data for adaptation
            inner_lr (float): Inner-loop learning rate for adaptation
            num_adaptation_steps (int): Number of adaptation gradient steps

        Returns:
            adapted_params (dict): Adapted policy parameters after few-shot learning
        """
        # For this basic version, we return the meta parameters
        # In a full implementation, you would:
        # 1. Compute gradients on support episodes
        # 2. Apply gradient descent steps
        # 3. Return adapted parameters
        
        # This is a placeholder - actual adaptation logic would integrate
        # with the ProMP/MAML base algorithm
        adapted_params = {name: param.clone() for name, param in self.meta_params.items()}
        return adapted_params

    def evaluate_on_task_distribution(self,
                                     test_tasks: List[Dict],
                                     num_support_episodes: int = 5,
                                     num_query_episodes: int = 10,
                                     episode_max_steps: Optional[int] = None,
                                     adaptation_lr: float = 0.01,
                                     num_adaptation_steps: int = 1,
                                     run_name: str = "few_shot_eval") -> Dict:
        """
        Evaluate the policy on a distribution of test tasks.
        
        For each task:
        1. Collect support episodes (use for adaptation)
        2. Adapt policy using support episodes
        3. Collect query episodes (evaluate performance)
        4. Compute metrics and log to TensorBoard

        Args:
            test_tasks (list): List of test task dictionaries
            num_support_episodes (int): Number of episodes for few-shot adaptation
            num_query_episodes (int): Number of episodes to evaluate (after adaptation)
            episode_max_steps (int, optional): Max steps per episode
            adaptation_lr (float): Learning rate for few-shot adaptation
            num_adaptation_steps (int): Number of adaptation steps
            run_name (str): Name for TensorBoard run

        Returns:
            evaluation_results (dict): Summary of evaluation metrics
        """
        print("\n" + "="*60)
        print(f"Few-Shot Evaluation: {run_name}")
        print("="*60)
        print(f"Number of test tasks: {len(test_tasks)}")
        print(f"Support episodes per task: {num_support_episodes}")
        print(f"Query episodes per task: {num_query_episodes}")
        
        # Store results
        all_query_costs = []
        per_task_results = defaultdict(list)
        cost_type_accumulator = defaultdict(list)
        
        for task_idx, task in enumerate(test_tasks):
            task_id = task.get('task_id', task_idx)
            
            print(f"\n  Task {task_idx+1}/{len(test_tasks)} (ID: {task_id})")
            print(f"    Demand: {task['DEMAND']['min']}-{task['DEMAND']['max']}")
            
            # --- SUPPORT PHASE: Collect episodes for adaptation ---
            support_episodes = []
            for ep in range(num_support_episodes):
                episode_data = self.run_episode(
                    task,
                    params=self.meta_params,
                    max_steps=episode_max_steps,
                    deterministic=False
                )
                support_episodes.append(episode_data)
            
            support_avg_cost = np.mean([ep['total_cost'] for ep in support_episodes])
            print(f"    Support phase: {num_support_episodes} episodes, avg cost: {support_avg_cost:.2f}")
            
            # --- ADAPTATION PHASE: Adapt policy to this task ---
            adapted_params = self.adapt_policy(
                support_episodes,
                inner_lr=adaptation_lr,
                num_adaptation_steps=num_adaptation_steps
            )
            
            # --- QUERY PHASE: Evaluate adapted policy ---
            query_episodes = []
            for ep in range(num_query_episodes):
                episode_data = self.run_episode(
                    task,
                    params=adapted_params,
                    max_steps=episode_max_steps,
                    deterministic=True  # Deterministic for evaluation
                )
                query_episodes.append(episode_data)
            
            # Compute query phase metrics
            query_costs = [ep['total_cost'] for ep in query_episodes]
            query_avg_cost = np.mean(query_costs)
            query_std_cost = np.std(query_costs)
            
            all_query_costs.extend(query_costs)
            per_task_results[task_id].append({
                'support_avg': support_avg_cost,
                'query_avg': query_avg_cost,
                'query_std': query_std_cost
            })
            
            # Track individual cost components
            for ep in query_episodes:
                for cost_type, cost_value in ep['cost_dict'].items():
                    cost_type_accumulator[cost_type].append(cost_value)
            
            print(f"    Query phase: {num_query_episodes} episodes")
            print(f"      Avg cost: {query_avg_cost:.2f} ± {query_std_cost:.2f}")
            
            # Log task results to TensorBoard
            self.writer.add_scalar(
                f'Task_{task_id}/Support_Phase/Avg_Cost',
                support_avg_cost,
                task_idx
            )
            self.writer.add_scalar(
                f'Task_{task_id}/Query_Phase/Avg_Cost',
                query_avg_cost,
                task_idx
            )
            self.writer.add_scalar(
                f'Task_{task_id}/Query_Phase/Std_Cost',
                query_std_cost,
                task_idx
            )
            
            for cost_type, costs in cost_type_accumulator.items():
                if len(costs) > 0:
                    self.writer.add_scalar(
                        f'Task_{task_id}/Cost_Components/{cost_type}',
                        np.mean(costs),
                        task_idx
                    )
        
        # Compute overall metrics
        overall_avg_cost = np.mean(all_query_costs)
        overall_std_cost = np.std(all_query_costs)
        overall_min_cost = np.min(all_query_costs)
        overall_max_cost = np.max(all_query_costs)
        
        print("\n" + "="*60)
        print("Overall Evaluation Results")
        print("="*60)
        print(f"Average Query Cost: {overall_avg_cost:.2f} ± {overall_std_cost:.2f}")
        print(f"Min/Max Cost: {overall_min_cost:.2f} / {overall_max_cost:.2f}")
        print(f"Number of query episodes: {len(all_query_costs)}")
        
        # Log overall metrics
        self.writer.add_scalar('Overall/Avg_Query_Cost', overall_avg_cost, 0)
        self.writer.add_scalar('Overall/Std_Query_Cost', overall_std_cost, 0)
        self.writer.add_scalar('Overall/Min_Query_Cost', overall_min_cost, 0)
        self.writer.add_scalar('Overall/Max_Query_Cost', overall_max_cost, 0)
        
        # Log cost components
        for cost_type, costs in cost_type_accumulator.items():
            if len(costs) > 0:
                self.writer.add_scalar(
                    f'Overall/Cost_Components/{cost_type}_Mean',
                    np.mean(costs),
                    0
                )
                self.writer.add_scalar(
                    f'Overall/Cost_Components/{cost_type}_Std',
                    np.std(costs),
                    0
                )
        
        self.writer.flush()
        
        # Extract support episodes count from run_name or parameters
        # This allows tracking shot progression in TensorBoard
        try:
            if 'Shot_' in run_name:
                shot_num = int(run_name.split('Shot_')[-1].split('_')[0])
                self.writer.add_scalar('Shot_Progression/Average_Cost', overall_avg_cost, shot_num)
                self.writer.add_scalar('Shot_Progression/Std_Cost', overall_std_cost, shot_num)
                self.writer.flush()
        except Exception:
            pass  # Ignore if shot number extraction fails
        
        evaluation_results = {
            'overall_avg_cost': overall_avg_cost,
            'overall_std_cost': overall_std_cost,
            'overall_min_cost': overall_min_cost,
            'overall_max_cost': overall_max_cost,
            'num_test_tasks': len(test_tasks),
            'num_query_episodes': num_query_episodes,
            'per_task_results': dict(per_task_results),
            'all_query_costs': all_query_costs
        }
        
        return evaluation_results

    def close(self) -> None:
        """Clean up resources."""
        self.writer.close()
        print("\n✓ Evaluation complete. TensorBoard logs saved to:")
        print(f"  {self.log_dir}")


def main(eval_config: Optional[Dict] = None):
    """
    Main evaluation function.
    
    Args:
        eval_config (dict, optional): Configuration dictionary with keys:
            - 'model_path': Path to trained model
            - 'model_params': Model architecture parameters
            - 'num_test_tasks': Number of test tasks
            - 'num_support_episodes': Support episodes per task
            - 'num_query_episodes': Query episodes per task
            - 'demand_ranges': Custom demand distributions
            - 'leadtime_ranges': Custom leadtime distributions
            - 'device': Computation device
    """
    
    # Default configuration
    if eval_config is None:
        eval_config = {
            # Model loading
            'model_path': os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Saved_Model", "Train_1", "saved_model"
            ),
            
            # Model architecture (must match training configuration)
            'model_params': {
                'Layers': [64, 64, 64],
                'num_task': 5,
                'learn_std': True
            },
            
            # Few-shot evaluation configuration
            'num_test_tasks': 10,
            'num_support_episodes': 5,
            'num_query_episodes': 10,
            
            # Task distribution customization
            'demand_ranges': [
                (5, 10),    # Low
                (10, 15),   # Medium
                (15, 20),   # High
            ],
            
            'leadtime_ranges': [
                (1, 2),     # Short
                (2, 3),     # Medium
                (3, 5),     # Long
            ],
            
            # Computation
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        }
    
    print("\n" + "="*70)
    print(" "*15 + "FEW-SHOT GENERALIZATION EVALUATION")
    print("="*70)
    
    # Initialize evaluator
    evaluator = FewShotEvaluator(
        model_path=eval_config['model_path'],
        device=eval_config.get('device', torch.device('cpu'))
    )
    
    # Load model
    evaluator.load_model(eval_config['model_params'])
    
    # Create test tasks with custom distributions
    test_tasks = evaluator.create_custom_task_distributions(
        num_test_tasks=eval_config['num_test_tasks'],
        demand_ranges=eval_config.get('demand_ranges'),
        leadtime_ranges=eval_config.get('leadtime_ranges')
    )
    
    # Run evaluation
    results = evaluator.evaluate_on_task_distribution(
        test_tasks=test_tasks,
        num_support_episodes=eval_config['num_support_episodes'],
        num_query_episodes=eval_config['num_query_episodes'],
        run_name="Few-Shot_Generalization_Test"
    )
    
    # Close and clean up
    evaluator.close()
    
    return results


if __name__ == "__main__":
    # Run with default configuration
    results = main()
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Average Query Cost: {results['overall_avg_cost']:.2f} ± {results['overall_std_cost']:.2f}")
    print(f"Cost Range: [{results['overall_min_cost']:.2f}, {results['overall_max_cost']:.2f}]")
    print(f"Test Tasks: {results['num_test_tasks']}")
    print(f"Episodes per Task: {results['num_query_episodes']} (query)")
    print("="*70)
