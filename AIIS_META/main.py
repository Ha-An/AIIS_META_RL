import os
import sys
from pathlib import Path
import numpy as np
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.config_SimPy import *
from envs.promp_env import MetaEnv
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Agents.Categorical.Meta_Categorical import MetaCategoricalAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Algos.MAML.maml import VPG_MAML
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
import torch
import torch.optim as optim
from envs.config_folders import *
import time
def main(params):

    env = MetaEnv()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    policy_dist = params.get("policy_dist", "gaussian").lower()
    if policy_dist == "categorical":
        from envs.config_RL import ACTION_SPACE
        num_actions = len(ACTION_SPACE)
        mlp = SimpleMLP(obs_dim, act_dim * num_actions, hidden_layers=params["Layers"])
        agent = MetaCategoricalAgent(
            mlp=mlp,
            num_tasks=params["num_task"],
            action_dim=act_dim,
            num_actions=num_actions,
        )
    else:
        mlp = SimpleMLP(obs_dim, act_dim, hidden_layers=params["Layers"])
        agent = MetaGaussianAgent(mlp=mlp, num_tasks=params["num_task"], learn_std=params["learn_std"])
    
    # Select algorithm based on params
    algorithm_name = params.get("algorithm", "ProMP")  # Default: ProMP
    
    if algorithm_name == "ProMP":
        meta_algo = ProMP(env = env, max_path_length = params["max_path_length"],
                        agent = agent, alpha = params["alpha"], beta = params["beta"],
                        baseline=LinearFeatureBaseline(),
                        tensor_log=params["tensor_log"],
                        inner_grad_steps= params["num_inner_grad"],
                        num_tasks=params["num_task"], 
                        outer_iters=params["outer_iters"], 
                        parallel=params["parallel"], 
                        rollout_per_task=params["rollout_per_task"], 
                        clip_eps=params["clip_eps"], 
                        device=params["device"],
                        action_log_interval=params["action_log_interval"])
    elif algorithm_name == "VPG_MAML":
        meta_algo = VPG_MAML(env = env, max_path_length = params["max_path_length"],
                        agent = agent, alpha = params["alpha"], beta = params["beta"],
                        baseline=LinearFeatureBaseline(),
                        tensor_log=params["tensor_log"],
                        inner_grad_steps= params["num_inner_grad"],
                        num_tasks=params["num_task"], 
                        outer_iters=params["outer_iters"], 
                        parallel=params["parallel"], 
                        rollout_per_task=params["rollout_per_task"], 
                        clip_eps=params["clip_eps"], 
                        device=params["device"],
                        action_log_interval=params["action_log_interval"])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Use 'ProMP' or 'VPG_MAML'")
    
    # Start timing
    print(f"\n{'='*60}")
    print(f"Training with {algorithm_name} algorithm...")
    print(f"{'='*60}")
    start_time = time.time()
    
    meta_algo.learn(params["epochs"])
    torch.save(meta_algo.state_dict(), f"{SAVED_MODEL_PATH}/saved_model")
    
    meta_algo.close()
    
    # End timing and report
    end_time = time.time()
    total_time = end_time - start_time
    
    # Format time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total training time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")
    print(f"{'='*60}\n")
if __name__ == "__main__":
    params = {
        # ===== Algorithm Selection =====
        "algorithm": "ProMP",  # Meta-learning algorithm: "ProMP" (PPO-style with KL penalty) or "VPG_MAML" (vanilla policy gradient)
        "policy_dist": "categorical",  # "gaussian" or "categorical"
        
        # ===== Network Architecture =====
        "Layers": [64, 64, 64],  # MLP hidden layer sizes. SimpleMLP will create: Input -> fc0(64) -> Tanh -> fc1(64) -> Tanh -> fc2(64) -> Tanh -> output_layer
        
        # ===== Task & Sampling Configuration =====
        "num_task": 10,  # Number of distinct meta-tasks sampled per epoch. MetaSampler creates num_task separate environment instances
        "rollout_per_task": 5,  # Number of trajectories (rollouts) collected per task. Total trajectories per epoch = num_task * rollout_per_task
        "max_path_length": SIM_TIME,  # Maximum timesteps per trajectory. Episode terminates when timestep >= max_path_length (set to SIM_TIME=200 from config)
        
        # ===== Meta-Learning Parameters =====
        "alpha": 0.003,  # Inner-loop learning rate (α): used for task-specific gradient descent during inner_loop adaptation
        "beta": 0.0005,  # Outer-loop (meta) learning rate (β): used by Adam optimizer to update meta-parameters θ (optimizer = optim.Adam(agent.parameters(), lr=beta))
        "num_inner_grad": 3,  # Number of inner gradient descent steps per task during inner_loop (inner adaptation steps)
        "outer_iters": 5,  # Number of outer-loop meta-updates per epoch WITHOUT re-sampling tasks. For each epoch: inner_loop once, then outer_loop runs self.outer_iters times on same task rollouts
        
        # ===== Policy Optimization =====
        "clip_eps": 0.2,  # PPO clipping epsilon: controls the clipping range in PPO loss (used only by ProMP, not VPG_MAML). Inner loop: -(ratio * A).mean() without clipping
        
        # ===== Advantage Estimation =====
        "discount": 0.99,  # Discount factor (γ): used in GAE (Generalized Advantage Estimation) to compute returns: return_t = reward_t + γ * V(s_{t+1})
        "gae_lambda": 1.0,  # GAE lambda (λ): λ=1.0 means standard return (no discounting of TD residuals), λ=0.0 means only 1-step advantage. Controls bias-variance tradeoff in advantage estimation
        
        # ===== Training Configuration =====
        "epochs": 4000,  # Total number of meta-training epochs (outer loop iterations with new task sampling). Change to 500 for full training
        "parallel": True,  # Use multiprocessing for parallel environment sampling. If False, samples sequentially (slower but less memory)
        "learn_std": True,  # Whether the policy's action standard deviation (log_std parameter) is learnable. If False, log_std remains fixed at init value
        
        # ===== Logging & Device =====
        "tensor_log": TENSORFLOW_LOGS,  # Directory path for TensorBoard event files (auto-created if not exists). Used by SummaryWriter for training visualization
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Use GPU if available
        "action_log_interval": 100  # Log action distribution every N epochs
    }
    main(params)
   
