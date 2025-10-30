import numpy as np
from envs.config_SimPy import *
from envs.promp_env import MetaEnv
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
import torch
import torch.optim as optim
from envs.config_folders import *
def main(params):

    env = MetaEnv()
    mlp = SimpleMLP(np.prod(env.observation_space.shape), 
                       np.prod(env.action_space.shape),
                    hidden_layers = params["Layers"])
    
    agent = MetaGaussianAgent(mlp = mlp, num_tasks=params["num_task"], learn_std = params["learn_std"])
    
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
                    device=params["device"])
    
    meta_algo.learn(params["epochs"])
    torch.save(meta_algo.state_dict(), f"{SAVED_MODEL_PATH}/saved_model")
if __name__ == "__main__":
    params = {
        "Layers":[64, 64, 64], # layers of Network
        "rollout_per_task": 20,
        "num_task": 5, # Number of tasks
        "max_path_length": SIM_TIME,
        "tensor_log": TENSORFLOW_LOGS,
        "alpha": 0.002,
        "beta": 0.0005,
        "outer_iters": 5, # number of ProMp steps without re-sampling
        "clip_eps": 0.3, # clip range for ProMP(outer) update
        "num_inner_grad": 1,
        "epochs": 500,
        "discount": 0.99,
        "gae_lambda": 1,
        "parallel": True,
        "learn_std": True,
        "device":torch.device("cpu")
    }
    main(params)
   