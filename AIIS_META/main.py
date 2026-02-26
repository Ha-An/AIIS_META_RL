import os
import sys
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.config_SimPy import *
from envs.promp_env import MetaEnv
import envs.scenarios as scenario_factory
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Agents.Categorical.Meta_Categorical import MetaCategoricalAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Algos.MAML.maml import VPG_MAML
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
from AIIS_META.Sampler.meta_sampler import MetaSampler
from AIIS_META.Sampler.meta_sample_processor import MetaSampleProcessor
import torch
from envs.config_folders import *
import time


def _build_fewshot_validation_tasks(params):
    val_cfg = dict(params.get("fewshot_validation", {}))
    num_tasks = max(1, int(val_cfg.get("num_tasks", 5)))
    seed = int(val_cfg.get("seed", 2026))
    scenario_overrides = dict(val_cfg.get("scenario_overrides", {}))
    return scenario_factory.create_scenarios(
        num_scenarios=num_tasks,
        seed=seed,
        **scenario_overrides,
    )


def _evaluate_fewshot_validation(meta_algo, validation_tasks, params):
    val_cfg = dict(params.get("fewshot_validation", {}))
    adapt_steps = sorted(set(int(s) for s in val_cfg.get("adapt_steps", [0, 1, 3, 5])))
    if not adapt_steps:
        return {}, float("inf")

    rollout_per_task = int(val_cfg.get("rollout_per_task", params["rollout_per_task"]))
    max_path_length = int(val_cfg.get("max_path_length", params["max_path_length"]))
    parallel = bool(val_cfg.get("parallel", False))
    envs_per_task = int(val_cfg.get("envs_per_task", 1))
    envs_per_task = max(1, min(envs_per_task, rollout_per_task))
    num_tasks = len(validation_tasks)

    val_env = MetaEnv()
    sampler = MetaSampler(
        env=val_env,
        agent=meta_algo.agent,
        rollout_per_task=rollout_per_task,
        num_tasks=num_tasks,
        max_path_length=max_path_length,
        envs_per_task=envs_per_task,
        parallel=parallel,
    )
    sample_processor = MetaSampleProcessor(
        baseline=LinearFeatureBaseline(),
        discount=0.99,
        gae_lambda=1.0,
        normalize_adv=True,
    )

    max_step = max(adapt_steps)
    step_costs = {}

    try:
        sampler.vec_env.set_tasks(validation_tasks)
        adapted_params_list = [
            OrderedDict(meta_algo.agent.named_parameters())
            for _ in range(num_tasks)
        ]

        for step in range(max_step + 1):
            if step > 0:
                adapt_paths = sampler.obtain_samples(adapted_params_list, post_update=True)
                processed = sample_processor.process_samples(adapt_paths)
                for task_id in range(num_tasks):
                    adapted_params_list[task_id] = meta_algo._theta_prime(
                        processed[task_id],
                        params=adapted_params_list[task_id],
                        create_graph=False,
                    )

            if step not in adapt_steps:
                continue

            eval_paths = sampler.obtain_samples(adapted_params_list, post_update=True)
            rollout_costs = []
            for task_paths in eval_paths.values():
                for traj in task_paths:
                    rollout_costs.append(float(-np.sum(traj["rewards"])))
            step_costs[step] = float(np.mean(rollout_costs)) if rollout_costs else float("nan")
    finally:
        sampler.close()

    valid = [step_costs[s] for s in adapt_steps if s in step_costs and np.isfinite(step_costs[s])]
    score = float(np.mean(valid)) if valid else float("inf")
    return step_costs, score


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
                        trainable_learning_rate=False,
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
                        trainable_learning_rate=False,
                        device=params["device"],
                        action_log_interval=params["action_log_interval"])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Use 'ProMP' or 'VPG_MAML'")
    
    # Start timing
    print(f"\n{'='*60}")
    print(f"Training with {algorithm_name} algorithm...")
    print(f"{'='*60}")
    start_time = time.time()

    # Checkpoint paths
    last_ckpt_path = os.path.join(SAVED_MODEL_PATH, "saved_model")
    best_ckpt_path = os.path.join(SAVED_MODEL_PATH, "saved_model_best_fewshot")
    best_meta_path = os.path.join(SAVED_MODEL_PATH, "saved_model_best_fewshot_meta.json")

    # Held-out few-shot validation setup (for best-checkpoint selection)
    val_cfg = dict(params.get("fewshot_validation", {}))
    val_enabled = bool(val_cfg.get("enabled", False))
    best_state = {"score": float("inf"), "epoch": -1, "step_costs": {}}
    epoch_callback = None

    if val_enabled:
        validation_tasks = _build_fewshot_validation_tasks(params)
        val_interval = max(1, int(val_cfg.get("interval", 20)))
        val_steps = sorted(set(int(s) for s in val_cfg.get("adapt_steps", [0, 1, 3, 5])))

        print(
            "[FewShot-Validation] enabled: "
            f"interval={val_interval}, steps={val_steps}, "
            f"num_tasks={len(validation_tasks)}, seed={val_cfg.get('seed', 2026)}"
        )

        def _on_epoch_end(epoch, algo, paths, logging_infos):
            epoch_idx = int(epoch) + 1
            if (epoch_idx % val_interval != 0) and (epoch_idx != int(params["epochs"])):
                return

            step_costs, score = _evaluate_fewshot_validation(algo, validation_tasks, params)
            print(f"[FewShot-Validation] epoch={epoch_idx}, score={score:.4f}, step_costs={step_costs}")

            if hasattr(algo, "writer"):
                algo.writer.add_scalar("FewShotVal/Score", score, global_step=epoch)
                for step, cost in sorted(step_costs.items()):
                    algo.writer.add_scalar(f"FewShotVal/TotalCost_Step{step}", cost, global_step=epoch)

            if np.isfinite(score) and score < best_state["score"]:
                best_state["score"] = float(score)
                best_state["epoch"] = epoch_idx
                best_state["step_costs"] = dict(step_costs)
                torch.save(algo.state_dict(), best_ckpt_path)
                with open(best_meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "best_epoch": best_state["epoch"],
                            "best_score": best_state["score"],
                            "step_costs": best_state["step_costs"],
                            "adapt_steps": val_steps,
                            "interval": val_interval,
                            "validation_num_tasks": len(validation_tasks),
                            "validation_seed": int(val_cfg.get("seed", 2026)),
                        },
                        f,
                        indent=2,
                    )
                print(
                    "[Checkpoint] new best few-shot model saved: "
                    f"epoch={best_state['epoch']}, score={best_state['score']:.4f}, path={best_ckpt_path}"
                )

        epoch_callback = _on_epoch_end

    meta_algo.learn(params["epochs"], epoch_callback=epoch_callback)
    torch.save(meta_algo.state_dict(), last_ckpt_path)

    if val_enabled and best_state["epoch"] < 0:
        torch.save(meta_algo.state_dict(), best_ckpt_path)
        with open(best_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": int(params["epochs"]),
                    "best_score": None,
                    "step_costs": {},
                    "adapt_steps": sorted(set(int(s) for s in val_cfg.get("adapt_steps", [0, 1, 3, 5]))),
                    "note": "No validation-improvement checkpoint was found; copied final model.",
                },
                f,
                indent=2,
            )

    if val_enabled:
        print(
            "[Checkpoint] final: "
            f"last={last_ckpt_path}, best={best_ckpt_path}, meta={best_meta_path}"
        )
    else:
        print(f"[Checkpoint] final: {last_ckpt_path}")
    
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
        "epochs": 3000,  # Total number of meta-training epochs (outer loop iterations with new task sampling). Change to 500 for full training
        "parallel": True,  # Use multiprocessing for parallel environment sampling. If False, samples sequentially (slower but less memory)
        "learn_std": True,  # Whether the policy's action standard deviation (log_std parameter) is learnable. If False, log_std remains fixed at init value
        
        # ===== Logging & Device =====
        "tensor_log": TENSORFLOW_LOGS,  # Directory path for TensorBoard event files (auto-created if not exists). Used by SummaryWriter for training visualization
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Use GPU if available
        "action_log_interval": 100,  # Log action distribution every N epochs

        # ===== Held-out Few-shot Validation (best-checkpoint selection) =====
        "fewshot_validation": {
            "enabled": True,
            "interval": 20,              # Run validation every N epochs (+ final epoch)
            "adapt_steps": [0, 1, 3, 5], # Requested few-shot adaptation steps
            "num_tasks": 5,              # Fixed held-out task pool size
            "rollout_per_task": 5,       # Rollouts per task for validation sampling
            "max_path_length": SIM_TIME,
            "parallel": False,
            "envs_per_task": 1,
            "seed": 2026,                # Fixed seed for held-out task generation
            "scenario_overrides": {},    # Optional: demand/leadtime range overrides
        },
    }
    main(params)
   
