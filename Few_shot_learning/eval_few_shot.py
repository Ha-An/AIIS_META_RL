import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from envs.promp_env import MetaEnv
from envs.config_SimPy import SIM_TIME
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Agents.Categorical.Meta_Categorical import MetaCategoricalAgent
from envs.config_RL import ACTION_SPACE
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Sampler.meta_sampler import MetaSampler
from AIIS_META.Sampler.meta_sample_processor import MetaSampleProcessor

import envs.scenarios as scenarios

import Few_shot_learning.config as cfg


def _create_scenarios():
    return scenarios.create_scenarios(**cfg.SCENARIO_DIST_CONFIG)

def _resolve_eval_settings():
    eval_cfg = dict(cfg.EVAL_CONFIG)
    model_cfg = cfg.MODEL_CONFIG

    num_task = int(eval_cfg.get("num_task") or model_cfg["num_task"])
    rollout_per_task = int(eval_cfg.get("rollout_per_task") or model_cfg["rollout_per_task"])
    max_path_length = int(eval_cfg.get("max_path_length") or model_cfg["max_path_length"])

    eval_rounds = int(eval_cfg.get("eval_rounds", 1))
    max_adapt_steps = int(eval_cfg.get("max_adapt_steps", 0))

    parallel = bool(eval_cfg.get("parallel", False))
    envs_per_task = eval_cfg.get("envs_per_task", 1)
    if envs_per_task is None:
        envs_per_task = 1
    envs_per_task = int(envs_per_task)
    envs_per_task = max(1, min(envs_per_task, rollout_per_task))

    max_total_env_steps = int(eval_cfg.get("max_total_env_steps", 5_000_000))
    auto_clamp = bool(eval_cfg.get("auto_clamp", True))
    action_log_interval = int(eval_cfg.get("action_log_interval", 10))
    if action_log_interval < 1:
        action_log_interval = 1

    def total_steps(r, s, rpt):
        return num_task * rpt * max_path_length * (s + 1) * r

    original = {
        "eval_rounds": eval_rounds,
        "max_adapt_steps": max_adapt_steps,
        "rollout_per_task": rollout_per_task,
    }

    current_total = total_steps(eval_rounds, max_adapt_steps, rollout_per_task)
    if current_total > max_total_env_steps:
        if not auto_clamp:
            raise RuntimeError(
                "Evaluation budget too large: "
                f"{current_total} env steps > {max_total_env_steps}. "
                "Reduce eval_rounds, max_adapt_steps, or rollout_per_task."
            )

        denom = num_task * rollout_per_task * max_path_length * (max_adapt_steps + 1)
        if denom > 0:
            eval_rounds = max(1, max_total_env_steps // denom)

        current_total = total_steps(eval_rounds, max_adapt_steps, rollout_per_task)
        if current_total > max_total_env_steps:
            denom = num_task * rollout_per_task * max_path_length * eval_rounds
            if denom > 0:
                max_adapt_steps = max(0, (max_total_env_steps // denom) - 1)

        current_total = total_steps(eval_rounds, max_adapt_steps, rollout_per_task)
        if current_total > max_total_env_steps:
            denom = num_task * max_path_length * (max_adapt_steps + 1) * eval_rounds
            if denom > 0:
                rollout_per_task = max(1, max_total_env_steps // denom)

        current_total = total_steps(eval_rounds, max_adapt_steps, rollout_per_task)
        if current_total > max_total_env_steps:
            raise RuntimeError(
                "Unable to clamp evaluation budget under the limit. "
                "Please lower eval settings manually."
            )

        envs_per_task = min(envs_per_task, rollout_per_task)

        if (eval_rounds, max_adapt_steps, rollout_per_task) != (
            original["eval_rounds"],
            original["max_adapt_steps"],
            original["rollout_per_task"],
        ):
            print(
                "Clamped evaluation budget to avoid overload: "
                f"eval_rounds {original['eval_rounds']} -> {eval_rounds}, "
                f"max_adapt_steps {original['max_adapt_steps']} -> {max_adapt_steps}, "
                f"rollout_per_task {original['rollout_per_task']} -> {rollout_per_task}"
            )

    return {
        "num_task": num_task,
        "rollout_per_task": rollout_per_task,
        "max_path_length": max_path_length,
        "eval_rounds": eval_rounds,
        "max_adapt_steps": max_adapt_steps,
        "parallel": parallel,
        "envs_per_task": envs_per_task,
        "action_log_interval": action_log_interval,
    }


def _next_run_dir(base_dir: Path, prefix: str = "Train_") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix):]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = max(existing, default=0) + 1
    return base_dir / f"{prefix}{next_idx}"


def _resolve_run_dir(base_dir: Path) -> Path:
    run_name = os.environ.get("FEWSHOT_RUN_NAME", "").strip()
    if not run_name:
        return _next_run_dir(base_dir)
    # Ensure a unique directory name
    candidate = base_dir / run_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        alt = base_dir / f"{run_name}_{suffix}"
        if not alt.exists():
            return alt
        suffix += 1


def build_agent(env: MetaEnv, device: torch.device):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    policy_dist = cfg.MODEL_CONFIG.get("policy_dist", "gaussian").lower()

    if policy_dist == "categorical":
        num_actions = len(ACTION_SPACE)
        mlp = SimpleMLP(
            input_dim=obs_dim,
            output_dim=act_dim * num_actions,
            hidden_layers=cfg.MODEL_CONFIG["layers"],
        )
        agent = MetaCategoricalAgent(
            mlp=mlp,
            num_tasks=cfg.MODEL_CONFIG["num_task"],
            action_dim=act_dim,
            num_actions=num_actions,
        )
    else:
        mlp = SimpleMLP(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_layers=cfg.MODEL_CONFIG["layers"],
        )
        agent = MetaGaussianAgent(
            mlp=mlp,
            num_tasks=cfg.MODEL_CONFIG["num_task"],
            learn_std=cfg.MODEL_CONFIG["learn_std"],
        )

    agent.to(device)
    return agent


def make_promp(env: MetaEnv, agent: MetaGaussianAgent) -> ProMP:
    return ProMP(
        env=env,
        max_path_length=cfg.MODEL_CONFIG["max_path_length"],
        agent=agent,
        alpha=cfg.MODEL_CONFIG["alpha"],
        beta=cfg.MODEL_CONFIG["beta"],
        baseline=LinearFeatureBaseline(),
        tensor_log="",
        inner_grad_steps=cfg.MODEL_CONFIG["num_inner_grad"],
        num_tasks=cfg.MODEL_CONFIG["num_task"],
        outer_iters=cfg.MODEL_CONFIG["outer_iters"],
        parallel=cfg.MODEL_CONFIG["parallel"],
        rollout_per_task=cfg.MODEL_CONFIG["rollout_per_task"],
        clip_eps=cfg.MODEL_CONFIG["clip_eps"],
        device=cfg.MODEL_CONFIG["device"],
    )


def evaluate_adaptation_curve(promp: ProMP, env: MetaEnv, writer: SummaryWriter):
    settings = _resolve_eval_settings()
    sampler = MetaSampler(
        env=env,
        agent=promp.agent,
        rollout_per_task=settings["rollout_per_task"],
        num_tasks=settings["num_task"],
        max_path_length=settings["max_path_length"],
        envs_per_task=settings["envs_per_task"],
        parallel=settings["parallel"],
    )

    sample_processor = MetaSampleProcessor(
        baseline=LinearFeatureBaseline(),
        discount=0.99,
        gae_lambda=1.0,
        normalize_adv=True,
    )

    max_steps = settings["max_adapt_steps"]
    eval_rounds = settings["eval_rounds"]
    action_log_interval = settings["action_log_interval"]

    rewards_by_step = [[] for _ in range(max_steps + 1)]

    try:
        for _round_idx in range(eval_rounds):
            sampler.update_tasks()

            adapted_params_list = [
                OrderedDict(promp.agent.named_parameters())
                for _ in range(settings["num_task"])
            ]

            for step_idx in range(max_steps + 1):
                paths = sampler.obtain_samples(adapted_params_list, post_update=True)
                processed = sample_processor.process_samples(paths)

                infos = env.report_scalar(processed, settings["rollout_per_task"])
                reward = infos.get("reward", None)
                if reward is not None:
                    rewards_by_step[step_idx].append(reward)

                if step_idx % action_log_interval == 0:
                    actions = np.concatenate(
                        [processed[task_id]["actions"] for task_id in range(settings["num_task"])],
                        axis=0,
                    )
                    if actions.size > 0:
                        actions_flat = actions.reshape(-1)
                        writer.add_histogram("FewShot/Actions", actions_flat, step_idx)
                        writer.add_scalar("FewShot/ActionMean", float(actions_flat.mean()), step_idx)
                        writer.add_scalar("FewShot/ActionStd", float(actions_flat.std()), step_idx)

                if step_idx < max_steps:
                    for task_id in range(settings["num_task"]):
                        batch = processed[task_id]
                        adapted_params_list[task_id] = promp._theta_prime(
                            batch,
                            params=adapted_params_list[task_id],
                            create_graph=False,
                        )
    finally:
        sampler.close()

    for step_idx, rewards in enumerate(rewards_by_step):
        if len(rewards) == 0:
            continue
        mean_reward = float(np.mean(rewards))
        writer.add_scalar("FewShot/Reward", mean_reward, step_idx)
        writer.add_scalar("FewShot/RewardStd", float(np.std(rewards)), step_idx)


def main():
    torch.manual_seed(cfg.EVAL_CONFIG["seed"] or 0)

    env = MetaEnv()
    env.create_scenarios = _create_scenarios

    agent = build_agent(env, cfg.MODEL_CONFIG["device"])
    promp = make_promp(env, agent)

    if not os.path.exists(cfg.PRETRAINED_MODEL_PATH):
        raise FileNotFoundError(f"Pretrained model not found: {cfg.PRETRAINED_MODEL_PATH}")

    state = torch.load(cfg.PRETRAINED_MODEL_PATH, map_location=cfg.MODEL_CONFIG["device"])
    state_keys = list(state.keys()) if hasattr(state, "keys") else []
    if any(k.startswith("agent.") for k in state_keys) or any(k.startswith("inner_step_sizes") for k in state_keys):
        # Meta-RL checkpoint (ProMP/VPG_MAML): load full algo state
        promp.load_state_dict(state, strict=False)
    else:
        # PPO checkpoint: agent-only state_dict
        promp.agent.load_state_dict(state, strict=False)

    root_dir = Path(__file__).parent
    log_root = root_dir / "Tensorboard_logs"
    run_dir = _resolve_run_dir(log_root)
    writer = SummaryWriter(log_dir=str(run_dir))

    evaluate_adaptation_curve(promp, env, writer)

    writer.flush()
    writer.close()
    promp.close()


if __name__ == "__main__":
    main()
