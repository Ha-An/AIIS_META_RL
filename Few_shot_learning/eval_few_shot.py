import os
import sys
from pathlib import Path
from collections import OrderedDict
import csv
import random
import copy
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def _create_scenarios():
    if isinstance(cfg.SCENARIO_DIST_CONFIG, dict) and "fixed_scenarios" in cfg.SCENARIO_DIST_CONFIG:
        fixed_pool = cfg.SCENARIO_DIST_CONFIG.get("fixed_scenarios", [])
        if not fixed_pool:
            raise ValueError("SCENARIO_DIST_CONFIG['fixed_scenarios'] is empty.")
        # Return deep-copied tasks so each run can mutate independently.
        return copy.deepcopy(fixed_pool)
    return scenarios.create_scenarios(**cfg.SCENARIO_DIST_CONFIG)


def _resolve_tb_mode():
    mode = str(os.environ.get("FEWSHOT_TB_MODE", getattr(cfg, "FEWSHOT_TB_MODE", "summary"))).strip().lower()
    if mode not in {"off", "summary", "full"}:
        mode = "summary"
    if str(os.environ.get("FEWSHOT_DISABLE_TENSORBOARD", "0")).strip() == "1":
        mode = "off"
    return mode

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


def make_promp(env: MetaEnv, agent: MetaGaussianAgent, tensor_log: Optional[str] = None) -> ProMP:
    return ProMP(
        env=env,
        max_path_length=cfg.MODEL_CONFIG["max_path_length"],
        agent=agent,
        alpha=cfg.MODEL_CONFIG["alpha"],
        beta=cfg.MODEL_CONFIG["beta"],
        baseline=LinearFeatureBaseline(),
        tensor_log=tensor_log,
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
    tb_mode = _resolve_tb_mode()
    log_actions = tb_mode == "full"
    print(
        "[FewShot] Adaptation curve settings: "
        f"parallel={settings['parallel']}, envs_per_task={settings['envs_per_task']}, "
        f"num_task={settings['num_task']}, rollout_per_task={settings['rollout_per_task']}, "
        f"max_path_length={settings['max_path_length']}, tb_mode={tb_mode}"
    )
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

                if log_actions and step_idx % action_log_interval == 0:
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


def evaluate_cost_boxplot(promp: ProMP, env: MetaEnv, writer: SummaryWriter, run_dir: Path):
    box_cfg = cfg.COST_BOXPLOT_CONFIG
    if not box_cfg.get("enabled", False):
        return

    adapt_steps = sorted(set(int(x) for x in box_cfg.get("adapt_steps", [])))
    repetitions = int(box_cfg.get("repetitions", 0))
    if not adapt_steps or repetitions <= 0:
        return

    settings = _resolve_eval_settings()
    tb_mode = _resolve_tb_mode()
    print(
        "[FewShot] Cost boxplot settings: "
        f"parallel={settings['parallel']}, envs_per_task={settings['envs_per_task']}, "
        f"num_task={settings['num_task']}, rollout_per_task={settings['rollout_per_task']}, "
        f"max_path_length={settings['max_path_length']}, "
        f"adapt_steps={adapt_steps}, repetitions={repetitions}, tb_mode={tb_mode}"
    )
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

    max_step = max(adapt_steps)
    costs_by_step = {step: [] for step in adapt_steps}
    rows = []

    try:
        for rep in range(repetitions):
            # Keep one sampled task set fixed within a repetition,
            # then compare step 0/5/10 under identical tasks.
            sampler.update_tasks()
            adapted_params_list = [
                OrderedDict(promp.agent.named_parameters())
                for _ in range(settings["num_task"])
            ]

            for step in range(max_step + 1):
                if step > 0:
                    adapt_paths = sampler.obtain_samples(adapted_params_list, post_update=True)
                    processed = sample_processor.process_samples(adapt_paths)
                    for task_id in range(settings["num_task"]):
                        adapted_params_list[task_id] = promp._theta_prime(
                            processed[task_id],
                            params=adapted_params_list[task_id],
                            create_graph=False,
                        )

                if step not in costs_by_step:
                    continue

                eval_paths = sampler.obtain_samples(adapted_params_list, post_update=True)

                rollout_costs = []
                for task_paths in eval_paths.values():
                    for traj in task_paths:
                        rollout_costs.append(float(-np.sum(traj["rewards"])))

                mean_total_cost = float(np.mean(rollout_costs)) if rollout_costs else float("nan")
                costs_by_step[step].append(mean_total_cost)
                rows.append(
                    {
                        "adapt_step": step,
                        "repetition": rep + 1,
                        "mean_total_cost": mean_total_cost,
                    }
                )

                if tb_mode == "full":
                    writer.add_scalar(f"FewShot/CostBoxplot/step_{step}", mean_total_cost, rep)
    finally:
        sampler.close()

    if tb_mode in {"summary", "full"}:
        for step in adapt_steps:
            vals = np.asarray(costs_by_step[step], dtype=float)
            if vals.size == 0:
                continue
            writer.add_scalar(f"FewShot/CostBoxplotMean/step_{step}", float(np.mean(vals)), 0)
            writer.add_scalar(f"FewShot/CostBoxplotStd/step_{step}", float(np.std(vals)), 0)

    no_artifacts = str(os.environ.get("FEWSHOT_DISABLE_RUN_ARTIFACTS", "0")).strip() == "1"
    if not no_artifacts:
        csv_path = run_dir / "fewshot_total_cost_boxplot_data.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["adapt_step", "repetition", "mean_total_cost"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        labels = [str(step) for step in adapt_steps]
        data = [costs_by_step[step] for step in adapt_steps]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title("Total Cost by Few-shot Adaptation Steps")
        ax.set_xlabel("Adaptation Step")
        ax.set_ylabel("Mean Total Cost (per repetition)")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fig_path = run_dir / "fewshot_total_cost_boxplot.png"
        fig.savefig(fig_path, dpi=150)
        writer.add_figure("FewShot/TotalCostBoxplot", fig, global_step=0)
        plt.close(fig)

        print(f"[FewShot] Saved boxplot CSV: {csv_path}")
        print(f"[FewShot] Saved boxplot PNG: {fig_path}")

    return rows


def _as_tensor_state_dict(candidate):
    if not isinstance(candidate, dict):
        return None
    tensor_state = {
        k: v for k, v in candidate.items()
        if isinstance(k, str) and torch.is_tensor(v)
    }
    if len(tensor_state) == 0:
        return None
    return tensor_state


def _extract_agent_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a dict-like state_dict.")

    if "agent_state_dict" in state:
        out = _as_tensor_state_dict(state["agent_state_dict"])
        if out is not None:
            return out

    if "state_dict" in state:
        nested = state["state_dict"]
        if isinstance(nested, dict):
            agent_prefixed = {
                str(k)[len("agent."):]: v
                for k, v in nested.items()
                if isinstance(k, str) and k.startswith("agent.") and torch.is_tensor(v)
            }
            if len(agent_prefixed) > 0:
                return agent_prefixed
            out = _as_tensor_state_dict(nested)
            if out is not None:
                return out

    agent_prefixed = {
        k[len("agent."):]: v
        for k, v in state.items()
        if isinstance(k, str) and k.startswith("agent.") and torch.is_tensor(v)
    }
    if len(agent_prefixed) > 0:
        return agent_prefixed

    out = _as_tensor_state_dict(state)
    if out is not None:
        return out

    raise ValueError("Could not extract agent state_dict from checkpoint.")


def _reset_inner_step_sizes(promp: ProMP):
    if not hasattr(promp, "inner_step_sizes"):
        return
    alpha = float(cfg.MODEL_CONFIG["alpha"])
    with torch.no_grad():
        for p in promp.inner_step_sizes.values():
            p.fill_(alpha)


def _load_pretrained(promp: ProMP):
    path = cfg.PRETRAINED_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained model not found: {path}")

    mode = str(getattr(cfg, "CHECKPOINT_LOAD_MODE", "auto")).strip().lower()
    state = torch.load(path, map_location=cfg.MODEL_CONFIG["device"])
    state_keys = list(state.keys()) if hasattr(state, "keys") else []

    if mode == "full":
        info = promp.load_state_dict(state, strict=False)
        print(
            f"[FewShot] Loaded FULL checkpoint: {path} "
            f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})"
        )
        return

    if mode == "agent_only":
        agent_state = _extract_agent_state_dict(state)
        info = promp.agent.load_state_dict(agent_state, strict=False)
        if bool(getattr(cfg, "RESET_INNER_STEP_SIZES_ON_LOAD", True)):
            _reset_inner_step_sizes(promp)
        print(
            f"[FewShot] Loaded AGENT-ONLY checkpoint: {path} "
            f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}), "
            f"reset_inner_step_sizes={bool(getattr(cfg, 'RESET_INNER_STEP_SIZES_ON_LOAD', True))}"
        )
        return

    if mode == "auto":
        if any(k.startswith("agent.") for k in state_keys) or any(k.startswith("inner_step_sizes") for k in state_keys):
            info = promp.load_state_dict(state, strict=False)
            print(
                f"[FewShot] AUTO mode -> FULL load: {path} "
                f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})"
            )
        else:
            agent_state = _extract_agent_state_dict(state)
            info = promp.agent.load_state_dict(agent_state, strict=False)
            print(
                f"[FewShot] AUTO mode -> AGENT load: {path} "
                f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})"
            )
        return

    raise ValueError(f"Unknown CHECKPOINT_LOAD_MODE: {mode}")


def main():
    seed = cfg.EVAL_CONFIG["seed"] or 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MetaEnv()
    env.create_scenarios = _create_scenarios

    root_dir = Path(__file__).parent
    log_root = root_dir / "Tensorboard_logs"
    flat_output = str(os.environ.get("FEWSHOT_FLAT_OUTPUT", "0")).strip() == "1"
    run_dir = log_root if flat_output else _resolve_run_dir(log_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_mode = _resolve_tb_mode()
    algo_tensor_log = str(run_dir) if tb_mode != "off" else None
    skip_adapt_curve = str(os.environ.get("FEWSHOT_SKIP_ADAPTATION_CURVE", "0")).strip() == "1"

    agent = build_agent(env, cfg.MODEL_CONFIG["device"])
    promp = make_promp(env, agent, tensor_log=algo_tensor_log)
    _load_pretrained(promp)

    writer = promp.writer if hasattr(promp, "writer") else _NullWriter()

    if not skip_adapt_curve:
        evaluate_adaptation_curve(promp, env, writer)
    else:
        print("[FewShot] Skipping adaptation-curve evaluation for faster batch run.")
    box_rows = evaluate_cost_boxplot(promp, env, writer, run_dir)
    promp.close()
    return {"run_dir": str(run_dir), "boxplot_rows": box_rows}


if __name__ == "__main__":
    main()

