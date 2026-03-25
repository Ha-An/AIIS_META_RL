import argparse
import copy
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRL.config import EVAL_RESULT_PATH, EVAL_RUN_CONFIG
from DRL.train_ppo_example import SCENARIO_DIST_CONFIG, create_fixed_task, create_ppo_agent
from envs.promp_env import MetaEnv
import envs.scenarios as scenarios


COST_COMPONENT_COLUMNS = [
    "holding_cost_200d",
    "process_cost_200d",
    "delivery_cost_200d",
    "order_cost_200d",
    "shortage_cost_200d",
]

COST_COMPONENT_LABELS = {
    "holding_cost_200d": "Holding",
    "process_cost_200d": "Process",
    "delivery_cost_200d": "Delivery",
    "order_cost_200d": "Order",
    "shortage_cost_200d": "Shortage",
}

COST_COMPONENT_COLORS = {
    "holding_cost_200d": "#E64B35",
    "process_cost_200d": "#4C9AD4",
    "delivery_cost_200d": "#8F7FD3",
    "order_cost_200d": "#7F7F7F",
    "shortage_cost_200d": "#FDBF57",
}

MAX_EPISODE_RETRIES = 5
_TRANSIENT_ENV_ERROR_KEYWORDS = (
    "'bool' object has no attribute 'on_hand_inventory'",
    "'list' object is not callable",
    "unsupported operand type(s) for -=: 'Inventory' and 'float'",
    "'int' object is not subscriptable",
    "'int' object is not callable",
    "'Inventory' object is not subscriptable",
    "object is not callable",
    "'list' object has no attribute 'in_transition_inventory'",
    "'Production' object has no attribute 'capacity_limit'",
)


def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deterministic_action(agent, state, policy_dist, device):
    obs = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
    params = dict(agent.named_parameters())
    dist = agent.distribution(obs, params=params)
    if policy_dist == "categorical":
        action = torch.argmax(dist.logits, dim=-1).squeeze(0)
    else:
        action = dist.mean.squeeze(0)
    return action.detach().cpu().numpy()


def _evaluate_episode(env, agent, policy_dist, device):
    s = env.reset()
    done = False
    total_cost = 0.0
    last_cost_dict = None

    while not done:
        a = _deterministic_action(agent, s, policy_dist, device)
        s, reward, done, cost_dict = env.step(a)
        total_cost += -float(reward)
        last_cost_dict = dict(cost_dict)

    return {
        "total_cost_200d": float(total_cost),
        "holding_cost_200d": float(last_cost_dict["Holding cost"]),
        "process_cost_200d": float(last_cost_dict["Process cost"]),
        "delivery_cost_200d": float(last_cost_dict["Delivery cost"]),
        "order_cost_200d": float(last_cost_dict["Order cost"]),
        "shortage_cost_200d": float(last_cost_dict["Shortage cost"]),
    }


def _is_transient_env_error(exc):
    message = str(exc)
    return any(keyword in message for keyword in _TRANSIENT_ENV_ERROR_KEYWORDS)


def _run_with_retry(fn, description):
    last_error = None
    for attempt in range(1, MAX_EPISODE_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if (not _is_transient_env_error(exc)) or attempt >= MAX_EPISODE_RETRIES:
                raise
            print(f"[Retry {attempt}/{MAX_EPISODE_RETRIES}] {description}: {exc}")
            gc.collect()
    raise last_error


def _normalize_model_specs(raw_model_specs):
    model_specs = raw_model_specs or []
    if not model_specs:
        model_specs = [
            {
                "name": "PPO",
                "model_path": EVAL_RUN_CONFIG["model_path"],
                "policy_dist": EVAL_RUN_CONFIG["policy_dist"],
            }
        ]

    normalized = []
    for index, spec in enumerate(model_specs, start=1):
        normalized.append(
            {
                "name": str(spec.get("name") or f"Model_{index}"),
                "model_path": Path(spec["model_path"]),
                "policy_dist": str(spec.get("policy_dist", EVAL_RUN_CONFIG["policy_dist"])),
            }
        )
    return normalized


def _build_eval_env(seed):
    env = MetaEnv()
    scenario_dist_config = dict(SCENARIO_DIST_CONFIG)
    scenario_dist_config["seed"] = int(seed)
    env.create_scenarios = lambda: scenarios.create_scenarios(**scenario_dist_config)
    return env


def _build_task_schedule(mode, reps, task_id, seed):
    reps = int(reps)
    seed = int(seed)
    schedule = []

    if mode == "fixed_task":
        base_task = create_fixed_task(int(task_id))
        for rep in range(1, reps + 1):
            schedule.append(
                {
                    "rep": rep,
                    "task": copy.deepcopy(base_task),
                    "task_id": int(task_id),
                    "episode_seed": seed + rep - 1,
                    "demand_dist": str(base_task.get("DEMAND", {})),
                }
            )
        return schedule

    # Keep randomized-task semantics aligned with the previous script:
    # sample repeated evaluations from one seeded randomized scenario pool.
    rng = random.Random(seed)
    scenario_dist_config = dict(SCENARIO_DIST_CONFIG)
    scenario_dist_config["seed"] = seed
    scenario_pool = scenarios.create_scenarios(**scenario_dist_config)
    if not scenario_pool:
        raise RuntimeError("Randomized scenario pool is empty.")

    for rep in range(1, reps + 1):
        task = copy.deepcopy(rng.choice(scenario_pool))
        schedule.append(
            {
                "rep": rep,
                "task": task,
                "task_id": -1,
                "episode_seed": seed + rep - 1,
                "demand_dist": str(task.get("DEMAND", {})),
            }
        )
    return schedule


def _load_agents(model_specs, device):
    loaded = []
    for spec in model_specs:
        if not spec["model_path"].exists():
            raise FileNotFoundError(f"Saved model not found: {spec['model_path']}")

        agent, _ = create_ppo_agent(device, num_tasks=1, policy_dist=spec["policy_dist"])
        state = torch.load(spec["model_path"], map_location=device)
        agent.load_state_dict(state, strict=False)
        agent.eval()
        loaded.append(
            {
                "name": spec["name"],
                "model_path": spec["model_path"],
                "policy_dist": spec["policy_dist"],
                "agent": agent,
            }
        )
    return loaded


def evaluate_models(model_specs, mode, reps, seed, task_id=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = _normalize_model_specs(model_specs)
    loaded_models = _load_agents(model_specs, device)
    task_schedule = _build_task_schedule(mode=mode, reps=reps, task_id=task_id, seed=seed)

    rows = []
    for task_info in task_schedule:
        for model in loaded_models:
            def _attempt():
                _seed_everything(task_info["episode_seed"])
                env = _build_eval_env(task_info["episode_seed"])
                env.set_task(copy.deepcopy(task_info["task"]))
                return _evaluate_episode(env, model["agent"], model["policy_dist"], device)

            metrics = _run_with_retry(
                _attempt,
                f"{model['name']} rep {task_info['rep']}",
            )
            metrics.update(
                {
                    "model": model["name"],
                    "model_path": str(model["model_path"]),
                    "policy_dist": model["policy_dist"],
                    "rep": int(task_info["rep"]),
                    "mode": mode,
                    "task_id": int(task_info["task_id"]),
                    "episode_seed": int(task_info["episode_seed"]),
                    "demand_dist": task_info["demand_dist"],
                }
            )
            rows.append(metrics)

    return pd.DataFrame(rows)


def plot_eval(df, out_png, model_order):
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    boxplot_data = [df.loc[df["model"] == model_name, "total_cost_200d"].to_numpy() for model_name in model_order]
    axes[0].boxplot(boxplot_data, tick_labels=model_order, showmeans=True)
    axes[0].set_title("Total Cost Box Plot (200-day)")
    axes[0].set_xlabel("Policy")
    axes[0].set_ylabel("Total Cost")
    axes[0].grid(alpha=0.3)

    ratio_df = df.groupby("model", as_index=True)[COST_COMPONENT_COLUMNS].mean()
    ratio_df = ratio_df.div(ratio_df.sum(axis=1), axis=0).fillna(0.0)
    ratio_df = ratio_df.reindex(model_order)

    bottom = np.zeros(len(model_order), dtype=float)
    for col in COST_COMPONENT_COLUMNS:
        values = ratio_df[col].fillna(0.0).to_numpy(dtype=float)
        axes[1].bar(
            model_order,
            values,
            bottom=bottom,
            label=COST_COMPONENT_LABELS[col],
            color=COST_COMPONENT_COLORS[col],
        )
        bottom += values

    axes[1].set_ylim(0, 1)
    axes[1].set_title("Cost Component Ratio (mean over reps)")
    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Ratio")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    for axis in axes:
        axis.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare saved PPO models")
    parser.add_argument("--mode", type=str, default=EVAL_RUN_CONFIG["mode"], choices=["fixed_task", "randomized_task"])
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(EVAL_RUN_CONFIG["task_id"]),
        choices=[0, 1, 2, 3],
        help="Used only in fixed_task mode",
    )
    parser.add_argument("--reps", type=int, default=int(EVAL_RUN_CONFIG["reps"]))
    parser.add_argument("--seed", type=int, default=int(EVAL_RUN_CONFIG["seed"]))
    parser.add_argument("--output-dir", type=Path, default=Path(EVAL_RUN_CONFIG.get("output_dir", EVAL_RESULT_PATH)))
    args = parser.parse_args()

    model_specs = _normalize_model_specs(EVAL_RUN_CONFIG.get("model_specs"))
    df = evaluate_models(
        model_specs=model_specs,
        mode=args.mode,
        reps=args.reps,
        seed=args.seed,
        task_id=args.task_id,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == "fixed_task":
        run_tag = f"compare_{len(model_specs)}models_fixed_task_{args.task_id}_reps_{args.reps}_{timestamp}"
    else:
        run_tag = f"compare_{len(model_specs)}models_randomized_task_reps_{args.reps}_{timestamp}"

    out_dir = args.output_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "ppo_eval_comparison.csv"
    out_png = out_dir / "ppo_eval_comparison.png"
    out_summary = out_dir / "ppo_eval_summary_by_model.csv"

    df.to_csv(out_csv, index=False)
    summary_df = (
        df.groupby("model")["total_cost_200d"]
        .agg(count="count", mean="mean", std="std", min="min", max="max", median="median")
        .reset_index()
    )
    summary_df.to_csv(out_summary, index=False)
    plot_eval(df, out_png, [spec["name"] for spec in model_specs])

    print(f"[Config] mode: {args.mode}")
    print(f"[Config] reps: {args.reps}")
    print(f"[Config] seed: {args.seed}")
    if args.mode == "fixed_task":
        print(f"[Config] task_id: {args.task_id}")
    print("[Config] models:")
    for spec in model_specs:
        print(f"  - {spec['name']}: {spec['model_path']}")
    print(f"[Saved] Eval CSV: {out_csv}")
    print(f"[Saved] Eval Summary: {out_summary}")
    print(f"[Saved] Eval Plot: {out_png}")
    print("[Summary total_cost_200d]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()


