import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DRL.train_ppo_example import SCENARIO_DIST_CONFIG, create_fixed_task, create_ppo_agent
from envs.promp_env import MetaEnv


def _deterministic_action(agent, state, policy_dist, device):
    obs = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
    params = dict(agent.named_parameters())
    dist = agent.distribution(obs, params=params)
    if policy_dist == "categorical":
        action = torch.argmax(dist.logits, dim=-1).squeeze(0)
    else:
        action = dist.mean.squeeze(0)
    return action.detach().cpu().numpy()


def evaluate_model(model_path, task_id, reps, policy_dist):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MetaEnv()
    env.create_scenarios = lambda: __import__("envs.scenarios", fromlist=["create_scenarios"]).create_scenarios(**SCENARIO_DIST_CONFIG)
    fixed_task = create_fixed_task(task_id)

    agent, _ = create_ppo_agent(device, num_tasks=1, policy_dist=policy_dist)
    state = torch.load(model_path, map_location=device)
    agent.load_state_dict(state, strict=False)
    agent.eval()

    rows = []
    for rep in range(1, reps + 1):
        env.set_task(fixed_task)
        s = env.reset()
        done = False
        total_cost = 0.0
        last_cost_dict = None

        while not done:
            a = _deterministic_action(agent, s, policy_dist, device)
            s, reward, done, cost_dict = env.step(a)
            total_cost += -float(reward)
            last_cost_dict = dict(cost_dict)

        rows.append(
            {
                "rep": rep,
                "task_id": task_id,
                "total_cost_200d": total_cost,
                "holding_cost_200d": last_cost_dict["Holding cost"],
                "process_cost_200d": last_cost_dict["Process cost"],
                "delivery_cost_200d": last_cost_dict["Delivery cost"],
                "order_cost_200d": last_cost_dict["Order cost"],
                "shortage_cost_200d": last_cost_dict["Shortage cost"],
            }
        )

    return pd.DataFrame(rows)


def plot_eval(df, out_png):
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].boxplot([df["total_cost_200d"].values], tick_labels=["PPO"], showmeans=True)
    axes[0].set_title("200-Day Total Cost Distribution by Policy")
    axes[0].set_xlabel("Policy")
    axes[0].set_ylabel("Total Cost (200 days)")
    axes[0].grid(alpha=0.3)

    mean_total = df["total_cost_200d"].mean()
    std_total = df["total_cost_200d"].std()
    axes[1].bar(["PPO"], [mean_total], yerr=[std_total], capsize=6, color="#1f77b4")
    axes[1].set_title("Mean +- Std of 200-Day Total Cost")
    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Total Cost (200 days)")
    axes[1].grid(alpha=0.3)

    comp_cols = [
        "holding_cost_200d",
        "process_cost_200d",
        "delivery_cost_200d",
        "order_cost_200d",
        "shortage_cost_200d",
    ]
    comp_mean = df[comp_cols].mean()
    ratio = (comp_mean / comp_mean.sum()).fillna(0.0)

    label_map = {
        "holding_cost_200d": "Holding",
        "process_cost_200d": "Process",
        "delivery_cost_200d": "Delivery",
        "order_cost_200d": "Order",
        "shortage_cost_200d": "Shortage",
    }
    color_map = {
        "holding_cost_200d": "#4C78A8",
        "process_cost_200d": "#F58518",
        "delivery_cost_200d": "#54A24B",
        "order_cost_200d": "#E45756",
        "shortage_cost_200d": "#B279A2",
    }
    bottom = 0.0
    for col in comp_cols:
        val = float(ratio[col])
        axes[2].bar(["PPO"], [val], bottom=[bottom], label=label_map[col], color=color_map[col])
        bottom += val
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Cost Composition Ratio (Mean over Repetitions)")
    axes[2].set_xlabel("Policy")
    axes[2].set_ylabel("Ratio of total cost")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model on a fixed scenario")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--policy-dist", type=str, default="categorical", choices=["categorical", "gaussian"])
    parser.add_argument("--output-dir", type=Path, default=Path("DRL") / "Eval_results")
    args = parser.parse_args()

    df = evaluate_model(
        model_path=args.model_path,
        task_id=args.task_id,
        reps=args.reps,
        policy_dist=args.policy_dist,
    )

    model_tag = args.model_path.parent.parent.name if args.model_path.exists() else "model"
    out_dir = args.output_dir / f"{model_tag}_task_{args.task_id}_reps_{args.reps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "ppo_eval.csv"
    out_png = out_dir / "ppo_eval_3charts.png"
    df.to_csv(out_csv, index=False)
    plot_eval(df, out_png)

    summary = df["total_cost_200d"].agg(["mean", "std", "min", "max"])
    print(f"[Saved] Eval CSV: {out_csv}")
    print(f"[Saved] Eval Plot: {out_png}")
    print("[Summary total_cost_200d]")
    print(summary.to_string())


if __name__ == "__main__":
    main()
