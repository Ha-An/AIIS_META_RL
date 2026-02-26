import argparse
import hashlib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from envs.simpy_diagnostics import run_simulation
from envs.scenarios import SCENARIO_SAMPLING_DEFAULTS


def _run_one(
    reorder_point,
    days,
    demand_min,
    demand_max,
    demand_mode,
    leadtime_min,
    leadtime_max,
    cust_order_cycle,
    leadtime_mode,
    seed,
):
    df, checks, dist_meta = run_simulation(
        days=days,
        lot_size=0,
        demand_min=demand_min,
        demand_max=demand_max,
        leadtime_min=leadtime_min,
        leadtime_max=leadtime_max,
        cust_order_cycle=cust_order_cycle,
        policy="heuristic_rop",
        reorder_point=reorder_point,
        demand_mode=demand_mode,
        leadtime_mode=leadtime_mode,
        seed=seed,
    )
    return {
        "total_cost_200d": float(df["total_cost"].sum()),
        "holding_cost_200d": float(df["holding_cost"].sum()),
        "process_cost_200d": float(df["process_cost"].sum()),
        "delivery_cost_200d": float(df["delivery_cost"].sum()),
        "order_cost_200d": float(df["order_cost"].sum()),
        "shortage_cost_200d": float(df["shortage_cost"].sum()),
        "shortage_units_200d": float(df["shortage_units"].sum()),
        "all_checks_passed": all(x["passed"] for x in checks),
        "demand_dist": dist_meta["demand"],
        "leadtime_mode": dist_meta["leadtime_mode"],
        "leadtime_profile": " | ".join(dist_meta.get("leadtime_lines", [])),
        "leadtime_profile_sig": hashlib.md5(
            " | ".join(dist_meta.get("leadtime_lines", [])).encode("utf-8")
        ).hexdigest()[:10],
    }


def run_experiment(
    reps,
    days,
    reorder_points,
    demand_min,
    demand_max,
    demand_mode,
    leadtime_min,
    leadtime_max,
    cust_order_cycle,
    leadtime_mode,
    base_seed,
    same_scenario_across_r,
):
    rows = []
    for rep in range(reps):
        # Common random numbers design:
        # if enabled, each R setting in the same repetition sees the same scenario seed.
        rep_seed = base_seed + rep if base_seed is not None else None
        for idx, rp in enumerate(reorder_points):
            seed = rep_seed if same_scenario_across_r else (None if base_seed is None else base_seed + rep * 100 + idx)
            metrics = _run_one(
                reorder_point=rp,
                days=days,
                demand_min=demand_min,
                demand_max=demand_max,
                demand_mode=demand_mode,
                leadtime_min=leadtime_min,
                leadtime_max=leadtime_max,
                cust_order_cycle=cust_order_cycle,
                leadtime_mode=leadtime_mode,
                seed=seed,
            )
            metrics["rep"] = rep + 1
            metrics["reorder_point"] = rp
            rows.append(metrics)
    return pd.DataFrame(rows)


def plot_results(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(df["reorder_point"].unique())
    labels = [f"R{r}" for r in order]
    groups = [df[df["reorder_point"] == r]["total_cost_200d"].values for r in order]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].boxplot(groups, tick_labels=labels, showmeans=True)
    axes[0].set_title("200-Day Total Cost Distribution by Reorder Point")
    axes[0].set_xlabel("Policy")
    axes[0].set_ylabel("Total Cost (200 days)")
    axes[0].grid(alpha=0.3)

    summary = (
        df.groupby("reorder_point")["total_cost_200d"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("reorder_point")
    )
    axes[1].bar([f"R{r}" for r in summary["reorder_point"]], summary["mean"], yerr=summary["std"], capsize=5)
    axes[1].set_title("Mean ± Std of 200-Day Total Cost")
    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Total Cost (200 days)")
    axes[1].grid(alpha=0.3)

    comp = (
        df.groupby("reorder_point")[
            [
                "holding_cost_200d",
                "process_cost_200d",
                "delivery_cost_200d",
                "order_cost_200d",
                "shortage_cost_200d",
            ]
        ]
        .mean()
        .sort_index()
    )
    comp_ratio = comp.div(comp.sum(axis=1), axis=0).fillna(0.0)
    x_labels = [f"R{r}" for r in comp_ratio.index]
    bottom = [0.0] * len(comp_ratio)
    color_map = {
        "holding_cost_200d": "#4C78A8",
        "process_cost_200d": "#F58518",
        "delivery_cost_200d": "#54A24B",
        "order_cost_200d": "#E45756",
        "shortage_cost_200d": "#B279A2",
    }
    label_map = {
        "holding_cost_200d": "Holding",
        "process_cost_200d": "Process",
        "delivery_cost_200d": "Delivery",
        "order_cost_200d": "Order",
        "shortage_cost_200d": "Shortage",
    }
    for col in [
        "holding_cost_200d",
        "process_cost_200d",
        "delivery_cost_200d",
        "order_cost_200d",
        "shortage_cost_200d",
    ]:
        vals = comp_ratio[col].tolist()
        axes[2].bar(
            x_labels,
            vals,
            bottom=bottom,
            label=label_map[col],
            color=color_map[col],
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Cost Composition Ratio (Mean over Repetitions)")
    axes[2].set_xlabel("Policy")
    axes[2].set_ylabel("Ratio of Total Cost")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = output_dir / "reorder_point_experiment.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return fig_path


def parse_args():
    parser = argparse.ArgumentParser(description="Reorder-point experiment runner (R1/R3/R5)")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--reorder-points", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--demand-min", type=int, default=SCENARIO_SAMPLING_DEFAULTS["demand_min"])
    parser.add_argument("--demand-max", type=int, default=SCENARIO_SAMPLING_DEFAULTS["demand_max"])
    parser.add_argument(
        "--demand-mode",
        type=str,
        default="per_scenario_random",
        choices=["fixed_uniform", "per_scenario_random"],
    )
    parser.add_argument("--leadtime-min", type=int, default=SCENARIO_SAMPLING_DEFAULTS["leadtime_min"])
    parser.add_argument("--leadtime-max", type=int, default=SCENARIO_SAMPLING_DEFAULTS["leadtime_max"])
    parser.add_argument("--cust-order-cycle", type=int, default=7)
    parser.add_argument(
        "--leadtime-mode",
        type=str,
        default=SCENARIO_SAMPLING_DEFAULTS["leadtime_mode"],
        choices=["shared", "per_material_random"],
    )
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument(
        "--same-scenario-across-r",
        action="store_true",
        help="Use the same scenario seed for R1/R3/R5 within each repetition (paired comparison).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("envs") / "diagnostics_outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = run_experiment(
        reps=args.reps,
        days=args.days,
        reorder_points=args.reorder_points,
        demand_min=args.demand_min,
        demand_max=args.demand_max,
        demand_mode=args.demand_mode,
        leadtime_min=args.leadtime_min,
        leadtime_max=args.leadtime_max,
        cust_order_cycle=args.cust_order_cycle,
        leadtime_mode=args.leadtime_mode,
        base_seed=args.base_seed,
        same_scenario_across_r=args.same_scenario_across_r,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reorder_point_experiment_results.csv"
    df.to_csv(csv_path, index=False)
    fig_path = plot_results(df, output_dir)

    summary = (
        df.groupby("reorder_point")["total_cost_200d"]
        .agg(["mean", "std", "min", "max"])
        .sort_index()
    )

    print(f"[Saved] CSV : {csv_path}")
    print(f"[Saved] Plot: {fig_path}")
    print("[Summary: total_cost_200d]")
    print(summary.to_string())


if __name__ == "__main__":
    main()
