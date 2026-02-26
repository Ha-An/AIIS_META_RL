import os
import copy
import sys
import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import Few_shot_learning.config as cfg
import Few_shot_learning.eval_few_shot as eval_few_shot

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


def _apply_overrides(base_cfg, overrides):
    for k, v in overrides.items():
        base_cfg[k] = v


def _find_run_dir(log_root: Path, run_name_prefix: str):
    candidates = [p for p in log_root.glob(f"{run_name_prefix}*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_run_name(run_dir_name: str):
    # New format: {model}__{case}__seed{seed}
    # Legacy format: {case}_seed{seed}
    if "__" in run_dir_name and "__seed" in run_dir_name:
        try:
            model_name, rest = run_dir_name.split("__", 1)
            case_name, seed_part = rest.rsplit("__seed", 1)
            seed = int(seed_part.split("_")[0])
            return model_name, case_name, seed
        except Exception:
            return None

    if "_seed" in run_dir_name:
        try:
            case_name, seed_part = run_dir_name.rsplit("_seed", 1)
            seed = int(seed_part.split("_")[0])
            return "UnknownModel", case_name, seed
        except Exception:
            return None

    return None


def _compute_paired_stats(rows, log_root: Path):
    """
    Paired comparison across models using same (case, adapt_step, seed).
    - Paired t-test (if scipy available)
    - Wilcoxon signed-rank (if scipy available)
    """
    model_names = sorted({r["model"] for r in rows})
    if len(model_names) < 2:
        return None, []

    cases = sorted({r["case"] for r in rows})
    steps = sorted({int(r["adapt_step"]) for r in rows})

    paired_rows = []
    for case_name in cases:
        for step in steps:
            by_model = {}
            for m in model_names:
                vals = {
                    int(r["seed"]): float(r["seed_mean_total_cost"])
                    for r in rows
                    if r["model"] == m and r["case"] == case_name and int(r["adapt_step"]) == step
                }
                by_model[m] = vals

            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a = model_names[i]
                    model_b = model_names[j]
                    shared_seeds = sorted(set(by_model[model_a].keys()) & set(by_model[model_b].keys()))
                    if len(shared_seeds) == 0:
                        continue

                    x = [by_model[model_a][s] for s in shared_seeds]
                    y = [by_model[model_b][s] for s in shared_seeds]
                    diffs = [a - b for a, b in zip(x, y)]

                    n = len(diffs)
                    mean_diff = float(statistics.mean(diffs))
                    std_diff = float(statistics.stdev(diffs)) if n >= 2 else 0.0

                    t_stat = float("nan")
                    p_ttest = float("nan")
                    w_stat = float("nan")
                    p_wilcoxon = float("nan")

                    if scipy_stats is not None and n >= 2:
                        try:
                            t_res = scipy_stats.ttest_rel(x, y, nan_policy="omit")
                            t_stat = float(t_res.statistic)
                            p_ttest = float(t_res.pvalue)
                        except Exception:
                            pass

                    if scipy_stats is not None and n >= 1:
                        try:
                            if any(abs(d) > 1e-12 for d in diffs):
                                w_res = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                                w_stat = float(w_res.statistic)
                                p_wilcoxon = float(w_res.pvalue)
                            else:
                                w_stat = 0.0
                                p_wilcoxon = 1.0
                        except Exception:
                            pass

                    paired_rows.append(
                        {
                            "case": case_name,
                            "adapt_step": step,
                            "model_a": model_a,
                            "model_b": model_b,
                            "n_pairs": n,
                            "mean_diff_a_minus_b": mean_diff,
                            "std_diff_a_minus_b": std_diff,
                            "t_stat": t_stat,
                            "p_value_ttest": p_ttest,
                            "w_stat": w_stat,
                            "p_value_wilcoxon": p_wilcoxon,
                        }
                    )

    if not paired_rows:
        return None, []

    paired_csv = log_root / "fewshot_paired_stats.csv"
    with open(paired_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "adapt_step",
                "model_a",
                "model_b",
                "n_pairs",
                "mean_diff_a_minus_b",
                "std_diff_a_minus_b",
                "t_stat",
                "p_value_ttest",
                "w_stat",
                "p_value_wilcoxon",
            ],
        )
        writer.writeheader()
        writer.writerows(paired_rows)

    return paired_csv, paired_rows


def _effective_pvalue(row):
    p_w = float(row.get("p_value_wilcoxon", float("nan")))
    if not math.isnan(p_w):
        return p_w
    return float(row.get("p_value_ttest", float("nan")))


def _plot_paired_stats(paired_rows, log_root: Path):
    if not paired_rows:
        return None, None

    # Plot 1: mean diff (model_a - model_b) by adaptation step, grouped by case.
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    curve_keys = sorted({(r["model_a"], r["model_b"], r["case"]) for r in paired_rows})
    for model_a, model_b, case_name in curve_keys:
        pts = [
            r for r in paired_rows
            if r["model_a"] == model_a and r["model_b"] == model_b and r["case"] == case_name
        ]
        pts = sorted(pts, key=lambda x: int(x["adapt_step"]))
        if not pts:
            continue

        x = [int(r["adapt_step"]) for r in pts]
        y = [float(r["mean_diff_a_minus_b"]) for r in pts]
        label = f"{case_name} ({model_a}-{model_b})"
        ax1.plot(x, y, marker="o", label=label)

        # Mark statistically significant points (p < 0.05) with red stars.
        for xx, yy, rr in zip(x, y, pts):
            p_val = _effective_pvalue(rr)
            if not math.isnan(p_val) and p_val < 0.05:
                ax1.scatter(xx, yy, color="red", marker="*", s=80, zorder=4)

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_title("Paired Mean Difference by Adaptation Step (model_a - model_b)")
    ax1.set_xlabel("Adaptation Step")
    ax1.set_ylabel("Mean Difference of Seed-level Total Cost")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1_path = log_root / "fewshot_paired_mean_diff.png"
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    # Plot 2: p-value heatmap (first model-pair, Wilcoxon prioritized).
    pair_keys = sorted({(r["model_a"], r["model_b"]) for r in paired_rows})
    if not pair_keys:
        return fig1_path, None

    model_a, model_b = pair_keys[0]
    sub = [r for r in paired_rows if r["model_a"] == model_a and r["model_b"] == model_b]
    cases = sorted({r["case"] for r in sub})
    steps = sorted({int(r["adapt_step"]) for r in sub})
    if not cases or not steps:
        return fig1_path, None

    pval_matrix = np.full((len(cases), len(steps)), np.nan, dtype=float)
    case_to_i = {c: i for i, c in enumerate(cases)}
    step_to_j = {s: j for j, s in enumerate(steps)}
    for r in sub:
        i = case_to_i[r["case"]]
        j = step_to_j[int(r["adapt_step"])]
        pval_matrix[i, j] = _effective_pvalue(r)

    fig2, ax2 = plt.subplots(figsize=(9, 4 + 0.5 * len(cases)))
    im = ax2.imshow(pval_matrix, aspect="auto", interpolation="nearest", cmap="viridis_r", vmin=0.0, vmax=0.1)
    ax2.set_title(f"Paired p-value Heatmap ({model_a} vs {model_b})")
    ax2.set_xlabel("Adaptation Step")
    ax2.set_ylabel("Scenario Case")
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels([str(s) for s in steps])
    ax2.set_yticks(range(len(cases)))
    ax2.set_yticklabels(cases)

    for i in range(len(cases)):
        for j in range(len(steps)):
            val = pval_matrix[i, j]
            text = "nan" if np.isnan(val) else f"{val:.3f}"
            color = "white" if (not np.isnan(val) and val < 0.05) else "black"
            ax2.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("p-value (Wilcoxon prioritized)")
    fig2.tight_layout()
    fig2_path = log_root / "fewshot_paired_pvalue_heatmap.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    return fig1_path, fig2_path


def _summarize_seed_stats(log_root: Path, run_records=None):
    """
    Aggregate per-run boxplot CSVs into seed-level statistics by model/case/adapt_step.
    """
    rows = []

    target_records = []
    if run_records is None:
        for p in [d for d in log_root.iterdir() if d.is_dir()]:
            parsed = _parse_run_name(p.name)
            if parsed is None:
                continue
            model_name, case_name, seed = parsed
            target_records.append(
                {
                    "model": model_name,
                    "case": case_name,
                    "seed": seed,
                    "run_dir": p,
                }
            )
    else:
        for rec in run_records:
            if rec is None:
                continue
            rd = Path(rec["run_dir"])
            if not rd.is_dir():
                continue
            target_records.append(rec)

    for rec in target_records:
        run_dir = Path(rec["run_dir"])
        csv_path = run_dir / "fewshot_total_cost_boxplot_data.csv"
        if not csv_path.exists():
            continue

        model_name = str(rec["model"])
        case_name = str(rec["case"])
        seed = int(rec["seed"])

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            by_step = {}
            for r in reader:
                step = int(r["adapt_step"])
                cost = float(r["mean_total_cost"])
                by_step.setdefault(step, []).append(cost)

            for step, costs in by_step.items():
                rows.append(
                    {
                        "model": model_name,
                        "case": case_name,
                        "seed": seed,
                        "adapt_step": step,
                        "seed_mean_total_cost": float(sum(costs) / len(costs)) if costs else float("nan"),
                        "seed_n_reps": len(costs),
                    }
                )

    if not rows:
        return None, None, None, None, None, None, None

    seed_csv = log_root / "fewshot_seed_level_summary.csv"
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "case", "seed", "adapt_step", "seed_mean_total_cost", "seed_n_reps"],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for r in rows:
        key = (r["model"], r["case"], int(r["adapt_step"]))
        grouped.setdefault(key, []).append(float(r["seed_mean_total_cost"]))

    agg_rows = []
    for (model_name, case_name, step), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = len(vals)
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if n >= 2 else 0.0
        ci95 = 1.96 * std_v / math.sqrt(n) if n >= 2 else 0.0
        agg_rows.append(
            {
                "model": model_name,
                "case": case_name,
                "adapt_step": step,
                "n_seeds": n,
                "mean_seed_cost": mean_v,
                "std_seed_cost": std_v,
                "ci95_halfwidth": ci95,
                "ci95_low": mean_v - ci95,
                "ci95_high": mean_v + ci95,
            }
        )

    agg_csv = log_root / "fewshot_stats_by_model_case_step.csv"
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "case",
                "adapt_step",
                "n_seeds",
                "mean_seed_cost",
                "std_seed_cost",
                "ci95_halfwidth",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    paired_csv, paired_rows = _compute_paired_stats(rows, log_root)
    paired_diff_fig, paired_pval_fig = _plot_paired_stats(paired_rows, log_root)

    # Integrated boxplot by adaptation step (all models/cases together)
    step_order = sorted({int(r["adapt_step"]) for r in rows})
    data_by_step = []
    labels_by_step = []
    for step in step_order:
        vals = [r["seed_mean_total_cost"] for r in rows if int(r["adapt_step"]) == step]
        if vals:
            data_by_step.append(vals)
            labels_by_step.append(str(step))

    if data_by_step:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.boxplot(data_by_step, tick_labels=labels_by_step, showmeans=True)
        ax1.set_title("Integrated Total Cost Distribution by Adaptation Step")
        ax1.set_xlabel("Adaptation Step")
        ax1.set_ylabel("Seed-level Mean Total Cost")
        ax1.grid(alpha=0.3)
        fig1.tight_layout()
        fig1_path = log_root / "fewshot_integrated_boxplot_by_step.png"
        fig1.savefig(fig1_path, dpi=150)
        plt.close(fig1)
    else:
        fig1_path = None

    if agg_rows:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        curve_keys = sorted({(r["model"], r["case"]) for r in agg_rows})
        for model_name, case_name in curve_keys:
            points = [r for r in agg_rows if r["model"] == model_name and r["case"] == case_name]
            points = sorted(points, key=lambda x: int(x["adapt_step"]))
            x = [int(r["adapt_step"]) for r in points]
            y = [float(r["mean_seed_cost"]) for r in points]
            ci = [float(r["ci95_halfwidth"]) for r in points]
            if not x:
                continue
            label = f"{model_name}:{case_name}"
            ax2.plot(x, y, marker="o", label=label)
            low = [yy - cc for yy, cc in zip(y, ci)]
            high = [yy + cc for yy, cc in zip(y, ci)]
            ax2.fill_between(x, low, high, alpha=0.12)

        ax2.set_title("Model/Case-wise Mean Total Cost with 95% CI")
        ax2.set_xlabel("Adaptation Step")
        ax2.set_ylabel("Mean Seed-level Total Cost")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7, ncol=2)
        fig2.tight_layout()
        fig2_path = log_root / "fewshot_model_case_mean_ci.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)
    else:
        fig2_path = None

    return seed_csv, agg_csv, paired_csv, fig1_path, fig2_path, paired_diff_fig, paired_pval_fig


def main():
    # Snapshot original configs so we can restore between runs
    orig_model = copy.deepcopy(cfg.MODEL_CONFIG)
    orig_eval = copy.deepcopy(cfg.EVAL_CONFIG)
    orig_scenario = copy.deepcopy(cfg.SCENARIO_DIST_CONFIG)
    orig_model_path = cfg.PRETRAINED_MODEL_PATH

    try:
        log_root = Path(__file__).parent / "Tensorboard_logs"
        run_records_this_batch = []
        random_eval_overrides = dict(getattr(cfg, "EVAL_CONFIG_OVERRIDE_RANDOM", cfg.EVAL_CONFIG_OVERRIDE))

        for exp in cfg.EXPERIMENTS:
            exp_name = str(exp.get("name", "Experiment"))
            exp_model_path = str(exp.get("pretrained_model_path", cfg.PRETRAINED_MODEL_PATH))

            for case_def in cfg.SCENARIO_CASES:
                case_name = str(case_def["case"])

                for seed in cfg.SEEDS:
                    # Reset configs each run
                    cfg.MODEL_CONFIG = copy.deepcopy(orig_model)
                    cfg.EVAL_CONFIG = copy.deepcopy(orig_eval)
                    cfg.SCENARIO_DIST_CONFIG = copy.deepcopy(orig_scenario)
                    cfg.PRETRAINED_MODEL_PATH = exp_model_path

                    # Apply overrides from config.py
                    _apply_overrides(cfg.MODEL_CONFIG, exp.get("model_config", {}))
                    _apply_overrides(cfg.EVAL_CONFIG, random_eval_overrides)
                    cfg.EVAL_CONFIG["seed"] = seed
                    case_cfg = dict(case_def)
                    case_cfg["seed"] = seed
                    cfg.SCENARIO_DIST_CONFIG = cfg.build_scenario_dist_config(case_cfg)

                    run_name = f"{exp_name}__{case_name}__seed{seed}"
                    os.environ["FEWSHOT_RUN_NAME"] = run_name
                    print(f"\n[FEWSHOT-RANDOM] Running: {run_name}")
                    print(f"  model_path: {cfg.PRETRAINED_MODEL_PATH}")
                    print(f"  scenario_config: {cfg.SCENARIO_DIST_CONFIG}")

                    eval_few_shot.main()

                    run_dir = _find_run_dir(log_root, run_name)
                    if run_dir is not None:
                        run_records_this_batch.append(
                            {
                                "model": exp_name,
                                "case": case_name,
                                "seed": int(seed),
                                "run_dir": run_dir,
                            }
                        )
                        print(f"  run_dir: {run_dir}")
                    else:
                        print("  run_dir: NOT FOUND")

        seed_csv, agg_csv, paired_csv, fig1_path, fig2_path, paired_diff_fig, paired_pval_fig = _summarize_seed_stats(
            log_root,
            run_records=run_records_this_batch,
        )
        if seed_csv and agg_csv:
            print(f"\n[FewShot-Random] Saved seed-level summary: {seed_csv}")
            print(f"[FewShot-Random] Saved stats by model/case/step: {agg_csv}")
            if paired_csv is not None:
                print(f"[FewShot-Random] Saved paired stats: {paired_csv}")
            else:
                print("[FewShot-Random] Paired stats skipped (need >=2 models).")
            if paired_diff_fig is not None:
                print(f"[FewShot-Random] Saved paired mean-diff plot: {paired_diff_fig}")
            if paired_pval_fig is not None:
                print(f"[FewShot-Random] Saved paired p-value heatmap: {paired_pval_fig}")
            if fig1_path is not None:
                print(f"[FewShot-Random] Saved integrated boxplot: {fig1_path}")
            if fig2_path is not None:
                print(f"[FewShot-Random] Saved model/case mean+/-CI plot: {fig2_path}")
        else:
            print("\n[FewShot-Random] No boxplot CSV found; skipped summary aggregation.")
    finally:
        # Restore original configs
        cfg.MODEL_CONFIG = orig_model
        cfg.EVAL_CONFIG = orig_eval
        cfg.SCENARIO_DIST_CONFIG = orig_scenario
        cfg.PRETRAINED_MODEL_PATH = orig_model_path
        os.environ.pop("FEWSHOT_RUN_NAME", None)


if __name__ == "__main__":
    main()
