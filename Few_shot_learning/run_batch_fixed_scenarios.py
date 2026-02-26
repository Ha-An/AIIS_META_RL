import os
import copy
import sys
import csv
import math
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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


def _format_duration(seconds):
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "n/a"
    total = max(0, int(round(seconds)))
    hours = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _append_progress_row(progress_csv: Path, row):
    fields = [
        "timestamp",
        "completed_runs",
        "total_runs",
        "progress_pct",
        "model",
        "case",
        "seed",
        "run_seconds",
        "elapsed_seconds",
        "eta_seconds",
        "eta_finish_local",
    ]
    exists = progress_csv.exists()
    with open(progress_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _compute_paired_stats(rows, log_root: Path, write_csv: bool = False):
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
                    if not shared_seeds:
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

    paired_csv = None
    if write_csv:
        paired_csv = log_root / "fewshot_fixed_paired_stats.csv"
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


def _plot_paired_stats(paired_rows, log_root: Path, save_mean_diff: bool = False):
    if not paired_rows:
        return None, None

    fig1_path = None
    if save_mean_diff:
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

            for xx, yy, rr in zip(x, y, pts):
                p_val = _effective_pvalue(rr)
                if not math.isnan(p_val) and p_val < 0.05:
                    ax1.scatter(xx, yy, color="red", marker="*", s=80, zorder=4)

        ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.set_title("Fixed Scenario: Paired Mean Difference by Adaptation Step")
        ax1.set_xlabel("Adaptation Step")
        ax1.set_ylabel("Mean Difference (model_a - model_b)")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)
        fig1.tight_layout()
        fig1_path = log_root / "fewshot_fixed_paired_mean_diff.png"
        fig1.savefig(fig1_path, dpi=150)
        plt.close(fig1)

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
    ax2.set_title(f"Fixed Scenario: Paired p-value Heatmap ({model_a} vs {model_b})")
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
    fig2_path = log_root / "fewshot_fixed_paired_pvalue_heatmap.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    return fig1_path, fig2_path


def _plot_case_boxplots(seed_rows, log_root: Path):
    """
    Per scenario case boxplot:
    distribution across seeds for each (model, adaptation_step).
    """
    out_paths = []
    cases = sorted({r["case"] for r in seed_rows})
    steps = sorted({int(r["adapt_step"]) for r in seed_rows})
    models = sorted({r["model"] for r in seed_rows})

    for case_name in cases:
        labels = []
        data = []
        for model_name in models:
            for step in steps:
                vals = [
                    float(r["seed_mean_total_cost"])
                    for r in seed_rows
                    if r["case"] == case_name and r["model"] == model_name and int(r["adapt_step"]) == step
                ]
                if vals:
                    labels.append(f"{model_name}\nstep {step}")
                    data.append(vals)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(f"Fixed Scenario Total Cost by Model/Adaptation Step: {case_name}")
        ax.set_xlabel("Model and Adaptation Step")
        ax.set_ylabel("Seed-level Mean Total Cost")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        safe_case = case_name.replace(" ", "_")
        out = log_root / f"fewshot_fixed_boxplot_{safe_case}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        out_paths.append(out)

    return out_paths


def _write_boxplot_tensorboard(seed_rows, log_root: Path):
    """
    Write only boxplot figures to TensorBoard (minimal logging).
    One run per fixed scenario case.
    """
    out_dirs = []
    cases = sorted({r["case"] for r in seed_rows})
    steps = sorted({int(r["adapt_step"]) for r in seed_rows})
    models = sorted({r["model"] for r in seed_rows})

    for case_name in cases:
        labels = []
        data = []
        for model_name in models:
            for step in steps:
                vals = [
                    float(r["seed_mean_total_cost"])
                    for r in seed_rows
                    if r["case"] == case_name and r["model"] == model_name and int(r["adapt_step"]) == step
                ]
                if vals:
                    labels.append(f"{model_name}\nstep {step}")
                    data.append(vals)
        if not data:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(f"Fixed Scenario Total Cost by Model/Adaptation Step: {case_name}")
        ax.set_xlabel("Model and Adaptation Step")
        ax.set_ylabel("Seed-level Mean Total Cost")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        tb_dir = log_root / f"FixedSummary__{case_name}"
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_figure("FewShot/TotalCostBoxplot", fig, global_step=0)
        writer.flush()
        writer.close()
        plt.close(fig)
        out_dirs.append(tb_dir)

    return out_dirs


def _summarize_fixed_seed_stats(log_root: Path, rows):
    if not rows:
        return None, None, None, [], rows

    paired_csv, paired_rows = _compute_paired_stats(rows, log_root, write_csv=False)
    paired_diff_fig, paired_pval_fig = _plot_paired_stats(paired_rows, log_root, save_mean_diff=False)
    case_boxplots = _plot_case_boxplots(rows, log_root)
    return paired_csv, paired_diff_fig, paired_pval_fig, case_boxplots, rows


def main():
    try:
        # Keep long-running batch logs visible in real time.
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    orig_model = copy.deepcopy(cfg.MODEL_CONFIG)
    orig_eval = copy.deepcopy(cfg.EVAL_CONFIG)
    orig_scenario = copy.deepcopy(cfg.SCENARIO_DIST_CONFIG)
    orig_model_path = cfg.PRETRAINED_MODEL_PATH
    orig_box_cfg = copy.deepcopy(cfg.COST_BOXPLOT_CONFIG)

    try:
        log_root = Path(__file__).parent / "Tensorboard_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_csv = log_root / f"fewshot_fixed_progress_{run_tag}.csv"
        progress_txt = log_root / "fewshot_fixed_progress_latest.txt"
        configured_cases = list(cfg.FIXED_SCENARIO_CASES)
        active_case_names = [str(x) for x in getattr(cfg, "FIXED_ACTIVE_CASES", [])]
        if active_case_names:
            configured_cases = [c for c in configured_cases if str(c.get("case", "")) in active_case_names]
            if not configured_cases:
                raise RuntimeError(
                    f"FIXED_ACTIVE_CASES has no matching entries: {active_case_names}"
                )
        total_runs = len(cfg.EXPERIMENTS) * len(configured_cases) * len(cfg.FIXED_SEEDS)
        completed_runs = 0
        batch_start = time.time()
        print(f"[FewShot-Fixed] Total planned runs: {total_runs}")
        print("[FewShot-Fixed] Active cases: " + ", ".join(str(c.get("case", "")) for c in configured_cases))
        print(f"[FewShot-Fixed] Progress CSV: {progress_csv}")
        print(f"[FewShot-Fixed] Progress TXT: {progress_txt}")

        seed_rows_this_batch = []
        # For fixed-scenario comparison, use seed-level distribution only.
        cfg.COST_BOXPLOT_CONFIG["adapt_steps"] = list(getattr(cfg, "FIXED_COST_BOXPLOT_ADAPT_STEPS", [0, 5, 10]))
        cfg.COST_BOXPLOT_CONFIG["repetitions"] = int(getattr(cfg, "FIXED_COST_BOXPLOT_REPETITIONS", 1))
        fixed_eval_overrides = dict(getattr(cfg, "EVAL_CONFIG_OVERRIDE_FIXED", {}))

        for exp in cfg.EXPERIMENTS:
            exp_name = str(exp.get("name", "Experiment"))
            exp_model_path = str(exp.get("pretrained_model_path", cfg.PRETRAINED_MODEL_PATH))

            for case_def in configured_cases:
                case_name = str(case_def["case"])

                for seed in cfg.FIXED_SEEDS:
                    cfg.MODEL_CONFIG = copy.deepcopy(orig_model)
                    cfg.EVAL_CONFIG = copy.deepcopy(orig_eval)
                    cfg.SCENARIO_DIST_CONFIG = copy.deepcopy(orig_scenario)
                    cfg.PRETRAINED_MODEL_PATH = exp_model_path

                    _apply_overrides(cfg.MODEL_CONFIG, exp.get("model_config", {}))
                    _apply_overrides(cfg.EVAL_CONFIG, fixed_eval_overrides)
                    cfg.EVAL_CONFIG["seed"] = int(seed)

                    cfg.SCENARIO_DIST_CONFIG = cfg.build_fixed_scenario_dist_config(case_def)

                    run_name = f"Fixed__{exp_name}__{case_name}__seed{seed}"
                    os.environ["FEWSHOT_DISABLE_TENSORBOARD"] = "1"
                    os.environ["FEWSHOT_DISABLE_RUN_ARTIFACTS"] = "1"
                    os.environ["FEWSHOT_FLAT_OUTPUT"] = "1"
                    os.environ["FEWSHOT_SKIP_ADAPTATION_CURVE"] = (
                        "1" if bool(getattr(cfg, "FIXED_SKIP_ADAPTATION_CURVE", True)) else "0"
                    )
                    run_idx = completed_runs + 1
                    print(f"\n[FEWSHOT-FIXED] Running ({run_idx}/{total_runs}): {run_name}")
                    print(f"  model_path: {cfg.PRETRAINED_MODEL_PATH}")
                    if "fixed_scenarios" in cfg.SCENARIO_DIST_CONFIG:
                        one = cfg.SCENARIO_DIST_CONFIG["fixed_scenarios"][0]
                        print(
                            "  scenario_config: fixed_scenarios="
                            f"{len(cfg.SCENARIO_DIST_CONFIG['fixed_scenarios'])}, "
                            f"demand={one['DEMAND']}, leadtime_example={one['LEADTIME'][0]}"
                        )
                    else:
                        print(f"  scenario_config: {cfg.SCENARIO_DIST_CONFIG}")

                    run_start = time.time()
                    result = eval_few_shot.main() or {}
                    run_seconds = time.time() - run_start
                    box_rows = result.get("boxplot_rows", []) or []
                    by_step = {}
                    for r in box_rows:
                        step = int(r["adapt_step"])
                        by_step.setdefault(step, []).append(float(r["mean_total_cost"]))

                    for step, vals in sorted(by_step.items()):
                        seed_rows_this_batch.append(
                            {
                                "model": exp_name,
                                "case": case_name,
                                "seed": int(seed),
                                "adapt_step": int(step),
                                "seed_mean_total_cost": float(sum(vals) / len(vals)) if vals else float("nan"),
                                "seed_n_reps": int(len(vals)),
                            }
                        )
                    print(f"  merged_rows: {sum(len(v) for v in by_step.values())}")

                    completed_runs += 1
                    elapsed_seconds = time.time() - batch_start
                    avg_run_seconds = elapsed_seconds / completed_runs if completed_runs > 0 else float("nan")
                    remaining_runs = max(total_runs - completed_runs, 0)
                    eta_seconds = avg_run_seconds * remaining_runs if remaining_runs > 0 else 0.0
                    progress_pct = (completed_runs / total_runs * 100.0) if total_runs > 0 else 100.0
                    eta_finish = datetime.now() + timedelta(seconds=eta_seconds)

                    progress_row = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "completed_runs": completed_runs,
                        "total_runs": total_runs,
                        "progress_pct": f"{progress_pct:.2f}",
                        "model": exp_name,
                        "case": case_name,
                        "seed": int(seed),
                        "run_seconds": f"{run_seconds:.2f}",
                        "elapsed_seconds": f"{elapsed_seconds:.2f}",
                        "eta_seconds": f"{eta_seconds:.2f}",
                        "eta_finish_local": eta_finish.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    _append_progress_row(progress_csv, progress_row)
                    progress_txt.write_text(
                        (
                            f"completed_runs={completed_runs}\n"
                            f"total_runs={total_runs}\n"
                            f"progress_pct={progress_pct:.2f}\n"
                            f"last_model={exp_name}\n"
                            f"last_case={case_name}\n"
                            f"last_seed={int(seed)}\n"
                            f"last_run_seconds={run_seconds:.2f}\n"
                            f"elapsed_seconds={elapsed_seconds:.2f}\n"
                            f"eta_seconds={eta_seconds:.2f}\n"
                            f"eta_finish_local={eta_finish.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        ),
                        encoding="utf-8",
                    )
                    print(
                        "[FEWSHOT-FIXED][PROGRESS] "
                        f"{completed_runs}/{total_runs} ({progress_pct:.2f}%), "
                        f"elapsed={_format_duration(elapsed_seconds)}, "
                        f"ETA={_format_duration(eta_seconds)}, "
                        f"finish~{eta_finish.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

        paired_csv, paired_diff_fig, paired_pval_fig, case_boxplots, seed_rows = _summarize_fixed_seed_stats(
            log_root, rows=seed_rows_this_batch
        )

        if case_boxplots or paired_pval_fig is not None:
            tb_case_dirs = _write_boxplot_tensorboard(seed_rows, log_root)
            if paired_pval_fig is not None:
                print(f"[FewShot-Fixed] Saved paired p-value heatmap: {paired_pval_fig}")
            else:
                print("[FewShot-Fixed] Paired p-value heatmap skipped (need >=2 models).")
            for boxplot in case_boxplots:
                print(f"[FewShot-Fixed] Saved case boxplot: {boxplot}")
            for tb_dir in tb_case_dirs:
                print(f"[FewShot-Fixed] Saved TensorBoard boxplot run: {tb_dir}")
        else:
            print("\n[FewShot-Fixed] No minimal artifacts generated.")
    finally:
        cfg.MODEL_CONFIG = orig_model
        cfg.EVAL_CONFIG = orig_eval
        cfg.SCENARIO_DIST_CONFIG = orig_scenario
        cfg.PRETRAINED_MODEL_PATH = orig_model_path
        cfg.COST_BOXPLOT_CONFIG = orig_box_cfg
        os.environ.pop("FEWSHOT_RUN_NAME", None)
        os.environ.pop("FEWSHOT_DISABLE_TENSORBOARD", None)
        os.environ.pop("FEWSHOT_DISABLE_RUN_ARTIFACTS", None)
        os.environ.pop("FEWSHOT_FLAT_OUTPUT", None)
        os.environ.pop("FEWSHOT_SKIP_ADAPTATION_CURVE", None)


if __name__ == "__main__":
    main()
