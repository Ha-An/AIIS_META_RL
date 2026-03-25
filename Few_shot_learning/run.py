import argparse
from pathlib import Path

from Few_shot_learning.core import EvalSettings, default_settings, run_experiment


def _parse_int_list(value: str):
    return [int(chunk.strip()) for chunk in str(value).split(",") if chunk.strip()]


def _parse_name_list(value: str):
    return [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomized few-shot PPO vs ProMP evaluator")
    parser.add_argument("--shots", type=str, default=None, help="Comma-separated shot list, e.g. 0,1,2,3")
    parser.add_argument("--modes", type=str, default=None, help="Comma-separated modes, e.g. stationary,nonstationary")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names, e.g. PPO,ProMP")
    parser.add_argument("--stationary-scenarios", type=int, default=None, help="Override stationary randomized task count")
    parser.add_argument("--nonstationary-sequences", type=int, default=None, help="Override nonstationary randomized task count")
    parser.add_argument("--query-rollouts", type=int, default=None, help="Override query trajectories per task")
    parser.add_argument("--adapt-updates", type=int, default=None, help="Override number of inner updates")
    parser.add_argument("--output-root", type=str, default=None, help="Optional custom output directory")
    parser.add_argument("--smoke", action="store_true", help="Run a very small smoke test")
    args = parser.parse_args()

    settings = default_settings()

    if args.shots:
        settings.shots = _parse_int_list(args.shots)
    if args.modes:
        settings.environment_modes = _parse_name_list(args.modes)
    if args.stationary_scenarios is not None:
        settings.stationary_scenario_count = max(1, int(args.stationary_scenarios))
    if args.nonstationary_sequences is not None:
        settings.nonstationary_sequence_count = max(1, int(args.nonstationary_sequences))
    if args.query_rollouts is not None:
        settings.query_rollout_per_task = max(1, int(args.query_rollouts))
    if args.adapt_updates is not None:
        settings.adapt_updates = max(1, int(args.adapt_updates))

    if args.smoke:
        settings.shots = [0, 1]
        settings.environment_modes = ["stationary", "nonstationary"]
        settings.stationary_scenario_count = 1
        settings.nonstationary_sequence_count = 1
        settings.query_rollout_per_task = 1
        settings.adapt_updates = 1

    selected_models = _parse_name_list(args.models) if args.models else None
    output_root = Path(args.output_root) if args.output_root else None

    report = run_experiment(
        settings=settings,
        selected_models=selected_models,
        output_root=output_root,
    )

    print("[FewShot-Randomized] run complete")
    print(f"[FewShot-Randomized] run_dir: {report['run_dir']}")
    print(f"[FewShot-Randomized] summary_csv: {report['summary_csv']}")
    print(f"[FewShot-Randomized] progress_txt: {report['progress_txt']}")


if __name__ == "__main__":
    main()
