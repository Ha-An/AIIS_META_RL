import argparse
import copy
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Evaluation import config as cfg

from AIIS_META.Agents.Categorical.Meta_Categorical import MetaCategoricalAgent
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
from envs.config_RL import ACTION_SPACE
from envs.config_SimPy import ASSEMBLY_PROCESS, I, P, SIM_TIME
import envs.environment as simpy_env_module
from envs.log_SimPy import DAILY_COST_REPORT, DAILY_EVENTS, DAILY_REPORTS, STATE_DICT
from envs.promp_env import MetaEnv
import envs.scenarios as scenario_factory
from envs.simpy_diagnostics import _start_heuristic_processes
import Few_shot_learning.config as fewshot_cfg


@dataclass
class EpisodeResult:
    total_cost: float
    holding_cost: float
    process_cost: float
    delivery_cost: float
    order_cost: float
    shortage_cost: float


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _copy_scenario(scenario: Dict) -> Dict:
    return {
        "DEMAND": copy.deepcopy(scenario["DEMAND"]),
        "LEADTIME": copy.deepcopy(scenario["LEADTIME"]),
    }


def _apply_scenario_inplace(target: Dict, source: Dict):
    target["DEMAND"].clear()
    target["DEMAND"].update(copy.deepcopy(source["DEMAND"]))
    target["LEADTIME"][:] = [copy.deepcopy(x) for x in source["LEADTIME"]]


def _segment_index_for_day(day: int, segments: Sequence[Tuple[int, int]]) -> int:
    for idx, (start_day, end_day) in enumerate(segments):
        if start_day <= day <= end_day:
            return idx
    raise ValueError(f"Day {day} is not covered by NONSTATIONARY_SEGMENTS.")


def _material_ids() -> List[int]:
    return [
        item_id
        for item_id, item_info in I[ASSEMBLY_PROCESS].items()
        if item_info.get("TYPE") == "Material"
    ]


def _restore_lot_sizes(original_lot_sizes: Dict[int, int]):
    for material_id, lot_size in original_lot_sizes.items():
        I[ASSEMBLY_PROCESS][material_id]["LOT_SIZE_ORDER"] = lot_size


def _extract_agent_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint format must be dict-like.")

    if "agent_state_dict" in state and isinstance(state["agent_state_dict"], dict):
        return state["agent_state_dict"]

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    prefixed = {
        k[len("agent."):]: v
        for k, v in state.items()
        if isinstance(k, str) and k.startswith("agent.") and torch.is_tensor(v)
    }
    if prefixed:
        return prefixed

    tensor_state = {
        k: v for k, v in state.items() if isinstance(k, str) and torch.is_tensor(v)
    }
    if tensor_state:
        return tensor_state

    raise ValueError("Could not extract agent parameters from checkpoint.")


def _build_agent(policy_dist: str, device: torch.device):
    env = MetaEnv()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    if policy_dist == "categorical":
        mlp = SimpleMLP(
            input_dim=obs_dim,
            output_dim=act_dim * len(ACTION_SPACE),
            hidden_layers=cfg.AGENT_HIDDEN_LAYERS,
        )
        agent = MetaCategoricalAgent(
            mlp=mlp,
            num_tasks=1,
            action_dim=act_dim,
            num_actions=len(ACTION_SPACE),
        )
    elif policy_dist == "gaussian":
        mlp = SimpleMLP(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_layers=cfg.AGENT_HIDDEN_LAYERS,
        )
        agent = MetaGaussianAgent(
            mlp=mlp,
            num_tasks=1,
            learn_std=cfg.LEARN_STD,
        )
    else:
        raise ValueError(f"Unsupported policy_dist: {policy_dist}")

    agent.to(device)
    return agent


def _load_pretrained_agent(checkpoint_path: str, policy_dist: str, device: torch.device):
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    agent = _build_agent(policy_dist=policy_dist, device=device)
    state = torch.load(checkpoint_file, map_location=device)
    agent_state = _extract_agent_state_dict(state)
    info = agent.load_state_dict(agent_state, strict=False)
    agent.eval()
    print(
        f"[Model] Loaded {checkpoint_file} "
        f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})"
    )
    return agent


def _deterministic_action(agent, state, policy_dist: str, device: torch.device) -> np.ndarray:
    obs = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
    params = dict(agent.named_parameters())
    dist = agent.distribution(obs, params=params)
    if policy_dist == "categorical":
        action = torch.argmax(dist.logits, dim=-1).squeeze(0)
    else:
        action = dist.mean.squeeze(0)
    return action.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _episode_components_from_cost_dict(cost_dict: Dict[str, float]) -> EpisodeResult:
    holding = float(cost_dict.get("Holding cost", 0.0))
    process = float(cost_dict.get("Process cost", 0.0))
    delivery = float(cost_dict.get("Delivery cost", 0.0))
    order = float(cost_dict.get("Order cost", 0.0))
    shortage = float(cost_dict.get("Shortage cost", 0.0))
    total = holding + process + delivery + order + shortage
    return EpisodeResult(
        total_cost=total,
        holding_cost=holding,
        process_cost=process,
        delivery_cost=delivery,
        order_cost=order,
        shortage_cost=shortage,
    )


def _run_pretrained_episode(
    agent,
    policy_dist: str,
    scenario_schedule: Sequence[Dict],
    segments: Sequence[Tuple[int, int]],
    days: int,
    seed: int,
    method_name: str,
) -> EpisodeResult:
    _set_seed(seed)
    env = MetaEnv()
    runtime_scenario = _copy_scenario(scenario_schedule[0])
    env.set_task(runtime_scenario)
    state = env.reset()

    active_segment = 0
    total_cost = 0.0
    last_cost_dict = None

    for day in range(1, days + 1):
        new_segment = _segment_index_for_day(day, segments)
        if new_segment != active_segment:
            _apply_scenario_inplace(env.scenario, scenario_schedule[new_segment])
            active_segment = new_segment

        action = _deterministic_action(agent, state, policy_dist, cfg.DEVICE)
        state, reward, done, cost_dict = env.step(action)
        total_cost += -float(reward)
        last_cost_dict = dict(cost_dict)
        if done:
            break

    if last_cost_dict is None:
        raise RuntimeError(f"No step was executed for {method_name}.")

    components = _episode_components_from_cost_dict(last_cost_dict)
    components.total_cost = float(total_cost)
    return components


def _run_heuristic_episode(
    reorder_point: int,
    scenario_schedule: Sequence[Dict],
    segments: Sequence[Tuple[int, int]],
    days: int,
    seed: int,
) -> EpisodeResult:
    _set_seed(seed)

    from envs.simpy_diagnostics import _reset_global_logs  # local import to keep dependency explicit

    _reset_global_logs()
    material_ids = _material_ids()
    original_lot_sizes = {
        material_id: I[ASSEMBLY_PROCESS][material_id]["LOT_SIZE_ORDER"] for material_id in material_ids
    }
    runtime_scenario = _copy_scenario(scenario_schedule[0])

    try:
        (
            simpy_env,
            inventory_list,
            procurement_list,
            production_list,
            sales,
            customer,
            supplier_list,
            daily_events,
        ) = simpy_env_module.create_env(I, P, DAILY_EVENTS)

        ordered_qty_daily = {material_id: 0 for material_id in material_ids}
        _start_heuristic_processes(
            env_module=simpy_env_module,
            simpy_env=simpy_env,
            inventory_list=inventory_list,
            procurement_list=procurement_list,
            production_list=production_list,
            sales=sales,
            customer=customer,
            supplier_list=supplier_list,
            daily_events=daily_events,
            scenario=runtime_scenario,
            reorder_point=reorder_point,
            ordered_qty_daily=ordered_qty_daily,
        )
        simpy_env_module.update_daily_report(inventory_list)

        active_segment = 0
        total_cost = 0.0
        holding_total = 0.0
        process_total = 0.0
        delivery_total = 0.0
        order_total = 0.0
        shortage_total = 0.0

        for day in range(1, days + 1):
            new_segment = _segment_index_for_day(day, segments)
            if new_segment != active_segment:
                _apply_scenario_inplace(runtime_scenario, scenario_schedule[new_segment])
                active_segment = new_segment

            simpy_env.run(until=simpy_env.now + 24)
            simpy_env_module.update_daily_report(inventory_list)
            day_total = float(simpy_env_module.Cost.update_cost_log(inventory_list))

            total_cost += day_total
            holding_total += float(DAILY_COST_REPORT["Holding cost"])
            process_total += float(DAILY_COST_REPORT["Process cost"])
            delivery_total += float(DAILY_COST_REPORT["Delivery cost"])
            order_total += float(DAILY_COST_REPORT["Order cost"])
            shortage_total += float(DAILY_COST_REPORT["Shortage cost"])

            for material_id in material_ids:
                ordered_qty_daily[material_id] = 0
            sales.num_shortages = 0
            simpy_env_module.Cost.clear_cost()

        return EpisodeResult(
            total_cost=total_cost,
            holding_cost=holding_total,
            process_cost=process_total,
            delivery_cost=delivery_total,
            order_cost=order_total,
            shortage_cost=shortage_total,
        )
    finally:
        _restore_lot_sizes(original_lot_sizes)
        DAILY_EVENTS.clear()
        DAILY_REPORTS.clear()
        STATE_DICT.clear()


def _generate_stationary_scenarios(n: int, seed: int) -> List[Dict]:
    return scenario_factory.create_scenarios(
        num_scenarios=n,
        seed=seed,
        **cfg.SCENARIO_SAMPLING_OVERRIDES,
    )


def _generate_nonstationary_sequences(n: int, seed: int) -> List[List[Dict]]:
    pool = scenario_factory.create_scenarios(
        num_scenarios=n * 3,
        seed=seed,
        **cfg.SCENARIO_SAMPLING_OVERRIDES,
    )
    return [pool[i * 3:(i + 1) * 3] for i in range(n)]


def _build_fixed_scenarios_from_fewshot() -> List[Dict]:
    """
    Build fixed scenario list from Few_shot_learning.config.FIXED_SCENARIO_CASES.
    """
    fixed_scenarios = []
    for case_cfg in fewshot_cfg.FIXED_SCENARIO_CASES:
        scenario_cfg = fewshot_cfg.build_fixed_scenario_dist_config(case_cfg)
        fixed_pool = scenario_cfg.get("fixed_scenarios", [])
        if not fixed_pool:
            raise ValueError(f"No fixed_scenarios built for case: {case_cfg.get('case')}")
        scenario = _copy_scenario(fixed_pool[0])
        scenario["_case_name"] = str(case_cfg.get("case", f"case_{len(fixed_scenarios)+1}"))
        fixed_scenarios.append(scenario)
    if len(fixed_scenarios) != 5:
        print(
            f"[Warning] FIXED_SCENARIO_CASES has {len(fixed_scenarios)} cases "
            f"(requested design expects 5)."
        )
    return fixed_scenarios


def _generate_fixed_stationary_scenarios() -> List[Dict]:
    return _build_fixed_scenarios_from_fewshot()


def _generate_fixed_nonstationary_sequences(n: int, seed: int) -> List[List[Dict]]:
    base = _build_fixed_scenarios_from_fewshot()
    if len(base) == 0:
        raise ValueError("No fixed scenarios available.")
    rng = random.Random(seed)
    sequences = []
    # For each nonstationary case, each segment picks one scenario among the 5 fixed cases.
    for i in range(n):
        sequence = [
            _copy_scenario(rng.choice(base)),
            _copy_scenario(rng.choice(base)),
            _copy_scenario(rng.choice(base)),
        ]
        sequences.append(sequence)
    return sequences


def _plot_boxplot(df: pd.DataFrame, title: str, output_png: Path):
    methods_present = set(df["method"].unique())
    plot_order = [m for m in cfg.METHOD_PLOT_ORDER if m in methods_present]
    plot_order.extend(sorted(methods_present - set(plot_order)))
    data = [df.loc[df["method"] == method, "total_cost"].values for method in plot_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, tick_labels=plot_order, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("Total Cost (200 days)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def _plot_fixed_case_boxplots(df: pd.DataFrame, output_dir: Path):
    fixed_df = df[df["environment"] == "Stationary"].copy()
    if fixed_df.empty or "case_name" not in fixed_df.columns:
        return []

    case_names = sorted(fixed_df["case_name"].dropna().unique().tolist())
    saved = []
    for case_name in case_names:
        case_df = fixed_df[fixed_df["case_name"] == case_name].copy()
        if case_df.empty:
            continue
        safe_case = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(case_name))
        output_png = output_dir / f"boxplot_fixed_case_{safe_case}.png"
        _plot_boxplot(
            case_df,
            f"Fixed Scenario: {case_name} - Total Cost by Method",
            output_png,
        )
        saved.append(output_png)
    return saved


def _load_pretrained_agents() -> Dict[str, Dict]:
    loaded = {}
    for method in cfg.METHODS:
        if method["kind"] != "pretrained_rl":
            continue
        loaded[method["name"]] = {
            "agent": _load_pretrained_agent(
                checkpoint_path=method["checkpoint_path"],
                policy_dist=method["policy_dist"],
                device=cfg.DEVICE,
            ),
            "policy_dist": method["policy_dist"],
        }
    return loaded


def run_experiments(
    days: int,
    scenario_mode: str,
    evaluation_seeds: Sequence[int],
    stationary_num_scenarios_per_seed: int,
    nonstationary_num_sequences_per_seed: int,
    output_root: Path,
):
    if days > SIM_TIME:
        raise ValueError(
            f"DAYS={days} exceeds envs.config_SimPy.SIM_TIME={SIM_TIME}. "
            "Please set DAYS <= SIM_TIME."
        )

    segments = list(cfg.NONSTATIONARY_SEGMENTS)
    if not segments:
        raise ValueError("NONSTATIONARY_SEGMENTS is empty.")
    if segments[-1][1] != days:
        raise ValueError(
            f"NONSTATIONARY_SEGMENTS last day ({segments[-1][1]}) must equal DAYS ({days})."
        )

    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    loaded_agents = _load_pretrained_agents()
    rows = []

    scenario_mode = str(scenario_mode).strip().lower()
    if scenario_mode not in {"randomized", "fixed"}:
        raise ValueError(f"Unknown scenario_mode: {scenario_mode}. Use 'randomized' or 'fixed'.")

    for seed_idx, eval_seed in enumerate(evaluation_seeds, start=1):
        print(f"[Seed] {seed_idx}/{len(evaluation_seeds)} -> {eval_seed}")

        # ===== Stationary =====
        if scenario_mode == "fixed":
            stationary_scenarios = _generate_fixed_stationary_scenarios()
        else:
            stationary_scenarios = _generate_stationary_scenarios(
                stationary_num_scenarios_per_seed,
                seed=eval_seed,
            )
        stationary_segment = [(1, days)]
        for scenario_idx, scenario in enumerate(stationary_scenarios, start=1):
            schedule = [scenario]
            shared_seed = eval_seed + scenario_idx
            for method in cfg.METHODS:
                name = method["name"]
                if method["kind"] == "heuristic_rop":
                    result = _run_heuristic_episode(
                        reorder_point=int(method["reorder_point"]),
                        scenario_schedule=schedule,
                        segments=stationary_segment,
                        days=days,
                        seed=shared_seed,
                    )
                else:
                    model_info = loaded_agents[name]
                    result = _run_pretrained_episode(
                        agent=model_info["agent"],
                        policy_dist=model_info["policy_dist"],
                        scenario_schedule=schedule,
                        segments=stationary_segment,
                        days=days,
                        seed=shared_seed,
                        method_name=name,
                    )
                rows.append(
                    {
                        "seed": int(eval_seed),
                        "environment": "Stationary",
                        "case_id": scenario_idx,
                        "case_name": str(scenario.get("_case_name", f"scenario_{scenario_idx}")),
                        "method": name,
                        "total_cost": result.total_cost,
                        "holding_cost": result.holding_cost,
                        "process_cost": result.process_cost,
                        "delivery_cost": result.delivery_cost,
                        "order_cost": result.order_cost,
                        "shortage_cost": result.shortage_cost,
                    }
                )

        # ===== Nonstationary =====
        if scenario_mode == "fixed":
            nonstationary_sequences = _generate_fixed_nonstationary_sequences(
                nonstationary_num_sequences_per_seed,
                seed=eval_seed + 10000,
            )
        else:
            nonstationary_sequences = _generate_nonstationary_sequences(
                nonstationary_num_sequences_per_seed,
                seed=eval_seed + 10000,
            )
        for case_idx, sequence in enumerate(nonstationary_sequences, start=1):
            shared_seed = eval_seed + 50000 + case_idx
            for method in cfg.METHODS:
                name = method["name"]
                if method["kind"] == "heuristic_rop":
                    result = _run_heuristic_episode(
                        reorder_point=int(method["reorder_point"]),
                        scenario_schedule=sequence,
                        segments=segments,
                        days=days,
                        seed=shared_seed,
                    )
                else:
                    model_info = loaded_agents[name]
                    result = _run_pretrained_episode(
                        agent=model_info["agent"],
                        policy_dist=model_info["policy_dist"],
                        scenario_schedule=sequence,
                        segments=segments,
                        days=days,
                        seed=shared_seed,
                        method_name=name,
                    )
                rows.append(
                    {
                        "seed": int(eval_seed),
                        "environment": "Nonstationary",
                        "case_id": case_idx,
                        "case_name": " -> ".join(
                            str(s.get("_case_name", f"scenario_{j+1}"))
                            for j, s in enumerate(sequence)
                        ),
                        "method": name,
                        "total_cost": result.total_cost,
                        "holding_cost": result.holding_cost,
                        "process_cost": result.process_cost,
                        "delivery_cost": result.delivery_cost,
                        "order_cost": result.order_cost,
                        "shortage_cost": result.shortage_cost,
                    }
                )

    df = pd.DataFrame(rows)
    all_csv = run_dir / "policy_comparison_results.csv"
    df.to_csv(all_csv, index=False)

    stationary_df = df[df["environment"] == "Stationary"].copy()
    nonstationary_df = df[df["environment"] == "Nonstationary"].copy()
    stationary_csv = run_dir / "stationary_results.csv"
    nonstationary_csv = run_dir / "nonstationary_results.csv"
    stationary_df.to_csv(stationary_csv, index=False)
    nonstationary_df.to_csv(nonstationary_csv, index=False)

    stationary_plot = run_dir / "boxplot_stationary_total_cost.png"
    nonstationary_plot = run_dir / "boxplot_nonstationary_total_cost.png"
    _plot_boxplot(stationary_df, "Stationary (200 days): Total Cost by Method", stationary_plot)
    _plot_boxplot(nonstationary_df, "Nonstationary (200 days): Total Cost by Method", nonstationary_plot)
    if scenario_mode == "fixed":
        case_plots = _plot_fixed_case_boxplots(df, run_dir)
        for p in case_plots:
            print(f"[Saved] {p}")
    if bool(getattr(cfg, "SAVE_SINGLE_INTEGRATED_BOXPLOT", True)):
        integrated_plot = run_dir / "boxplot_total_cost_all_methods.png"
        _plot_boxplot(
            df,
            "All Seeds/Cases Combined: Total Cost by Method",
            integrated_plot,
        )
        print(f"[Saved] {integrated_plot}")

    print(f"[Saved] {all_csv}")
    print(f"[Saved] {stationary_csv}")
    print(f"[Saved] {nonstationary_csv}")
    print(f"[Saved] {stationary_plot}")
    print(f"[Saved] {nonstationary_plot}")

    summary = (
        df.groupby(["environment", "method"])["total_cost"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary_csv = run_dir / "summary_by_environment_method.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")
    print("[Summary]")
    print(summary.to_string(index=False))

    summary_overall = (
        df.groupby(["method"])["total_cost"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary_overall_csv = run_dir / "summary_overall_by_method.csv"
    summary_overall.to_csv(summary_overall_csv, index=False)
    print(f"[Saved] {summary_overall_csv}")
    print("[Summary Overall]")
    print(summary_overall.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare R1/R3/R5 heuristics vs pretrained PPO/ProMP on stationary and nonstationary scenarios."
    )
    parser.add_argument("--days", type=int, default=cfg.DAYS)
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default=cfg.SCENARIO_MODE,
        choices=["randomized", "fixed"],
        help="fixed: use Few_shot_learning fixed 5 scenarios, randomized: sample new scenarios",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(cfg.EVALUATION_SEEDS),
        help="Evaluation seeds. Each seed generates its own scenario pool.",
    )
    parser.add_argument(
        "--stationary-num-scenarios-per-seed",
        type=int,
        default=cfg.STATIONARY_NUM_SCENARIOS_PER_SEED,
    )
    parser.add_argument(
        "--nonstationary-num-sequences-per-seed",
        type=int,
        default=cfg.NONSTATIONARY_NUM_SEQUENCES_PER_SEED,
    )
    parser.add_argument("--output-root", type=Path, default=cfg.OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments(
        days=args.days,
        scenario_mode=args.scenario_mode,
        evaluation_seeds=args.seeds,
        stationary_num_scenarios_per_seed=args.stationary_num_scenarios_per_seed,
        nonstationary_num_sequences_per_seed=args.nonstationary_num_sequences_per_seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
