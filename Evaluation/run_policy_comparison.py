import argparse
import copy
import gc
import hashlib
import importlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from Evaluation import config as cfg
from AIIS_META.Agents.Categorical.Meta_Categorical import MetaCategoricalAgent
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
from envs.config_RL import ACTION_SPACE
import envs.scenarios as scenario_factory


_TRANSIENT_ENV_ERROR_KEYWORDS = (
    "'bool' object has no attribute 'on_hand_inventory'",
    "'list' object is not callable",
    "unsupported operand type(s) for -=: 'Inventory' and 'float'",
    "'int' object is not subscriptable",
    "'int' object is not callable",
    "object is not callable",
    "'list' object has no attribute 'in_transition_inventory'",
    "'Production' object has no attribute 'capacity_limit'",
)

_COST_KEYS = [
    "Holding cost",
    "Process cost",
    "Delivery cost",
    "Order cost",
    "Shortage cost",
]


@dataclass
class EpisodeMetrics:
    total_cost: float
    holding_cost: float
    process_cost: float
    delivery_cost: float
    order_cost: float
    shortage_cost: float


def _stable_seed(*parts: object) -> int:
    raw = "|".join([str(cfg.RANDOM_SEED)] + [str(part) for part in parts])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fresh_env_modules():
    import envs.environment as env_module
    import envs.log_SimPy as log_module
    import envs.promp_env as promp_env_module
    import envs.simpy_diagnostics as diag_module

    log_module = importlib.reload(log_module)
    env_module = importlib.reload(env_module)
    promp_env_module = importlib.reload(promp_env_module)
    diag_module = importlib.reload(diag_module)
    return env_module, log_module, promp_env_module, diag_module


def _reset_logs(log_module) -> None:
    log_module.DAILY_EVENTS.clear()
    log_module.DAILY_REPORTS.clear()
    log_module.STATE_DICT.clear()
    for key in list(log_module.DAILY_COST_REPORT.keys()):
        log_module.DAILY_COST_REPORT[key] = 0


def _metrics_from_cost_dict(cost_dict: Dict[str, float]) -> EpisodeMetrics:
    return EpisodeMetrics(
        total_cost=float(sum(cost_dict.get(key, 0.0) for key in _COST_KEYS)),
        holding_cost=float(cost_dict.get("Holding cost", 0.0)),
        process_cost=float(cost_dict.get("Process cost", 0.0)),
        delivery_cost=float(cost_dict.get("Delivery cost", 0.0)),
        order_cost=float(cost_dict.get("Order cost", 0.0)),
        shortage_cost=float(cost_dict.get("Shortage cost", 0.0)),
    )


def _deepcopy_scenario(scenario: Dict) -> Dict:
    return copy.deepcopy(scenario)


def _case_to_scenario(case: Dict, mat_count: int) -> Dict:
    scenario = {"DEMAND": copy.deepcopy(case["demand_dist"])}
    if "leadtime_by_material" in case:
        leadtime = copy.deepcopy(case["leadtime_by_material"])
    else:
        leadtime = [copy.deepcopy(case["leadtime_dist"]) for _ in range(mat_count)]
    scenario["LEADTIME"] = leadtime
    return scenario


def _apply_scenario_inplace(target: Dict, source: Dict) -> None:
    if "DEMAND" not in target or not isinstance(target["DEMAND"], dict):
        target["DEMAND"] = {}
    target["DEMAND"].clear()
    target["DEMAND"].update(copy.deepcopy(source["DEMAND"]))

    source_leadtime = copy.deepcopy(source["LEADTIME"])
    if "LEADTIME" not in target or not isinstance(target["LEADTIME"], list):
        target["LEADTIME"] = source_leadtime
        return

    target_leadtime = target["LEADTIME"]
    while len(target_leadtime) < len(source_leadtime):
        target_leadtime.append({})
    while len(target_leadtime) > len(source_leadtime):
        target_leadtime.pop()

    for index, source_item in enumerate(source_leadtime):
        if not isinstance(target_leadtime[index], dict):
            target_leadtime[index] = source_item
            continue
        target_leadtime[index].clear()
        target_leadtime[index].update(source_item)


def _run_with_retry(fn, description: str):
    last_error = None
    for attempt in range(1, int(cfg.MAX_EPISODE_RETRIES) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= int(cfg.MAX_EPISODE_RETRIES):
                raise
            print(f"[Retry {attempt}/{cfg.MAX_EPISODE_RETRIES}] {description}: {exc}")
            gc.collect()
    raise last_error

def _extract_policy_state_dict(checkpoint) -> Dict:
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for candidate_key in ("state_dict", "agent_state_dict", "model_state_dict"):
            candidate_value = checkpoint.get(candidate_key)
            if isinstance(candidate_value, dict):
                state_dict = candidate_value
                break

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict).__name__}")

    if any(key.startswith("agent.") for key in state_dict.keys()):
        state_dict = {
            key.removeprefix("agent."): value
            for key, value in state_dict.items()
            if key.startswith("agent.")
        }

    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("inner_step_sizes.")
    }
    return state_dict


def _build_policy_agent(model_spec: Dict):
    _, _, promp_env_module, _ = _fresh_env_modules()
    probe_env = promp_env_module.MetaEnv()
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    action_dim = int(np.prod(probe_env.action_space.shape))

    if model_spec["policy_dist"] == "categorical":
        mlp = SimpleMLP(obs_dim, action_dim * len(ACTION_SPACE), list(cfg.AGENT_HIDDEN_LAYERS))
        agent = MetaCategoricalAgent(
            num_tasks=1,
            mlp=mlp,
            action_dim=action_dim,
            num_actions=len(ACTION_SPACE),
        )
    elif model_spec["policy_dist"] == "gaussian":
        mlp = SimpleMLP(obs_dim, action_dim, list(cfg.AGENT_HIDDEN_LAYERS))
        agent = MetaGaussianAgent(
            num_tasks=1,
            mlp=mlp,
            learn_std=cfg.LEARN_STD,
        )
    else:
        raise ValueError(f"Unsupported policy_dist: {model_spec['policy_dist']}")

    checkpoint_path = Path(model_spec["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    state_dict = _extract_policy_state_dict(checkpoint)
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint load mismatch for {model_spec['name']}: missing={missing}, unexpected={unexpected}"
        )

    agent.to(cfg.DEVICE)
    agent.eval()
    return agent


def _deterministic_action(agent, policy_dist: str, state) -> np.ndarray:
    obs = torch.as_tensor(state, dtype=torch.float32, device=cfg.DEVICE).view(1, -1)
    params = dict(agent.named_parameters())
    dist = agent.distribution(obs, params=params)
    if policy_dist == "categorical":
        action = torch.argmax(dist.logits, dim=-1).squeeze(0)
    else:
        action = dist.mean.squeeze(0)
    return action.detach().cpu().numpy()


def _evaluate_pretrained_stationary(agent, policy_dist: str, scenario: Dict, episode_seed: int) -> EpisodeMetrics:
    def _impl():
        _seed_everything(episode_seed)
        _, log_module, promp_env_module, _ = _fresh_env_modules()
        _reset_logs(log_module)

        env = promp_env_module.MetaEnv()
        env.set_task(_deepcopy_scenario(scenario))
        state = env.reset()

        done = False
        last_cost_dict = None
        while not done:
            action = _deterministic_action(agent, policy_dist, state)
            state, _, done, cost_dict = env.step(action)
            last_cost_dict = dict(cost_dict)

        return _metrics_from_cost_dict(last_cost_dict or {})

    return _run_with_retry(_impl, "pretrained stationary episode")


def _evaluate_pretrained_nonstationary(
    agent,
    policy_dist: str,
    segment_scenarios: Sequence[Dict],
    episode_seed: int,
) -> EpisodeMetrics:
    def _impl():
        _seed_everything(episode_seed)
        _, log_module, promp_env_module, _ = _fresh_env_modules()
        _reset_logs(log_module)

        active_scenario = _deepcopy_scenario(segment_scenarios[0])
        env = promp_env_module.MetaEnv()
        env.set_task(active_scenario)
        state = env.reset()

        segment_index = 0
        day = 1
        done = False
        last_cost_dict = None

        while not done:
            next_segment_index = segment_index + 1
            if next_segment_index < len(cfg.NONSTATIONARY_SEGMENTS):
                next_segment_start = cfg.NONSTATIONARY_SEGMENTS[next_segment_index][0]
                if day == next_segment_start:
                    _apply_scenario_inplace(active_scenario, segment_scenarios[next_segment_index])
                    segment_index = next_segment_index

            action = _deterministic_action(agent, policy_dist, state)
            state, _, done, cost_dict = env.step(action)
            last_cost_dict = dict(cost_dict)
            day += 1

        return _metrics_from_cost_dict(last_cost_dict or {})

    return _run_with_retry(_impl, "pretrained nonstationary episode")


def _evaluate_heuristic_stationary(scenario: Dict, reorder_point: int, episode_seed: int) -> EpisodeMetrics:
    def _impl():
        _seed_everything(episode_seed)
        env_module, log_module, _, diag_module = _fresh_env_modules()
        _reset_logs(log_module)

        (
            simpy_env,
            inventory_list,
            procurement_list,
            production_list,
            sales,
            customer,
            supplier_list,
            daily_events,
        ) = env_module.create_env(env_module.I, env_module.P, log_module.DAILY_EVENTS)

        ordered_qty_daily = {supplier.item_id: 0 for supplier in supplier_list}
        diag_module._start_heuristic_processes(
            env_module=env_module,
            simpy_env=simpy_env,
            inventory_list=inventory_list,
            procurement_list=procurement_list,
            production_list=production_list,
            sales=sales,
            customer=customer,
            supplier_list=supplier_list,
            daily_events=daily_events,
            scenario=_deepcopy_scenario(scenario),
            reorder_point=reorder_point,
            ordered_qty_daily=ordered_qty_daily,
        )

        simpy_env.run(until=cfg.DAYS * 24)
        env_module.Cost.update_cost_log(inventory_list)
        return _metrics_from_cost_dict(dict(log_module.DAILY_COST_REPORT))

    return _run_with_retry(_impl, f"heuristic stationary episode R{reorder_point}")


def _evaluate_heuristic_nonstationary(
    segment_scenarios: Sequence[Dict],
    reorder_point: int,
    episode_seed: int,
) -> EpisodeMetrics:
    def _impl():
        _seed_everything(episode_seed)
        env_module, log_module, _, diag_module = _fresh_env_modules()
        _reset_logs(log_module)

        (
            simpy_env,
            inventory_list,
            procurement_list,
            production_list,
            sales,
            customer,
            supplier_list,
            daily_events,
        ) = env_module.create_env(env_module.I, env_module.P, log_module.DAILY_EVENTS)

        ordered_qty_daily = {supplier.item_id: 0 for supplier in supplier_list}
        active_scenario = _deepcopy_scenario(segment_scenarios[0])
        diag_module._start_heuristic_processes(
            env_module=env_module,
            simpy_env=simpy_env,
            inventory_list=inventory_list,
            procurement_list=procurement_list,
            production_list=production_list,
            sales=sales,
            customer=customer,
            supplier_list=supplier_list,
            daily_events=daily_events,
            scenario=active_scenario,
            reorder_point=reorder_point,
            ordered_qty_daily=ordered_qty_daily,
        )

        for segment_bounds, segment_scenario in zip(cfg.NONSTATIONARY_SEGMENTS, segment_scenarios):
            _apply_scenario_inplace(active_scenario, segment_scenario)
            simpy_env.run(until=segment_bounds[1] * 24)

        env_module.Cost.update_cost_log(inventory_list)
        return _metrics_from_cost_dict(dict(log_module.DAILY_COST_REPORT))

    return _run_with_retry(_impl, f"heuristic nonstationary episode R{reorder_point}")


def _generate_stationary_scenarios() -> List[Dict]:
    import envs.config_SimPy as simpy_cfg

    if cfg.SCENARIO_MODE == "fixed":
        rows = []
        for index, case in enumerate(cfg.FIXED_SCENARIO_CASES, start=1):
            rows.append(
                {
                    "scenario_id": index,
                    "scenario_label": case["case"],
                    "scenario": _case_to_scenario(case, simpy_cfg.MAT_COUNT),
                    "episodes_per_scenario": int(cfg.FIXED_EPISODES_PER_SCENARIO),
                }
            )
        return rows

    sampling_kwargs = dict(cfg.RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES)
    sampling_kwargs["num_scenarios"] = int(cfg.RANDOMIZED_STATIONARY_SCENARIO_COUNT)
    sampling_kwargs["seed"] = _stable_seed("randomized_stationary_pool")
    sampled = scenario_factory.create_scenarios(**sampling_kwargs)
    return [
        {
            "scenario_id": index,
            "scenario_label": f"Randomized_Stationary_{index:03d}",
            "scenario": scenario,
            "episodes_per_scenario": int(cfg.RANDOMIZED_STATIONARY_EPISODES_PER_SCENARIO),
        }
        for index, scenario in enumerate(sampled, start=1)
    ]


def _generate_nonstationary_sequences() -> List[Dict]:
    if cfg.SCENARIO_MODE != "randomized":
        return []

    segment_count = len(cfg.NONSTATIONARY_SEGMENTS)
    sampling_kwargs = dict(cfg.RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES)
    sampling_kwargs["num_scenarios"] = int(cfg.RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT) * segment_count
    sampling_kwargs["seed"] = _stable_seed("randomized_nonstationary_pool")
    sampled = scenario_factory.create_scenarios(**sampling_kwargs)

    rows = []
    for index in range(int(cfg.RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT)):
        start = index * segment_count
        end = start + segment_count
        rows.append(
            {
                "sequence_id": index + 1,
                "scenario_label": f"Randomized_Nonstationary_{index + 1:03d}",
                "segment_scenarios": sampled[start:end],
                "episodes_per_sequence": int(cfg.RANDOMIZED_NONSTATIONARY_EPISODES_PER_SEQUENCE),
            }
        )
    return rows


def _build_policy_specs() -> List[Dict]:
    specs = []
    for reorder_point in cfg.HEURISTIC_REORDER_POINTS:
        specs.append({"name": f"R{reorder_point}", "type": "heuristic", "reorder_point": int(reorder_point)})
    for model_spec in cfg.PRETRAINED_MODELS:
        specs.append(
            {
                "name": model_spec["name"],
                "type": "pretrained",
                "policy_dist": model_spec["policy_dist"],
                "agent": _build_policy_agent(model_spec),
            }
        )
    return specs


def _evaluate_stationary(policy_specs: Sequence[Dict]) -> List[Dict]:
    rows = []
    for scenario_info in _generate_stationary_scenarios():
        print(f"[Stationary] {scenario_info['scenario_label']}")
        for episode_index in range(1, int(scenario_info["episodes_per_scenario"]) + 1):
            episode_seed = _stable_seed("stationary", scenario_info["scenario_label"], episode_index)
            for policy in policy_specs:
                if policy["type"] == "heuristic":
                    metrics = _evaluate_heuristic_stationary(
                        scenario=scenario_info["scenario"],
                        reorder_point=policy["reorder_point"],
                        episode_seed=episode_seed,
                    )
                else:
                    metrics = _evaluate_pretrained_stationary(
                        agent=policy["agent"],
                        policy_dist=policy["policy_dist"],
                        scenario=scenario_info["scenario"],
                        episode_seed=episode_seed,
                    )

                row = {
                    "scenario_mode": cfg.SCENARIO_MODE,
                    "environment_mode": "stationary",
                    "scenario_label": scenario_info["scenario_label"],
                    "episode_index": episode_index,
                    "method": policy["name"],
                    "seed": episode_seed,
                }
                row.update(asdict(metrics))
                rows.append(row)
    return rows


def _evaluate_nonstationary(policy_specs: Sequence[Dict]) -> List[Dict]:
    if cfg.SCENARIO_MODE != "randomized":
        return []

    rows = []
    for sequence_info in _generate_nonstationary_sequences():
        print(f"[Nonstationary] {sequence_info['scenario_label']}")
        for episode_index in range(1, int(sequence_info["episodes_per_sequence"]) + 1):
            episode_seed = _stable_seed("nonstationary", sequence_info["scenario_label"], episode_index)
            for policy in policy_specs:
                if policy["type"] == "heuristic":
                    metrics = _evaluate_heuristic_nonstationary(
                        segment_scenarios=sequence_info["segment_scenarios"],
                        reorder_point=policy["reorder_point"],
                        episode_seed=episode_seed,
                    )
                else:
                    metrics = _evaluate_pretrained_nonstationary(
                        agent=policy["agent"],
                        policy_dist=policy["policy_dist"],
                        segment_scenarios=sequence_info["segment_scenarios"],
                        episode_seed=episode_seed,
                    )

                row = {
                    "scenario_mode": cfg.SCENARIO_MODE,
                    "environment_mode": "nonstationary",
                    "scenario_label": sequence_info["scenario_label"],
                    "episode_index": episode_index,
                    "method": policy["name"],
                    "seed": episode_seed,
                }
                row.update(asdict(metrics))
                rows.append(row)
    return rows


def _save_config_snapshot(output_dir: Path) -> None:
    shutil.copy2(Path(__file__).with_name("config.py"), output_dir / "config_snapshot.py")


def _save_run_metadata(output_dir: Path, environment_modes: Sequence[str]) -> None:
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario_mode": cfg.SCENARIO_MODE,
        "environment_modes": list(environment_modes),
        "random_seed": int(cfg.RANDOM_SEED),
        "days": int(cfg.DAYS),
        "fixed_episodes_per_scenario": int(cfg.FIXED_EPISODES_PER_SCENARIO),
        "randomized_stationary_scenario_count": int(cfg.RANDOMIZED_STATIONARY_SCENARIO_COUNT),
        "randomized_stationary_episodes_per_scenario": int(cfg.RANDOMIZED_STATIONARY_EPISODES_PER_SCENARIO),
        "randomized_nonstationary_sequence_count": int(cfg.RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT),
        "randomized_nonstationary_episodes_per_sequence": int(cfg.RANDOMIZED_NONSTATIONARY_EPISODES_PER_SEQUENCE),
        "nonstationary_segments": list(cfg.NONSTATIONARY_SEGMENTS),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _plot_placeholder(output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.4, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_total_cost_boxplot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        _plot_placeholder(output_path, title, "No evaluation rows were generated.")
        return

    order = [method for method in cfg.PLOT_METHOD_ORDER if method in set(df["method"])]
    if not order:
        _plot_placeholder(output_path, title, "No methods available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    data = [df.loc[df["method"] == method, "total_cost"].to_numpy() for method in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("Total Cost")
    ax.set_ylim(0, 55000)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def _plot_fixed_stationary_by_case(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        _plot_placeholder(output_path, "Fixed Stationary Boxplots by Case", "No stationary rows were generated.")
        return

    labels = list(df["scenario_label"].drop_duplicates())
    order = [method for method in cfg.PLOT_METHOD_ORDER if method in set(df["method"])]
    if not labels or not order:
        _plot_placeholder(output_path, "Fixed Stationary Boxplots by Case", "No fixed-case data was available.")
        return

    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")
    flat_axes = axes.flatten()

    for axis, label in zip(flat_axes, labels):
        axis.set_facecolor("white")
        subset = df[df["scenario_label"] == label]
        data = [subset.loc[subset["method"] == method, "total_cost"].to_numpy() for method in order]
        axis.boxplot(data, tick_labels=order, showmeans=True)
        axis.set_title(label)
        axis.set_ylabel("Total Cost")
        axis.grid(alpha=0.3)

    for axis in flat_axes[len(labels):]:
        axis.axis("off")

    fig.suptitle("Fixed Stationary Total Cost by Case", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_promp_significance_heatmap(
    significance_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    if significance_df.empty:
        _plot_placeholder(output_path, title, "No significance results were generated.")
        return

    required_cols = {"environment_mode", "baseline_method", "p_value"}
    if not required_cols.issubset(significance_df.columns):
        _plot_placeholder(output_path, title, "Required significance columns are missing.")
        return

    env_order = [mode for mode in ("stationary", "nonstationary") if mode in set(significance_df["environment_mode"])]
    method_order = [
        method
        for method in cfg.PLOT_METHOD_ORDER
        if method != "ProMP" and method in set(significance_df["baseline_method"])
    ]
    if not env_order or not method_order:
        _plot_placeholder(output_path, title, "No heatmap rows were available.")
        return

    heatmap_df = (
        significance_df.pivot_table(
            index="baseline_method",
            columns="environment_mode",
            values="p_value",
            aggfunc="first",
        )
        .reindex(index=method_order, columns=env_order)
    )

    fig, ax = plt.subplots(figsize=(1.8 + 2.4 * len(env_order), 2.2 + 0.8 * len(method_order)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    matrix = heatmap_df.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="viridis_r", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(env_order)))
    ax.set_xticklabels(env_order)
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order)
    ax.set_xlabel("Environment")
    ax.set_ylabel("Method Compared Against ProMP")
    ax.set_title(title)

    for row_idx, method in enumerate(method_order):
        for col_idx, env_name in enumerate(env_order):
            value = heatmap_df.loc[method, env_name]
            label = "NA" if not np.isfinite(value) else f"{value:.3f}"
            text_color = "white" if np.isfinite(value) and value < 0.5 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("p-value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_cost_composition(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        _plot_placeholder(output_path, title, "No evaluation rows were generated.")
        return

    order = [method for method in cfg.PLOT_METHOD_ORDER if method in set(df["method"])]
    if not order:
        _plot_placeholder(output_path, title, "No methods available for plotting.")
        return

    cost_cols = [
        "holding_cost",
        "process_cost",
        "delivery_cost",
        "order_cost",
        "shortage_cost",
    ]
    pretty_labels = {
        "holding_cost": "Holding",
        "process_cost": "Process",
        "delivery_cost": "Delivery",
        "order_cost": "Order",
        "shortage_cost": "Shortage",
    }

    grouped = df.groupby("method", as_index=False)[cost_cols].mean()
    grouped = grouped.set_index("method").reindex(order)
    row_sums = grouped.sum(axis=1).replace(0.0, np.nan)
    ratio_df = grouped.div(row_sums, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

    for idx, col in enumerate(cost_cols):
        values = ratio_df[col].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.7,
            label=pretty_labels[col],
            color=colors[idx],
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cost Ratio")
    ax.set_xlabel("Method")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _paired_wilcoxon_promp_vs_method(
    df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    if df.empty or "ProMP" not in set(df["method"]):
        return pd.DataFrame()

    pair_index_cols = list(group_cols)
    for col in ("scenario_label", "episode_index"):
        if col not in pair_index_cols:
            pair_index_cols.append(col)

    pair_cols = pair_index_cols + ["method", "total_cost"]
    base = df[pair_cols].copy()
    pivot = (
        base.pivot_table(
            index=pair_index_cols,
            columns="method",
            values="total_cost",
            aggfunc="mean",
        )
        .reset_index()
    )

    if "ProMP" not in pivot.columns:
        return pd.DataFrame()

    methods = [m for m in cfg.PLOT_METHOD_ORDER if m != "ProMP" and m in pivot.columns]
    rows = []
    for method in methods:
        compare_cols = list(group_cols) + ["ProMP", method]
        subset = pivot[compare_cols].dropna().copy()
        if subset.empty:
            continue

        diffs = subset[method] - subset["ProMP"]
        nonzero_diffs = diffs[np.abs(diffs) > 1e-12]

        p_value = np.nan
        stat = np.nan
        if len(nonzero_diffs) > 0:
            try:
                res = wilcoxon(nonzero_diffs, zero_method="wilcox", alternative="greater")
                stat = float(res.statistic)
                p_value = float(res.pvalue)
            except ValueError:
                pass

        for group_key, group_frame in subset.groupby(list(group_cols), dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            local_diffs = group_frame[method] - group_frame["ProMP"]
            local_nonzero = local_diffs[np.abs(local_diffs) > 1e-12]
            local_stat = np.nan
            local_p = np.nan
            if len(local_nonzero) > 0:
                try:
                    local_res = wilcoxon(local_nonzero, zero_method="wilcox", alternative="greater")
                    local_stat = float(local_res.statistic)
                    local_p = float(local_res.pvalue)
                except ValueError:
                    pass

            row = {col: value for col, value in zip(group_cols, group_key)}
            row.update(
                {
                    "comparison": f"ProMP vs {method}",
                    "baseline_method": method,
                    "n_pairs": int(len(group_frame)),
                    "promp_mean": float(group_frame["ProMP"].mean()),
                    "other_mean": float(group_frame[method].mean()),
                    "mean_diff_other_minus_promp": float(local_diffs.mean()),
                    "median_diff_other_minus_promp": float(local_diffs.median()),
                    "promp_better_rate": float((local_diffs > 0).mean()),
                    "wilcoxon_statistic": local_stat,
                    "p_value": local_p,
                    "significant_0_05": bool(np.isfinite(local_p) and local_p < 0.05),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)

def _save_outputs(df: pd.DataFrame, output_dir: Path, environment_modes: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(output_dir)
    _save_run_metadata(output_dir, environment_modes)

    df.to_csv(output_dir / "episode_results.csv", index=False)

    if df.empty:
        summary_by_environment_method = pd.DataFrame()
        summary_by_scenario_method = pd.DataFrame()
    else:
        summary_by_environment_method = (
            df.groupby(["environment_mode", "method"], as_index=False)["total_cost"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )
        summary_by_scenario_method = (
            df.groupby(["environment_mode", "scenario_label", "method"], as_index=False)["total_cost"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )

    summary_by_environment_method.to_csv(output_dir / "summary_by_environment_method.csv", index=False)
    summary_by_scenario_method.to_csv(output_dir / "summary_by_scenario_method.csv", index=False)

    significance_by_environment = _paired_wilcoxon_promp_vs_method(
        df,
        group_cols=["environment_mode"],
    )
    significance_by_scenario = _paired_wilcoxon_promp_vs_method(
        df,
        group_cols=["environment_mode", "scenario_label"],
    )
    significance_by_environment.to_csv(
        output_dir / "significance_promp_vs_others_by_environment.csv",
        index=False,
    )
    significance_by_scenario.to_csv(
        output_dir / "significance_promp_vs_others_by_scenario.csv",
        index=False,
    )
    _plot_promp_significance_heatmap(
        significance_by_environment,
        output_dir / "significance_promp_vs_others_heatmap.png",
        "ProMP vs Others: Paired Wilcoxon p-values",
    )

    stationary_df = df[df["environment_mode"] == "stationary"] if not df.empty else df
    nonstationary_df = df[df["environment_mode"] == "nonstationary"] if not df.empty else df

    _plot_total_cost_boxplot(
        stationary_df,
        output_dir / "stationary_boxplot_total_cost.png",
        "Stationary Total Cost Comparison",
    )

    if cfg.SCENARIO_MODE == "fixed":
        _plot_fixed_stationary_by_case(
            stationary_df,
            output_dir / "stationary_boxplot_by_case.png",
        )

    if nonstationary_df.empty:
        message = "Nonstationary evaluation was not executed."
        if cfg.SCENARIO_MODE == "fixed":
            message = "Fixed mode intentionally skips nonstationary evaluation."
        _plot_placeholder(
            output_dir / "nonstationary_boxplot_total_cost.png",
            "Nonstationary Total Cost Comparison",
            message,
        )
    else:
        _plot_total_cost_boxplot(
            nonstationary_df,
            output_dir / "nonstationary_boxplot_total_cost.png",
            "Nonstationary Total Cost Comparison",
        )

    _plot_cost_composition(
        stationary_df,
        output_dir / "stationary_cost_composition.png",
        "Stationary Cost Composition by Method",
    )

    if nonstationary_df.empty:
        _plot_placeholder(
            output_dir / "nonstationary_cost_composition.png",
            "Nonstationary Cost Composition by Method",
            "Nonstationary evaluation was not executed.",
        )
    else:
        _plot_cost_composition(
            nonstationary_df,
            output_dir / "nonstationary_cost_composition.png",
            "Nonstationary Cost Composition by Method",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare heuristic, PPO, and ProMP total cost.")
    parser.add_argument(
        "--scenario-mode",
        choices=["fixed", "randomized"],
        default=cfg.SCENARIO_MODE,
        help="Override config.SCENARIO_MODE for this run.",
    )
    parser.add_argument(
        "--environment-modes",
        nargs="*",
        choices=["stationary", "nonstationary"],
        default=None,
        help="Optional override for config.ENVIRONMENT_MODES.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg.SCENARIO_MODE = args.scenario_mode

    environment_modes = list(args.environment_modes) if args.environment_modes else list(cfg.ENVIRONMENT_MODES)
    if cfg.SCENARIO_MODE == "fixed":
        environment_modes = ["stationary"]

    plt.style.use("default")

    policy_specs = _build_policy_specs()
    all_rows: List[Dict] = []

    if "stationary" in environment_modes:
        all_rows.extend(_evaluate_stationary(policy_specs))
    if "nonstationary" in environment_modes and cfg.SCENARIO_MODE == "randomized":
        all_rows.extend(_evaluate_nonstationary(policy_specs))

    df = pd.DataFrame(all_rows)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(cfg.OUTPUT_ROOT) / timestamp
    _save_outputs(df, output_dir, environment_modes)

    print(f"[Saved] Output directory: {output_dir}")
    if not df.empty:
        summary = (
            df.groupby(["environment_mode", "method"], as_index=False)["total_cost"]
            .mean()
            .sort_values(["environment_mode", "total_cost"])
        )
        print("[Summary] Mean total cost by environment and method")
        print(summary.to_string(index=False))
    else:
        print("[Summary] No evaluation rows were produced.")


if __name__ == "__main__":
    main()






