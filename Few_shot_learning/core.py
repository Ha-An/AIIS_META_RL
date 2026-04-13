import copy
import csv
import hashlib
import json
import math
import random
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from envs.config_RL import ACTION_SPACE
from envs.promp_env import MetaEnv
from envs.scenarios import create_scenarios

from AIIS_META.Agents.Categorical.Meta_Categorical import MetaCategoricalAgent
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Sampler.meta_sample_processor import MetaSampleProcessor
from AIIS_META.Sampler.meta_sampler import MetaSampler

import Few_shot_learning.config as cfg


RAW_FIELDS = [
    "timestamp", "status", "scenario_mode", "environment_mode", "case_name", "task_label", "model", "shot_k",
    "mean_total_cost", "num_query_trajectories", "adapt_updates",
    "adapt_param_delta_l2_mean", "adapt_param_delta_l2_std",
    "support_seed", "query_seed", "run_seconds", "error",
]

SUMMARY_FIELDS = [
    "scenario_mode", "environment_mode", "model", "shot_k", "n", "mean_total_cost",
    "std_total_cost", "sem_total_cost", "mean_adapt_param_delta_l2",
    "std_adapt_param_delta_l2", "sem_adapt_param_delta_l2",
]

TASK_SUMMARY_FIELDS = [
    "scenario_mode", "environment_mode", "case_name", "task_label", "model", "shot_k", "n", "mean_total_cost",
    "std_total_cost", "sem_total_cost",
]

CASE_SUMMARY_FIELDS = [
    "scenario_mode", "environment_mode", "case_name", "model", "shot_k", "n",
    "mean_total_cost", "std_total_cost", "sem_total_cost",
]

PROGRESS_FIELDS = [
    "timestamp", "completed_runs", "total_runs", "progress_pct", "environment_mode",
    "case_name", "task_label", "model", "shot_k", "status", "run_seconds", "elapsed_seconds",
    "eta_seconds", "eta_finish_local",
]


@dataclass
class EvalSettings:
    scenario_mode: str
    shots: List[int]
    environment_modes: List[str]
    stationary_scenario_count: int
    nonstationary_sequence_count: int
    query_rollout_per_task: int
    adapt_updates: int
    num_tasks: int
    envs_per_task: int
    parallel: bool
    max_path_length: int
    fail_fast: bool


def default_settings() -> EvalSettings:
    return EvalSettings(
        scenario_mode=str(cfg.SCENARIO_MODE),
        shots=list(cfg.EVAL_SHOTS),
        environment_modes=list(cfg.ENVIRONMENT_MODES),
        stationary_scenario_count=int(cfg.RANDOMIZED_STATIONARY_SCENARIO_COUNT),
        nonstationary_sequence_count=int(cfg.RANDOMIZED_NONSTATIONARY_SEQUENCE_COUNT),
        query_rollout_per_task=int(cfg.QUERY_ROLLOUT_PER_TASK),
        adapt_updates=int(cfg.EVAL_ADAPT_UPDATES),
        num_tasks=int(cfg.NUM_TASKS),
        envs_per_task=int(cfg.ENVS_PER_TASK),
        parallel=bool(cfg.PARALLEL),
        max_path_length=int(cfg.DAYS),
        fail_fast=bool(cfg.FAIL_FAST),
    )


class RandomizedFewShotEnv(MetaEnv):
    def __init__(self, task_pool: Sequence[Dict[str, Any]], segments: Sequence[Tuple[int, int]]):
        super().__init__()
        self._fewshot_task_pool = copy.deepcopy(list(task_pool))
        self._segments = list(segments)
        self.task = None
        self._task_mode = None
        self._segment_scenarios: List[Dict[str, Any]] = []
        self._segment_index = 0
        self._current_day = 1

    def create_scenarios(self):
        return copy.deepcopy(self._fewshot_task_pool)

    def sample_tasks(self, n_tasks):
        n_tasks = int(n_tasks)
        if n_tasks > len(self._fewshot_task_pool):
            raise ValueError(f"Requested {n_tasks} tasks, but pool size is {len(self._fewshot_task_pool)}")
        return copy.deepcopy(random.sample(self._fewshot_task_pool, n_tasks))

    def set_task(self, task):
        task = copy.deepcopy(task)
        self.task = task
        self._task_mode = str(task.get("mode", "stationary"))
        if self._task_mode == "stationary":
            self.scenario = copy.deepcopy(task["scenario"])
            self._segment_scenarios = [copy.deepcopy(task["scenario"])]
        elif self._task_mode == "nonstationary":
            self._segment_scenarios = copy.deepcopy(task["segment_scenarios"])
            self.scenario = copy.deepcopy(self._segment_scenarios[0])
        else:
            raise ValueError(f"Unsupported task mode: {self._task_mode}")
        self._segment_index = 0
        self._current_day = 1

    def get_task(self):
        return copy.deepcopy(self.task)

    def reset(self):
        if self.task is None:
            raise ValueError("Task must be set before reset().")
        self._segment_index = 0
        self._current_day = 1
        if self._task_mode == "stationary":
            self.scenario = copy.deepcopy(self.task["scenario"])
        else:
            self.scenario = copy.deepcopy(self._segment_scenarios[0])
        return super().reset()

    def step(self, action):
        if self._task_mode == "nonstationary":
            next_segment_index = self._segment_index + 1
            if next_segment_index < len(self._segments):
                next_start = int(self._segments[next_segment_index][0])
                if self._current_day == next_start:
                    _apply_scenario_inplace(self.scenario, self._segment_scenarios[next_segment_index])
                    self._segment_index = next_segment_index
        next_state, reward, done, info = super().step(action)
        self._current_day += 1
        return next_state, reward, done, info


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any) -> int:
    raw = "|".join([str(cfg.RANDOM_SEED)] + [str(part) for part in parts])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _format_seconds(seconds: float) -> str:
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "n/a"
    total = max(0, int(round(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _apply_scenario_inplace(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target["DEMAND"] = copy.deepcopy(source["DEMAND"])
    target["LEADTIME"] = copy.deepcopy(source["LEADTIME"])


def _as_tensor_state_dict(candidate):
    if not isinstance(candidate, dict):
        return None
    tensor_state = {k: v for k, v in candidate.items() if isinstance(k, str) and torch.is_tensor(v)}
    return tensor_state if tensor_state else None


def _extract_agent_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a dict-like state_dict.")
    if "agent_state_dict" in state:
        extracted = _as_tensor_state_dict(state["agent_state_dict"])
        if extracted is not None:
            return extracted
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        nested = state["state_dict"]
        agent_prefixed = {
            str(k)[len("agent."):]: v
            for k, v in nested.items()
            if isinstance(k, str) and k.startswith("agent.") and torch.is_tensor(v)
        }
        if agent_prefixed:
            return agent_prefixed
        extracted = _as_tensor_state_dict(nested)
        if extracted is not None:
            return extracted
    agent_prefixed = {
        k[len("agent."):]: v
        for k, v in state.items()
        if isinstance(k, str) and k.startswith("agent.") and torch.is_tensor(v)
    }
    if agent_prefixed:
        return agent_prefixed
    extracted = _as_tensor_state_dict(state)
    if extracted is not None:
        return extracted
    raise ValueError("Could not extract agent state_dict from checkpoint.")


def _reset_inner_step_sizes(adapter: ProMP, alpha: float) -> None:
    if not hasattr(adapter, "inner_step_sizes"):
        return
    with torch.no_grad():
        for param in adapter.inner_step_sizes.values():
            param.fill_(float(alpha))

def _load_checkpoint(adapter: ProMP, checkpoint_path: str, model_cfg: Dict[str, Any]) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(path, map_location=model_cfg["device"])
    mode = str(cfg.CHECKPOINT_LOAD_MODE).strip().lower()
    state_keys = list(state.keys()) if hasattr(state, "keys") else []

    if mode == "full":
        adapter.load_state_dict(state, strict=False)
        return
    if mode == "agent_only":
        agent_state = _extract_agent_state_dict(state)
        missing, unexpected = adapter.agent.load_state_dict(agent_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"Agent-only checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if bool(cfg.RESET_INNER_STEP_SIZES_ON_AGENT_ONLY_LOAD):
            _reset_inner_step_sizes(adapter, alpha=float(model_cfg["alpha"]))
        return
    if mode == "auto":
        if any(str(key).startswith("agent.") for key in state_keys) or any(str(key).startswith("inner_step_sizes") for key in state_keys):
            adapter.load_state_dict(state, strict=False)
            return
        agent_state = _extract_agent_state_dict(state)
        missing, unexpected = adapter.agent.load_state_dict(agent_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"Auto agent checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return
    raise ValueError(f"Unknown checkpoint load mode: {cfg.CHECKPOINT_LOAD_MODE}")


def _model_cfg(settings: EvalSettings) -> Dict[str, Any]:
    model_cfg = copy.deepcopy(cfg.BASE_MODEL_CONFIG)
    model_cfg["num_task"] = int(settings.num_tasks)
    model_cfg["max_path_length"] = int(settings.max_path_length)
    model_cfg["parallel"] = bool(settings.parallel)
    return model_cfg


def _build_env(task: Dict[str, Any]) -> RandomizedFewShotEnv:
    return RandomizedFewShotEnv(task_pool=[copy.deepcopy(task)], segments=cfg.NONSTATIONARY_SEGMENTS)


def _build_agent(env: MetaEnv, model_cfg: Dict[str, Any]):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    policy_dist = str(model_cfg.get("policy_dist", "categorical")).lower()
    if policy_dist == "categorical":
        num_actions = len(ACTION_SPACE)
        mlp = SimpleMLP(input_dim=obs_dim, output_dim=act_dim * num_actions, hidden_layers=model_cfg["layers"])
        agent = MetaCategoricalAgent(mlp=mlp, num_tasks=model_cfg["num_task"], action_dim=act_dim, num_actions=num_actions)
    elif policy_dist == "gaussian":
        mlp = SimpleMLP(input_dim=obs_dim, output_dim=act_dim, hidden_layers=model_cfg["layers"])
        agent = MetaGaussianAgent(mlp=mlp, num_tasks=model_cfg["num_task"], learn_std=bool(model_cfg.get("learn_std", True)))
    else:
        raise ValueError(f"Unsupported policy distribution: {policy_dist}")
    agent.to(model_cfg["device"])
    return agent


def _build_adapter(env: MetaEnv, agent, model_cfg: Dict[str, Any]) -> ProMP:
    return ProMP(
        env=env,
        max_path_length=model_cfg["max_path_length"],
        agent=agent,
        alpha=model_cfg["alpha"],
        beta=3e-4,
        baseline=LinearFeatureBaseline(),
        tensor_log=None,
        inner_grad_steps=1,
        num_tasks=model_cfg["num_task"],
        outer_iters=1,
        parallel=model_cfg["parallel"],
        rollout_per_task=1,
        clip_eps=0.2,
        trainable_learning_rate=bool(model_cfg.get("trainable_learning_rate", True)),
        inner_step_size_max=float(model_cfg.get("inner_step_size_max", 0.05)),
        device=model_cfg["device"],
    )


def _build_sampler(env: MetaEnv, adapter: ProMP, settings: EvalSettings, rollout_per_task: int) -> MetaSampler:
    rollout_per_task = int(rollout_per_task)
    if rollout_per_task < 1:
        raise ValueError("rollout_per_task must be >= 1")
    envs_per_task = min(int(settings.envs_per_task), rollout_per_task)
    return MetaSampler(
        env=env,
        agent=adapter.agent,
        rollout_per_task=rollout_per_task,
        num_tasks=int(settings.num_tasks),
        max_path_length=int(settings.max_path_length),
        envs_per_task=envs_per_task,
        parallel=bool(settings.parallel),
    )


def _set_tasks_for_samplers(task: Dict[str, Any], samplers: Sequence[MetaSampler]) -> None:
    tasks = [copy.deepcopy(task)]
    for sampler in samplers:
        sampler.vec_env.set_tasks(copy.deepcopy(tasks))


def _adapt_on_support(adapter: ProMP, sample_processor: MetaSampleProcessor, support_sampler: MetaSampler, adapted_params_list: List[OrderedDict], adapt_updates: int, support_seed: int) -> None:
    if int(adapt_updates) <= 0:
        return
    seed_everything(support_seed)
    support_paths = support_sampler.obtain_samples(adapted_params_list, post_update=True)
    support_processed = sample_processor.process_samples(support_paths)
    for _ in range(int(adapt_updates)):
        adapted_params_list[0] = adapter._theta_prime(support_processed[0], params=adapted_params_list[0], create_graph=False)


def _adapt_delta(base_params: OrderedDict, adapted_params: OrderedDict) -> Tuple[float, float]:
    total_sq = 0.0
    for name, p0 in base_params.items():
        delta = adapted_params[name] - p0
        total_sq += float((delta * delta).sum().item())
    return total_sq ** 0.5, 0.0


def _mean_total_cost(query_paths: Dict[int, List[Dict[str, Any]]]) -> Tuple[float, int]:
    rollout_costs = []
    for task_paths in query_paths.values():
        for traj in task_paths:
            rollout_costs.append(float(-np.sum(traj["rewards"])))
    if not rollout_costs:
        return float("nan"), 0
    arr = np.asarray(rollout_costs, dtype=float)
    return float(np.mean(arr)), int(arr.size)


def _retry(fn, description: str):
    last_error = None
    for attempt in range(1, int(cfg.MAX_EPISODE_RETRIES) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= int(cfg.MAX_EPISODE_RETRIES):
                raise
            print(f"[Retry {attempt}/{cfg.MAX_EPISODE_RETRIES}] {description}: {exc}")
    raise last_error


def _generate_stationary_tasks(count: int) -> List[Dict[str, Any]]:
    kwargs = dict(cfg.RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES)
    kwargs["num_scenarios"] = int(count)
    kwargs["seed"] = stable_seed("stationary_task_pool")
    scenarios = create_scenarios(**kwargs)
    return [
        {
            "mode": "stationary",
            "case_name": "All",
            "task_id": idx,
            "task_label": f"Randomized_Stationary_{idx:03d}",
            "scenario": copy.deepcopy(scenario),
        }
        for idx, scenario in enumerate(scenarios, start=1)
    ]


def _generate_nonstationary_tasks(count: int) -> List[Dict[str, Any]]:
    kwargs = dict(cfg.RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES)
    segment_count = len(cfg.NONSTATIONARY_SEGMENTS)
    kwargs["num_scenarios"] = int(count) * segment_count
    kwargs["seed"] = stable_seed("nonstationary_task_pool")
    scenarios = create_scenarios(**kwargs)
    tasks = []
    for index in range(int(count)):
        start = index * segment_count
        end = start + segment_count
        tasks.append(
            {
                "mode": "nonstationary",
                "case_name": "All",
                "task_id": index + 1,
                "task_label": f"Randomized_Nonstationary_{index + 1:03d}",
                "segment_scenarios": [copy.deepcopy(x) for x in scenarios[start:end]],
            }
        )
    return tasks


def _generate_case_randomized_stationary_tasks(count_per_case: int) -> List[Dict[str, Any]]:
    tasks = []
    next_task_id = 1
    for case_index, case_cfg in enumerate(cfg.CASE_RANDOMIZED_CASES, start=1):
        case_name = str(case_cfg["name"])
        kwargs = dict(cfg.RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES)
        kwargs.update(
            {
                "num_scenarios": int(count_per_case),
                "seed": stable_seed("case_randomized", case_name),
                "demand_min": int(case_cfg["demand_min"]),
                "demand_max": int(case_cfg["demand_max"]),
                "leadtime_min": int(case_cfg["leadtime_min"]),
                "leadtime_max": int(case_cfg["leadtime_max"]),
            }
        )
        scenarios = create_scenarios(**kwargs)
        for scenario_index, scenario in enumerate(scenarios, start=1):
            tasks.append(
                {
                    "mode": "stationary",
                    "case_name": case_name,
                    "task_id": next_task_id,
                    "task_label": f"CaseRandomized_{case_index:02d}_{case_name}_{scenario_index:03d}",
                    "scenario": copy.deepcopy(scenario),
                }
            )
            next_task_id += 1
    return tasks


def _select_models(model_names: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not model_names:
        return list(cfg.MODEL_SPECS)
    requested = {str(name).strip() for name in model_names}
    selected = [spec for spec in cfg.MODEL_SPECS if str(spec["name"]) in requested]
    missing = requested - {str(spec["name"]) for spec in selected}
    if missing:
        raise ValueError(f"Unknown model names: {sorted(missing)}")
    return selected


def _validate_modes(modes: Sequence[str]) -> List[str]:
    normalized = [str(mode).strip().lower() for mode in modes]
    invalid = [mode for mode in normalized if mode not in {"stationary", "nonstationary"}]
    if invalid:
        raise ValueError(f"Unsupported environment modes: {invalid}")
    return normalized


def _validate_scenario_mode(scenario_mode: str) -> str:
    normalized = str(scenario_mode).strip().lower()
    if normalized not in {"randomized", "case_randomized"}:
        raise ValueError(f"Unsupported scenario mode: {scenario_mode}")
    return normalized


def _plot_boxplot(rows: Sequence[Dict[str, Any]], environment_mode: str, output_path: Path) -> None:
    mode_rows = [row for row in rows if str(row["environment_mode"]) == environment_mode]
    if not mode_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No {environment_mode} rows were generated.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    shots = sorted({int(row["shot_k"]) for row in mode_rows})
    models = [name for name in cfg.PLOT_MODEL_ORDER if name in {str(row['model']) for row in mode_rows}]
    labels = []
    data = []
    for model_name in models:
        for shot_k in shots:
            vals = [float(row["mean_total_cost"]) for row in mode_rows if str(row["model"]) == model_name and int(row["shot_k"]) == shot_k]
            if vals:
                labels.append(f"{model_name}\nK={shot_k}")
                data.append(vals)

    fig, ax = plt.subplots(figsize=(max(8, len(data)), 6))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(f"{environment_mode.title()} Total Cost Comparison")
    ax.set_xlabel("Model and Shot K")
    ax.set_ylabel("Mean Total Cost")
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_case_boxplot(rows: Sequence[Dict[str, Any]], case_name: str, output_path: Path) -> None:
    case_rows = [row for row in rows if str(row.get("case_name", "")) == case_name]
    if not case_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No rows were generated for case {case_name}.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    shots = sorted({int(row["shot_k"]) for row in case_rows})
    models = [name for name in cfg.PLOT_MODEL_ORDER if name in {str(row['model']) for row in case_rows}]
    labels = []
    data = []
    for model_name in models:
        for shot_k in shots:
            vals = [float(row["mean_total_cost"]) for row in case_rows if str(row["model"]) == model_name and int(row["shot_k"]) == shot_k]
            if vals:
                labels.append(f"{model_name}\nK={shot_k}")
                data.append(vals)

    fig, ax = plt.subplots(figsize=(max(8, len(data)), 6))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(f"{case_name} Total Cost Comparison")
    ax.set_xlabel("Model and Shot K")
    ax.set_ylabel("Mean Total Cost")
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def run_experiment(settings: Optional[EvalSettings] = None, selected_models: Optional[Sequence[str]] = None, output_root: Optional[Path] = None) -> Dict[str, Any]:
    settings = settings or default_settings()
    settings.scenario_mode = _validate_scenario_mode(settings.scenario_mode)
    settings.environment_modes = _validate_modes(settings.environment_modes)
    settings.shots = sorted({max(0, int(shot)) for shot in settings.shots})
    if int(settings.num_tasks) != 1:
        raise ValueError("This randomized evaluator currently supports NUM_TASKS=1 only.")
    if not settings.shots:
        raise ValueError("At least one shot value is required.")
    if settings.scenario_mode == "case_randomized" and settings.environment_modes != ["stationary"]:
        raise ValueError("case_randomized mode supports stationary evaluation only.")

    models = _select_models(selected_models)
    output_root = Path(output_root) if output_root is not None else Path(cfg.RESULTS_ROOT)
    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if settings.scenario_mode == "case_randomized":
        stationary_tasks = _generate_case_randomized_stationary_tasks(int(cfg.CASE_RANDOMIZED_SCENARIO_COUNT_PER_CASE))
        nonstationary_tasks = []
    else:
        stationary_tasks = _generate_stationary_tasks(int(settings.stationary_scenario_count)) if "stationary" in settings.environment_modes else []
        nonstationary_tasks = _generate_nonstationary_tasks(int(settings.nonstationary_sequence_count)) if "nonstationary" in settings.environment_modes else []
    task_specs = list(stationary_tasks) + list(nonstationary_tasks)

    raw_csv = run_dir / "raw_results.csv"
    error_csv = run_dir / "errors.csv"
    progress_csv = run_dir / "progress.csv"
    progress_txt = run_dir / "progress_latest.txt"
    summary_csv = run_dir / "summary_by_mode_model_shot.csv"
    task_summary_csv = run_dir / "summary_by_task_model_shot.csv"
    case_summary_csv = run_dir / "summary_by_case_model_shot.csv"
    stationary_plot = run_dir / "stationary_boxplot_total_cost.png"
    nonstationary_plot = run_dir / "nonstationary_boxplot_total_cost.png"

    _write_json(
        run_dir / "manifest.json",
        {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(run_dir),
            "scenario_mode": settings.scenario_mode,
            "random_seed": int(cfg.RANDOM_SEED),
            "days": int(cfg.DAYS),
            "shots": list(settings.shots),
            "environment_modes": list(settings.environment_modes),
            "stationary_scenario_count": int(settings.stationary_scenario_count),
            "nonstationary_sequence_count": int(settings.nonstationary_sequence_count),
            "query_rollout_per_task": int(settings.query_rollout_per_task),
            "adapt_updates": int(settings.adapt_updates),
            "num_tasks": int(settings.num_tasks),
            "envs_per_task": int(settings.envs_per_task),
            "parallel": bool(settings.parallel),
            "checkpoint_load_mode": str(cfg.CHECKPOINT_LOAD_MODE),
            "models": models,
            "scenario_sampling_overrides": dict(cfg.RANDOMIZED_SCENARIO_SAMPLING_OVERRIDES),
            "case_randomized_scenario_count_per_case": int(cfg.CASE_RANDOMIZED_SCENARIO_COUNT_PER_CASE),
            "case_randomized_cases": copy.deepcopy(cfg.CASE_RANDOMIZED_CASES),
        },
    )
    _write_json(run_dir / "stationary_task_pool.json", {"tasks": copy.deepcopy(stationary_tasks)})
    _write_json(run_dir / "nonstationary_task_pool.json", {"tasks": copy.deepcopy(nonstationary_tasks)})

    total_runs = len(task_specs) * len(settings.shots) * len(models)
    completed_runs = 0
    started = datetime.now()
    ok_rows: List[Dict[str, Any]] = []

    print(f"[FewShot-Randomized] run_dir: {run_dir}")
    print(f"[FewShot-Randomized] total planned runs: {total_runs}")

    for task in task_specs:
        for shot_k in settings.shots:
            for model_spec in models:
                task_label = str(task["task_label"])
                environment_mode = str(task["mode"])
                case_name = str(task.get("case_name", "All"))
                model_name = str(model_spec["name"])
                run_started = datetime.now()
                print(f"[FewShot-Randomized] {completed_runs + 1}/{total_runs} -> mode={environment_mode}, case={case_name}, task={task_label}, model={model_name}, shot={shot_k}")

                status = "ok"
                error_text = ""

                def _cell():
                    support_seed = stable_seed(environment_mode, task_label, "support", shot_k)
                    query_seed = stable_seed(environment_mode, task_label, "query")
                    adapter = None
                    query_sampler = None
                    support_sampler = None
                    try:
                        model_cfg = _model_cfg(settings)
                        seed_everything(stable_seed(environment_mode, task_label, model_name, "init"))
                        env = _build_env(task)
                        agent = _build_agent(env, model_cfg)
                        adapter = _build_adapter(env, agent, model_cfg)
                        _load_checkpoint(adapter, checkpoint_path=str(model_spec["checkpoint_path"]), model_cfg=model_cfg)

                        sample_processor = MetaSampleProcessor(
                            baseline=LinearFeatureBaseline(),
                            discount=0.99,
                            gae_lambda=1.0,
                            normalize_adv=True,
                        )
                        query_sampler = _build_sampler(env, adapter, settings, rollout_per_task=int(settings.query_rollout_per_task))
                        samplers = [query_sampler]
                        if int(shot_k) > 0:
                            support_sampler = _build_sampler(env, adapter, settings, rollout_per_task=int(shot_k))
                            samplers.append(support_sampler)
                        _set_tasks_for_samplers(task, samplers)

                        adapted_params_list = [OrderedDict(adapter.agent.named_parameters())]
                        base_params = OrderedDict((name, param.detach().clone()) for name, param in adapter.agent.named_parameters())
                        adapt_delta_mean = 0.0
                        adapt_delta_std = 0.0
                        if int(shot_k) > 0:
                            _adapt_on_support(adapter, sample_processor, support_sampler, adapted_params_list, int(settings.adapt_updates), int(support_seed))
                            adapt_delta_mean, adapt_delta_std = _adapt_delta(base_params, adapted_params_list[0])

                        seed_everything(int(query_seed))
                        query_paths = query_sampler.obtain_samples(adapted_params_list, post_update=True)
                        mean_total_cost, num_query_trajectories = _mean_total_cost(query_paths)
                        return {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "ok",
                            "scenario_mode": settings.scenario_mode,
                            "environment_mode": environment_mode,
                            "case_name": case_name,
                            "task_label": task_label,
                            "model": model_name,
                            "shot_k": int(shot_k),
                            "mean_total_cost": float(mean_total_cost),
                            "num_query_trajectories": int(num_query_trajectories),
                            "adapt_updates": int(settings.adapt_updates),
                            "adapt_param_delta_l2_mean": float(adapt_delta_mean),
                            "adapt_param_delta_l2_std": float(adapt_delta_std),
                            "support_seed": int(support_seed) if int(shot_k) > 0 else "",
                            "query_seed": int(query_seed),
                            "run_seconds": (datetime.now() - run_started).total_seconds(),
                            "error": "",
                        }
                    finally:
                        if support_sampler is not None:
                            support_sampler.close()
                        if query_sampler is not None:
                            query_sampler.close()
                        if adapter is not None:
                            adapter.close()

                try:
                    row = _retry(_cell, f"{environment_mode}::{task_label}::{model_name}::K={shot_k}")
                    ok_rows.append(row)
                    _append_csv_row(raw_csv, RAW_FIELDS, row)
                except Exception as exc:  # noqa: BLE001
                    status = "error"
                    error_text = f"{type(exc).__name__}: {exc}"
                    _append_csv_row(
                        error_csv,
                        ["timestamp", "scenario_mode", "environment_mode", "case_name", "task_label", "model", "shot_k", "error", "traceback"],
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "scenario_mode": settings.scenario_mode,
                            "environment_mode": environment_mode,
                            "case_name": case_name,
                            "task_label": task_label,
                            "model": model_name,
                            "shot_k": int(shot_k),
                            "error": error_text,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    _append_csv_row(
                        raw_csv,
                        RAW_FIELDS,
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "error",
                            "scenario_mode": settings.scenario_mode,
                            "environment_mode": environment_mode,
                            "case_name": case_name,
                            "task_label": task_label,
                            "model": model_name,
                            "shot_k": int(shot_k),
                            "mean_total_cost": float("nan"),
                            "num_query_trajectories": 0,
                            "adapt_updates": int(settings.adapt_updates),
                            "adapt_param_delta_l2_mean": float("nan"),
                            "adapt_param_delta_l2_std": float("nan"),
                            "support_seed": "",
                            "query_seed": "",
                            "run_seconds": (datetime.now() - run_started).total_seconds(),
                            "error": error_text,
                        },
                    )
                    print(f"[FewShot-Randomized][ERROR] {error_text}")
                    if bool(settings.fail_fast):
                        raise

                completed_runs += 1
                elapsed_seconds = (datetime.now() - started).total_seconds()
                avg_run_seconds = elapsed_seconds / max(completed_runs, 1)
                remaining_runs = max(total_runs - completed_runs, 0)
                eta_seconds = avg_run_seconds * remaining_runs
                eta_finish = datetime.now() + timedelta(seconds=eta_seconds)
                run_seconds = (datetime.now() - run_started).total_seconds()
                progress_pct = (completed_runs / total_runs * 100.0) if total_runs > 0 else 100.0
                _append_csv_row(
                    progress_csv,
                    PROGRESS_FIELDS,
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "completed_runs": completed_runs,
                        "total_runs": total_runs,
                        "progress_pct": f"{progress_pct:.2f}",
                        "environment_mode": environment_mode,
                        "task_label": task_label,
                        "case_name": case_name,
                        "model": model_name,
                        "shot_k": int(shot_k),
                        "status": status,
                        "run_seconds": f"{run_seconds:.2f}",
                        "elapsed_seconds": f"{elapsed_seconds:.2f}",
                        "eta_seconds": f"{eta_seconds:.2f}",
                        "eta_finish_local": eta_finish.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                progress_txt.write_text(
                    (
                        f"completed_runs={completed_runs}\n"
                        f"total_runs={total_runs}\n"
                        f"progress_pct={progress_pct:.2f}\n"
                        f"scenario_mode={settings.scenario_mode}\n"
                        f"last_environment_mode={environment_mode}\n"
                        f"last_case_name={case_name}\n"
                        f"last_task_label={task_label}\n"
                        f"last_model={model_name}\n"
                        f"last_shot_k={int(shot_k)}\n"
                        f"last_status={status}\n"
                        f"last_error={error_text}\n"
                        f"elapsed_seconds={elapsed_seconds:.2f}\n"
                        f"elapsed_hms={_format_seconds(elapsed_seconds)}\n"
                        f"eta_seconds={eta_seconds:.2f}\n"
                        f"eta_hms={_format_seconds(eta_seconds)}\n"
                        f"eta_finish_local={eta_finish.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    ),
                    encoding="utf-8",
                )

    summary_rows = []
    task_summary_rows = []
    case_summary_rows = []
    grouped = {}
    grouped_task = {}
    grouped_case = {}
    for row in ok_rows:
        key = (str(row["scenario_mode"]), str(row["environment_mode"]), str(row["model"]), int(row["shot_k"]))
        grouped.setdefault(key, {"cost": [], "delta": []})
        grouped[key]["cost"].append(float(row["mean_total_cost"]))
        grouped[key]["delta"].append(float(row.get("adapt_param_delta_l2_mean", 0.0)))
        task_key = (str(row["scenario_mode"]), str(row["environment_mode"]), str(row.get("case_name", "All")), str(row["task_label"]), str(row["model"]), int(row["shot_k"]))
        grouped_task.setdefault(task_key, []).append(float(row["mean_total_cost"]))
        case_key = (str(row["scenario_mode"]), str(row["environment_mode"]), str(row.get("case_name", "All")), str(row["model"]), int(row["shot_k"]))
        grouped_case.setdefault(case_key, []).append(float(row["mean_total_cost"]))

    for (scenario_mode, environment_mode, model, shot_k), values in sorted(grouped.items()):
        cost_arr = np.asarray(values["cost"], dtype=float)
        delta_arr = np.asarray(values["delta"], dtype=float)
        n = int(cost_arr.size)
        cost_std = float(np.std(cost_arr, ddof=1)) if n > 1 else 0.0
        delta_std = float(np.std(delta_arr, ddof=1)) if n > 1 else 0.0
        summary_rows.append(
            {
                "scenario_mode": scenario_mode,
                "environment_mode": environment_mode,
                "model": model,
                "shot_k": int(shot_k),
                "n": n,
                "mean_total_cost": float(np.mean(cost_arr)) if n else float("nan"),
                "std_total_cost": cost_std,
                "sem_total_cost": float(cost_std / math.sqrt(n)) if n > 1 else 0.0,
                "mean_adapt_param_delta_l2": float(np.mean(delta_arr)) if n else float("nan"),
                "std_adapt_param_delta_l2": delta_std,
                "sem_adapt_param_delta_l2": float(delta_std / math.sqrt(n)) if n > 1 else 0.0,
            }
        )

    for (scenario_mode, environment_mode, case_name, task_label, model, shot_k), values in sorted(grouped_task.items()):
        arr = np.asarray(values, dtype=float)
        n = int(arr.size)
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        task_summary_rows.append(
            {
                "scenario_mode": scenario_mode,
                "environment_mode": environment_mode,
                "case_name": case_name,
                "task_label": task_label,
                "model": model,
                "shot_k": int(shot_k),
                "n": n,
                "mean_total_cost": float(np.mean(arr)) if n else float("nan"),
                "std_total_cost": std,
                "sem_total_cost": float(std / math.sqrt(n)) if n > 1 else 0.0,
            }
        )

    for (scenario_mode, environment_mode, case_name, model, shot_k), values in sorted(grouped_case.items()):
        arr = np.asarray(values, dtype=float)
        n = int(arr.size)
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        case_summary_rows.append(
            {
                "scenario_mode": scenario_mode,
                "environment_mode": environment_mode,
                "case_name": case_name,
                "model": model,
                "shot_k": int(shot_k),
                "n": n,
                "mean_total_cost": float(np.mean(arr)) if n else float("nan"),
                "std_total_cost": std,
                "sem_total_cost": float(std / math.sqrt(n)) if n > 1 else 0.0,
            }
        )

    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _write_csv(task_summary_csv, TASK_SUMMARY_FIELDS, task_summary_rows)
    _write_csv(case_summary_csv, CASE_SUMMARY_FIELDS, case_summary_rows)
    _plot_boxplot(ok_rows, "stationary", stationary_plot)
    _plot_boxplot(ok_rows, "nonstationary", nonstationary_plot)
    if settings.scenario_mode == "case_randomized":
        for case_cfg in cfg.CASE_RANDOMIZED_CASES:
            case_name = str(case_cfg["name"])
            _plot_case_boxplot(ok_rows, case_name, run_dir / f"case_boxplot_{case_name}.png")

    finished = datetime.now()
    final_report = {
        "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": finished.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "successful_runs": len(ok_rows),
        "failed_runs": total_runs - len(ok_rows),
        "run_dir": str(run_dir),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "task_summary_csv": str(task_summary_csv),
        "case_summary_csv": str(case_summary_csv),
        "progress_csv": str(progress_csv),
        "progress_txt": str(progress_txt),
        "stationary_boxplot": str(stationary_plot),
        "nonstationary_boxplot": str(nonstationary_plot),
    }
    _write_json(run_dir / "final_report.json", final_report)

    if ok_rows:
        print("[FewShot-Randomized] mean total cost by mode/model/shot")
        for row in summary_rows:
            print(f"  mode={row['environment_mode']}, model={row['model']}, K={row['shot_k']}, mean_total_cost={row['mean_total_cost']:.4f}")
    else:
        print("[FewShot-Randomized] no successful rows were produced")

    return final_report
