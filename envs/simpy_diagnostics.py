import argparse
import math
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for visualization. Install with: pip install matplotlib"
    ) from exc

from envs.config_SimPy import ASSEMBLY_PROCESS, INVEN_LEVEL_MAX, I, P, TIME_CORRECTION
from envs.log_SimPy import DAILY_COST_REPORT, DAILY_EVENTS, DAILY_REPORTS, STATE_DICT


def _material_item_ids():
    return [
        item_id
        for item_id, item_info in I[ASSEMBLY_PROCESS].items()
        if item_info.get("TYPE") == "Material"
    ]


def _item_names():
    return {item_id: item["NAME"] for item_id, item in I[ASSEMBLY_PROCESS].items()}


def _leadtime_dist_candidates(leadtime_min, leadtime_max):
    uniform_dists = [
        {"Dist_Type": "UNIFORM", "min": i, "max": j}
        for i in range(leadtime_min, leadtime_max + 1)
        for j in range(i, leadtime_max + 1)
    ]
    normal_dists = [
        {"Dist_Type": "NORMAL", "mean": mean, "std": std}
        for mean in [x / 2 for x in range(leadtime_min * 2, leadtime_max * 2 + 1)]
        for std in [0.4, 0.8]
    ]
    poisson_dists = [
        {"Dist_Type": "POISSON", "lam": x / 2}
        for x in range(leadtime_min * 2, leadtime_max * 2 + 1)
    ]
    return uniform_dists + normal_dists + poisson_dists


def _demand_dist_candidates(demand_min, demand_max):
    uniform_dists = [
        {"Dist_Type": "UNIFORM", "min": i, "max": j}
        for i in range(demand_min, demand_max + 1)
        for j in range(i, demand_max + 1)
    ]
    normal_dists = [
        {"Dist_Type": "NORMAL", "mean": mean, "std": std}
        for mean in range(demand_min, demand_max + 1)
        for std in [1, 2, 3]
    ]
    poisson_dists = [
        {"Dist_Type": "POISSON", "lam": lam}
        for lam in range(demand_min, demand_max + 1)
    ]
    return uniform_dists + normal_dists + poisson_dists


def _build_scenario(
    demand_min,
    demand_max,
    leadtime_min,
    leadtime_max,
    demand_mode,
    leadtime_mode,
    seed,
):
    rng = random.Random(seed)
    max_material_id = max(_material_item_ids(), default=0)
    leadtime_profile = [None for _ in range(max_material_id)]
    candidates = _leadtime_dist_candidates(leadtime_min, leadtime_max)
    demand_candidates = _demand_dist_candidates(demand_min, demand_max)

    if not candidates:
        raise ValueError("No leadtime distribution candidates. Check leadtime-min/max values.")
    if not demand_candidates:
        raise ValueError("No demand distribution candidates. Check demand-min/max values.")

    if demand_mode == "fixed_uniform":
        demand_dist = {"Dist_Type": "UNIFORM", "min": demand_min, "max": demand_max}
    elif demand_mode == "per_scenario_random":
        demand_dist = dict(rng.choice(demand_candidates))
    else:
        raise ValueError(f"Unknown demand_mode: {demand_mode}")

    material_ids = _material_item_ids()
    if leadtime_mode == "shared":
        shared_dist = rng.choice(candidates)
        for material_id in material_ids:
            leadtime_profile[material_id - 1] = dict(shared_dist)
    elif leadtime_mode == "per_material_random":
        for material_id in material_ids:
            leadtime_profile[material_id - 1] = dict(rng.choice(candidates))
    else:
        raise ValueError(f"Unknown leadtime_mode: {leadtime_mode}")

    return {
        "DEMAND": demand_dist,
        "LEADTIME": leadtime_profile,
    }


def _format_dist_label(dist_dict):
    dist_type = dist_dict.get("Dist_Type", "UNKNOWN")
    if dist_type == "UNIFORM":
        return f"{dist_type}(min={dist_dict.get('min')}, max={dist_dict.get('max')})"
    if dist_type in ("GAUSSIAN", "NORMAL"):
        return f"{dist_type}(mean={dist_dict.get('mean')}, std={dist_dict.get('std')})"
    if dist_type == "POISSON":
        return f"{dist_type}(lam={dist_dict.get('lam')})"
    return f"{dist_type}({dist_dict})"


def _material_requirements_per_product():
    output_to_process = {}
    for process in P[ASSEMBLY_PROCESS].values():
        output_to_process[process["OUTPUT"]["ID"]] = process

    req = defaultdict(int)

    def explode(item_id, qty):
        item_type = I[ASSEMBLY_PROCESS][item_id]["TYPE"]
        if item_type == "Material":
            req[item_id] += qty
            return
        process = output_to_process.get(item_id)
        if process is None:
            return
        for input_item, input_qty in zip(
            process["INPUT_TYPE_LIST"],
            process["QNTY_FOR_INPUT_ITEM"],
        ):
            explode(input_item["ID"], qty * input_qty)

    explode(I[ASSEMBLY_PROCESS][0]["ID"], 1)
    return dict(req)


def _heuristic_weekly_procurement_process(
    env_module,
    simpy_env,
    procurement,
    supplier,
    inventory,
    daily_events,
    lead_time_dict,
    reorder_point,
    demand_cycle_days,
    material_per_product,
    ordered_qty_daily,
):
    # Run slightly after customer demand generation at the same simulation timestamp.
    yield simpy_env.timeout(TIME_CORRECTION)
    while True:
        item_id = procurement.item_id
        item_name = I[ASSEMBLY_PROCESS][item_id]["NAME"]
        demand_qty = int(I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"])
        on_hand = inventory.on_hand_inventory
        order_size = 0

        if on_hand <= reorder_point:
            order_size = int(demand_qty * material_per_product)

        I[ASSEMBLY_PROCESS][item_id]["LOT_SIZE_ORDER"] = order_size

        daily_events.append(f"==============={item_name}'s Inventory (Heuristic) ===============")
        daily_events.append(
            f"{int(simpy_env.now)}: Heuristic check {item_name} on_hand={on_hand}, "
            f"reorder_point={reorder_point}, demand={demand_qty}, order={order_size}"
        )

        if order_size > 0:
            procurement.order_qty = order_size
            ordered_qty_daily[item_id] += order_size
            inventory.update_inven_level(order_size, "IN_TRANSIT", daily_events)
            env_module.Cost.cal_cost(procurement, "Order cost")
            simpy_env.process(
                supplier.deliver_to_manufacturer(
                    procurement,
                    order_size,
                    inventory,
                    daily_events,
                    lead_time_dict,
                )
            )

        daily_events.append(
            f"{int(simpy_env.now)}: {item_name} in_transit={inventory.in_transition_inventory}, "
            f"total={inventory.in_transition_inventory + inventory.on_hand_inventory}"
        )
        yield simpy_env.timeout(demand_cycle_days * 24)


def _start_heuristic_processes(
    env_module,
    simpy_env,
    inventory_list,
    procurement_list,
    production_list,
    sales,
    customer,
    supplier_list,
    daily_events,
    scenario,
    reorder_point,
    ordered_qty_daily,
):
    simpy_env.process(
        customer.order_product(
            sales,
            inventory_list[I[ASSEMBLY_PROCESS][0]["ID"]],
            daily_events,
            scenario["DEMAND"],
        )
    )

    for production in production_list:
        simpy_env.process(production.process_items(daily_events))

    material_requirements = _material_requirements_per_product()
    demand_cycle_days = int(I[ASSEMBLY_PROCESS][0]["CUST_ORDER_CYCLE"])

    for i, supplier in enumerate(supplier_list):
        material_id = supplier.item_id
        material_per_product = material_requirements.get(material_id, 0)
        simpy_env.process(
            _heuristic_weekly_procurement_process(
                env_module=env_module,
                simpy_env=simpy_env,
                procurement=procurement_list[i],
                supplier=supplier,
                inventory=inventory_list[material_id],
                daily_events=daily_events,
                lead_time_dict=scenario["LEADTIME"],
                reorder_point=reorder_point,
                demand_cycle_days=demand_cycle_days,
                material_per_product=material_per_product,
                ordered_qty_daily=ordered_qty_daily,
            )
        )


def _reset_global_logs():
    DAILY_EVENTS.clear()
    DAILY_REPORTS.clear()
    STATE_DICT.clear()
    for k in DAILY_COST_REPORT.keys():
        DAILY_COST_REPORT[k] = 0


def _set_material_lot_size(lot_size):
    for item_id in _material_item_ids():
        I[ASSEMBLY_PROCESS][item_id]["LOT_SIZE_ORDER"] = lot_size


def run_simulation(
    days,
    lot_size,
    demand_min,
    demand_max,
    leadtime_min,
    leadtime_max,
    cust_order_cycle,
    policy,
    reorder_point,
    demand_mode,
    leadtime_mode,
    seed,
):
    try:
        import envs.environment as env
    except ModuleNotFoundError as exc:
        if "simpy" in str(exc).lower():
            raise SystemExit("simpy is required. Install with: pip install simpy") from exc
        raise

    _reset_global_logs()
    material_ids = _material_item_ids()
    original_cust_cycle = I[ASSEMBLY_PROCESS][0]["CUST_ORDER_CYCLE"]
    original_lot_sizes = {
        item_id: I[ASSEMBLY_PROCESS][item_id]["LOT_SIZE_ORDER"] for item_id in material_ids
    }

    if cust_order_cycle is not None:
        I[ASSEMBLY_PROCESS][0]["CUST_ORDER_CYCLE"] = cust_order_cycle
    _set_material_lot_size(lot_size)
    scenario = _build_scenario(
        demand_min=demand_min,
        demand_max=demand_max,
        leadtime_min=leadtime_min,
        leadtime_max=leadtime_max,
        demand_mode=demand_mode,
        leadtime_mode=leadtime_mode,
        seed=seed,
    )

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
        ) = env.create_env(I, P, DAILY_EVENTS)
        ordered_qty_daily = {mid: 0 for mid in material_ids}

        if policy == "heuristic_rop":
            _start_heuristic_processes(
                env_module=env,
                simpy_env=simpy_env,
                inventory_list=inventory_list,
                procurement_list=procurement_list,
                production_list=production_list,
                sales=sales,
                customer=customer,
                supplier_list=supplier_list,
                daily_events=daily_events,
                scenario=scenario,
                reorder_point=reorder_point,
                ordered_qty_daily=ordered_qty_daily,
            )
        else:
            env.simpy_event_processes(
                simpy_env,
                inventory_list,
                procurement_list,
                production_list,
                sales,
                customer,
                supplier_list,
                daily_events,
                I,
                scenario,
            )
        env.update_daily_report(inventory_list)

        item_names = _item_names()
        rows = []

        for day in range(1, days + 1):
            simpy_env.run(until=simpy_env.now + 24)
            env.update_daily_report(inventory_list)
            total_cost = env.Cost.update_cost_log(inventory_list)
            holding_cost = DAILY_COST_REPORT["Holding cost"]
            process_cost = DAILY_COST_REPORT["Process cost"]
            delivery_cost = DAILY_COST_REPORT["Delivery cost"]
            order_cost = DAILY_COST_REPORT["Order cost"]
            shortage_cost = DAILY_COST_REPORT["Shortage cost"]

            row = {
                "day": day,
                "demand": I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"],
                "total_cost": total_cost,
                "holding_cost": holding_cost,
                "process_cost": process_cost,
                "delivery_cost": delivery_cost,
                "order_cost": order_cost,
                "shortage_cost": shortage_cost,
                "shortage_units": sales.num_shortages,
                "event_count_cumulative": len(DAILY_EVENTS),
            }

            state = STATE_DICT[-1]
            for item_id, item_name in item_names.items():
                row[f"on_hand::{item_name}"] = state[f"On_Hand_{item_name}"]
                row[f"daily_change::{item_name}"] = state[f"Daily_Change_{item_name}"]
                if I[ASSEMBLY_PROCESS][item_id]["TYPE"] == "Material":
                    row[f"in_transit::{item_name}"] = state[f"In_Transit_{item_name}"]
                    if policy == "heuristic_rop":
                        row[f"order_qty::{item_name}"] = ordered_qty_daily[item_id]
                    else:
                        row[f"order_qty::{item_name}"] = I[ASSEMBLY_PROCESS][item_id]["LOT_SIZE_ORDER"]

            rows.append(row)
            for item_id in material_ids:
                ordered_qty_daily[item_id] = 0
            sales.num_shortages = 0
            env.Cost.clear_cost()

        df = pd.DataFrame(rows)
        item_names = _item_names()
        leadtime_lines = []
        for material_id in material_ids:
            material_name = item_names[material_id]
            dist = scenario["LEADTIME"][material_id - 1]
            leadtime_lines.append(f"{material_name}: {_format_dist_label(dist)}")
        dist_meta = {
            "demand": _format_dist_label(scenario["DEMAND"]),
            "demand_mode": demand_mode,
            "leadtime_mode": leadtime_mode,
            "leadtime_lines": leadtime_lines,
        }
        checks = run_sanity_checks(df, item_names, material_ids)
        return df, checks, dist_meta
    finally:
        I[ASSEMBLY_PROCESS][0]["CUST_ORDER_CYCLE"] = original_cust_cycle
        for item_id, lot_size_restore in original_lot_sizes.items():
            I[ASSEMBLY_PROCESS][item_id]["LOT_SIZE_ORDER"] = lot_size_restore


def run_sanity_checks(df, item_names, material_ids):
    checks = []
    for item_name in item_names.values():
        series = df[f"on_hand::{item_name}"]
        checks.append(
            {
                "check": f"0 <= on_hand({item_name}) <= capacity",
                "passed": bool((series >= 0).all() and (series <= INVEN_LEVEL_MAX).all()),
                "details": f"min={series.min()}, max={series.max()}",
            }
        )

    for item_id in material_ids:
        item_name = item_names[item_id]
        in_transit = df[f"in_transit::{item_name}"]
        checks.append(
            {
                "check": f"in_transit({item_name}) >= 0",
                "passed": bool((in_transit >= 0).all()),
                "details": f"min={in_transit.min()}",
            }
        )

    finite_cost = df["total_cost"].apply(math.isfinite).all()
    checks.append(
        {"check": "daily total_cost is finite", "passed": bool(finite_cost), "details": ""}
    )
    component_sum = (
        df["holding_cost"]
        + df["process_cost"]
        + df["delivery_cost"]
        + df["order_cost"]
        + df["shortage_cost"]
    )
    cost_match = (component_sum - df["total_cost"]).abs().max() < 1e-6
    checks.append(
        {
            "check": "daily total_cost equals sum of cost components",
            "passed": bool(cost_match),
            "details": f"max_abs_error={(component_sum - df['total_cost']).abs().max():.6g}",
        }
    )
    return checks


def save_visualizations(df, output_dir, dist_meta=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    on_hand_cols = [c for c in df.columns if c.startswith("on_hand::")]
    order_cols = [c for c in df.columns if c.startswith("order_qty::")]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    for col in on_hand_cols:
        axes[0].plot(df["day"], df[col], label=col.replace("on_hand::", ""))
    axes[0].set_title("Daily On-hand Inventory by Item")
    axes[0].set_ylabel("Units")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[0].grid(alpha=0.3)

    for col in order_cols:
        axes[1].step(df["day"], df[col], where="mid", label=col.replace("order_qty::", ""))
    demand_label = "Demand"
    leadtime_lines = []
    if dist_meta:
        demand_label = f"Demand [{dist_meta.get('demand', 'N/A')}]"
        leadtime_mode = dist_meta.get("leadtime_mode", "N/A")
        leadtime_lines = [f"Leadtime mode: {leadtime_mode}"] + dist_meta.get("leadtime_lines", [])

    axes[1].plot(df["day"], df["demand"], color="black", linewidth=2, label=demand_label)
    # Add leadtime distribution info to legend without plotting extra series.
    for line in leadtime_lines:
        axes[1].plot([], [], color="none", label=line)
    axes[1].set_title("Daily Ordered Quantity and Demand")
    axes[1].set_ylabel("Units")
    axes[1].legend(loc="upper left", ncol=2, fontsize=7)
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["day"], df["total_cost"], color="black", linewidth=2, label="Total Cost")
    axes[2].plot(df["day"], df["holding_cost"], label="Holding Cost")
    axes[2].plot(df["day"], df["process_cost"], label="Process Cost")
    axes[2].plot(df["day"], df["delivery_cost"], label="Delivery Cost")
    axes[2].plot(df["day"], df["order_cost"], label="Order Cost")
    axes[2].plot(df["day"], df["shortage_cost"], label="Shortage Cost")
    axes[2].set_title("Daily Total Cost and Cost Components")
    axes[2].set_xlabel("Day")
    axes[2].set_ylabel("Value")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    cumulative_cols = [
        "holding_cost",
        "process_cost",
        "delivery_cost",
        "order_cost",
        "shortage_cost",
        "total_cost",
    ]
    cumulative_df = df[cumulative_cols].cumsum()
    axes[3].plot(df["day"], cumulative_df["total_cost"], color="black", linewidth=2, label="Cum Total")
    axes[3].plot(df["day"], cumulative_df["holding_cost"], label="Cum Holding")
    axes[3].plot(df["day"], cumulative_df["process_cost"], label="Cum Process")
    axes[3].plot(df["day"], cumulative_df["delivery_cost"], label="Cum Delivery")
    axes[3].plot(df["day"], cumulative_df["order_cost"], label="Cum Order")
    axes[3].plot(df["day"], cumulative_df["shortage_cost"], label="Cum Shortage")
    axes[3].set_title("Cumulative Cost by Component")
    axes[3].set_xlabel("Day")
    axes[3].set_ylabel("Cumulative Cost")
    axes[3].legend(loc="upper left")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    png_path = output_dir / "simpy_diagnostics.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    csv_path = output_dir / "simpy_daily_metrics.csv"
    df.to_csv(csv_path, index=False)
    return png_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="SimPy inventory diagnostics runner")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--lot-size", type=int, default=5)
    parser.add_argument("--demand-min", type=int, default=6)
    parser.add_argument("--demand-max", type=int, default=12)
    parser.add_argument(
        "--demand-mode",
        type=str,
        default="fixed_uniform",
        choices=["fixed_uniform", "per_scenario_random"],
        help="fixed_uniform: keep demand as UNIFORM(min,max), per_scenario_random: sample distribution type/params per scenario.",
    )
    parser.add_argument("--leadtime-min", type=int, default=1)
    parser.add_argument("--leadtime-max", type=int, default=3)
    parser.add_argument(
        "--leadtime-mode",
        type=str,
        default="per_material_random",
        choices=["shared", "per_material_random"],
        help="shared: all suppliers share one leadtime distribution, per_material_random: each supplier draws its own.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for scenario generation.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="fixed_lot",
        choices=["fixed_lot", "heuristic_rop"],
        help="fixed_lot: use constant LOT_SIZE_ORDER, heuristic_rop: reorder-point policy.",
    )
    parser.add_argument(
        "--reorder-point",
        type=int,
        default=1,
        help="Used only when --policy heuristic_rop.",
    )
    parser.add_argument(
        "--cust-order-cycle",
        type=int,
        default=7,
        help="Customer demand order cycle in days.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("envs") / "diagnostics_outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df, checks, dist_meta = run_simulation(
        days=args.days,
        lot_size=args.lot_size,
        demand_min=args.demand_min,
        demand_max=args.demand_max,
        leadtime_min=args.leadtime_min,
        leadtime_max=args.leadtime_max,
        cust_order_cycle=args.cust_order_cycle,
        policy=args.policy,
        reorder_point=args.reorder_point,
        demand_mode=args.demand_mode,
        leadtime_mode=args.leadtime_mode,
        seed=args.seed,
    )
    png_path, csv_path = save_visualizations(df, args.output_dir, dist_meta=dist_meta)

    print(f"[Saved] Plot: {png_path}")
    print(f"[Saved] CSV : {csv_path}")
    print("[Sanity Checks]")
    for result in checks:
        mark = "PASS" if result["passed"] else "FAIL"
        details = f" ({result['details']})" if result["details"] else ""
        print(f"  - {mark}: {result['check']}{details}")


if __name__ == "__main__":
    main()
