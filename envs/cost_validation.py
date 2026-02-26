import simpy

import envs.environment as env
from envs.config_SimPy import ASSEMBLY_PROCESS, I
from envs.log_SimPy import DAILY_COST_REPORT


def _reset_cost_report():
    for key in DAILY_COST_REPORT:
        DAILY_COST_REPORT[key] = 0


def _assert_close(actual, expected, name, tol=1e-9):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name} mismatch: actual={actual}, expected={expected}")


def validate_holding_cost():
    sim = simpy.Environment()
    inv = env.Inventory(sim, 0, holding_cost=24)  # 24/day -> 1/hour
    inv.on_hand_inventory = 10
    _reset_cost_report()
    sim.run(until=24)
    total = env.Cost.update_cost_log([inv])
    _assert_close(DAILY_COST_REPORT["Holding cost"], 240.0, "holding_cost")
    _assert_close(total, 240.0, "total_cost_from_holding")


def validate_order_cost():
    sim = simpy.Environment()
    procurement = env.Procurement(sim, item_id=1, purchase_cost=2, setup_cost=1)
    procurement.order_qty = 7
    _reset_cost_report()
    env.Cost.cal_cost(procurement, "Order cost")
    _assert_close(DAILY_COST_REPORT["Order cost"], 15.0, "order_cost")


def validate_delivery_and_shortage_cost_no_carryover():
    sim = simpy.Environment()
    product_inv = env.Inventory(sim, 0, holding_cost=1)
    sales = env.Sales(sim, 0, delivery_cost=1, setup_cost=1, shortage=50, due_date=1)
    events = []

    _reset_cost_report()
    product_inv.on_hand_inventory = 5
    sim.process(sales._deliver_to_cust(5, product_inv, events))
    sim.run(until=25)
    _assert_close(DAILY_COST_REPORT["Delivery cost"], 6.0, "delivery_cost_full_fill")
    _assert_close(DAILY_COST_REPORT["Shortage cost"], 0.0, "shortage_cost_full_fill")

    _reset_cost_report()
    product_inv.on_hand_inventory = 0
    sim.process(sales._deliver_to_cust(5, product_inv, events))
    sim.run(until=50)
    # Setup cost only when nothing was delivered, plus shortage penalty.
    _assert_close(DAILY_COST_REPORT["Delivery cost"], 1.0, "delivery_cost_zero_delivery")
    _assert_close(DAILY_COST_REPORT["Shortage cost"], 250.0, "shortage_cost_zero_delivery")
    _assert_close(sales.delivery_item, 0.0, "delivery_item_reset")


def validate_due_date_parameter_is_used():
    sim = simpy.Environment()
    product_inv = env.Inventory(sim, 0, holding_cost=1)
    sales = env.Sales(sim, 0, delivery_cost=1, setup_cost=1, shortage=50, due_date=1)
    events = []

    # Temporarily set global due date to a different value to ensure instance due_date is used.
    original_due_date = I[ASSEMBLY_PROCESS][0]["DUE_DATE"]
    I[ASSEMBLY_PROCESS][0]["DUE_DATE"] = 7
    try:
        _reset_cost_report()
        product_inv.on_hand_inventory = 1
        sim.process(sales._deliver_to_cust(1, product_inv, events))
        sim.run(until=23.9)
        _assert_close(DAILY_COST_REPORT["Delivery cost"], 0.0, "no_delivery_before_due")
        sim.run(until=24.1)
        if DAILY_COST_REPORT["Delivery cost"] <= 0:
            raise AssertionError("delivery did not occur at due_date=1 day")
    finally:
        I[ASSEMBLY_PROCESS][0]["DUE_DATE"] = original_due_date


def main():
    validate_holding_cost()
    validate_order_cost()
    validate_delivery_and_shortage_cost_no_carryover()
    validate_due_date_parameter_is_used()
    print("All cost validation checks passed.")


if __name__ == "__main__":
    main()
