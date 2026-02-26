# AIIS_META_PACKAGE
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118</br>
pip install scipy</br>
pip install simpy</br>
pip install gym</br>
pip install gymnasium</br>
pip install pandas</br>
pip install matplotlib</br>

## SimPy Diagnostics
Run the command below to inspect daily inventory, order quantity, demand, and sanity checks.

```bash
python -m envs.simpy_diagnostics --days 60 --lot-size 5 --policy fixed_lot --cust-order-cycle 7
```

Heuristic standalone mode (no DRL):
```bash
python -m envs.simpy_diagnostics --policy heuristic_rop --reorder-point 1 --days 200 --cust-order-cycle 7
```

Outputs:
- `envs/diagnostics_outputs/simpy_diagnostics.png`
- `envs/diagnostics_outputs/simpy_daily_metrics.csv`

Built-in checks:
- Each item's on-hand inventory stays within `0 ~ INVEN_LEVEL_MAX(50)`
- Each material's in-transit inventory is non-negative
- Daily `total_cost` is finite
## FlowChart
<img width="638" height="586" alt="meta-package_without_envs drawio" src="https://github.com/user-attachments/assets/e32b34aa-d7b7-4126-a3a5-c1e1fc3b1bd3" />
