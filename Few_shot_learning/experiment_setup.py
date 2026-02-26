"""
Compatibility entrypoint.

This file is kept to avoid breaking existing run commands, but the actual
random-scenario batch experiment now lives in:
    Few_shot_learning/run_batch_random_scenarios.py

For fixed-scenario comparison (4 fixed tasks x 30 seeds), use:
    Few_shot_learning/run_batch_fixed_scenarios.py
"""

from Few_shot_learning.run_batch_random_scenarios import main


if __name__ == "__main__":
    print("[FewShot] Redirecting to run_batch_random_scenarios.py")
    main()
