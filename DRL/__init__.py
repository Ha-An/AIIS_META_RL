# __init__.py
"""
DRL (Deep Reinforcement Learning) Module
=========================================
Standard RL algorithms without meta-learning optimization.

Algorithms:
- PPO: Proximal Policy Optimization

Features:
- Supports fixed task training
- Supports randomized task training
- Reuses Meta-RL framework components for fair comparison
- Compatible hyperparameters with Meta-RL experiments
"""

from .PPO import PPO

__all__ = ['PPO']
