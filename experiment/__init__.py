"""
Experiment Module
=================
Contains evaluation and testing scripts for generalization performance assessment.
"""

from .few_shot_learning import FewShotEvaluator, main

__all__ = ['FewShotEvaluator', 'main']
