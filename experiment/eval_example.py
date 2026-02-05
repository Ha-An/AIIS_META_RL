"""
Example: Few-Shot Learning Evaluation Script
==============================================
This script demonstrates how to use the FewShotEvaluator class
to evaluate the generalization performance of a trained meta-RL model.

Usage:
    python eval_example.py
    
    Or with custom parameters:
    python eval_example.py --model-path path/to/model --num-tasks 20
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment.few_shot_learning import FewShotEvaluator


def create_example_configs():
    """
    Create several example evaluation configurations.
    
    Each configuration tests different aspects of generalization:
    1. Low demand scenarios
    2. High demand scenarios
    3. Mixed demand scenarios with varying leadtimes
    """
    
    configs = {
        'low_demand': {
            'name': 'Low Demand Distribution',
            'demand_ranges': [(3, 8)],
            'leadtime_ranges': [(1, 2), (2, 3)],
            'num_test_tasks': 8,
        },
        
        'high_demand': {
            'name': 'High Demand Distribution',
            'demand_ranges': [(18, 25)],
            'leadtime_ranges': [(2, 3), (3, 5)],
            'num_test_tasks': 8,
        },
        
        'mixed': {
            'name': 'Mixed Demand & Leadtime Distribution',
            'demand_ranges': [(5, 10), (10, 15), (15, 20)],
            'leadtime_ranges': [(1, 2), (2, 3), (3, 5)],
            'num_test_tasks': 12,
        },
        
        'extreme': {
            'name': 'Extreme Scenarios',
            'demand_ranges': [(2, 5), (5, 10), (20, 25), (25, 30)],
            'leadtime_ranges': [(1, 1), (1, 5)],
            'num_test_tasks': 12,
        }
    }
    
    return configs


def run_evaluation(config_name='mixed', num_support_episodes=5, num_query_episodes=10):
    """
    Run few-shot evaluation with specified configuration.
    
    Args:
        config_name (str): Name of configuration ('low_demand', 'high_demand', 'mixed', 'extreme')
        num_support_episodes (int): Number of episodes for adaptation
        num_query_episodes (int): Number of episodes for evaluation
    """
    
    # Get configuration
    configs = create_example_configs()
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config_spec = configs[config_name]
    
    # Build full config
    base_dir = Path(__file__).parent.parent
    eval_config = {
        'model_path': base_dir / 'Saved_Model' / 'Train_1' / 'saved_model',
        
        'model_params': {
            'Layers': [64, 64, 64],
            'num_task': 5,
            'learn_std': True
        },
        
        'num_test_tasks': config_spec['num_test_tasks'],
        'num_support_episodes': num_support_episodes,
        'num_query_episodes': num_query_episodes,
        'demand_ranges': config_spec['demand_ranges'],
        'leadtime_ranges': config_spec['leadtime_ranges'],
        
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }
    
    print("\n" + "="*70)
    print(f"CONFIGURATION: {config_spec['name']}")
    print("="*70)
    print(f"Demand ranges: {config_spec['demand_ranges']}")
    print(f"Leadtime ranges: {config_spec['leadtime_ranges']}")
    print(f"Test tasks: {config_spec['num_test_tasks']}")
    print(f"Support episodes per task: {num_support_episodes}")
    print(f"Query episodes per task: {num_query_episodes}")
    print(f"Device: {eval_config['device']}")
    
    try:
        # Initialize evaluator
        evaluator = FewShotEvaluator(
            model_path=str(eval_config['model_path']),
            device=eval_config['device']
        )
        
        # Load model
        evaluator.load_model(eval_config['model_params'])
        
        # Create test tasks
        test_tasks = evaluator.create_custom_task_distributions(
            num_test_tasks=eval_config['num_test_tasks'],
            demand_ranges=eval_config['demand_ranges'],
            leadtime_ranges=eval_config['leadtime_ranges']
        )
        
        # Run evaluation
        results = evaluator.evaluate_on_task_distribution(
            test_tasks=test_tasks,
            num_support_episodes=eval_config['num_support_episodes'],
            num_query_episodes=eval_config['num_query_episodes'],
            run_name=f"{config_name.upper()}_Evaluation"
        )
        
        # Close evaluator
        evaluator.close()
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"Average Query Cost: {results['overall_avg_cost']:.2f}")
        print(f"Std Dev: {results['overall_std_cost']:.2f}")
        print(f"Min Cost: {results['overall_min_cost']:.2f}")
        print(f"Max Cost: {results['overall_max_cost']:.2f}")
        print(f"\nMetrics:")
        print(f"  - Test tasks completed: {results['num_test_tasks']}")
        print(f"  - Total episodes evaluated: {results['num_test_tasks'] * results['num_query_episodes']}")
        print(f"\nTensorBoard logs saved to: {evaluator.log_dir}")
        print("="*70)
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"  Model not found at: {eval_config['model_path']}")
        print("  Please train a model first using AIIS_META/main.py")
        return None
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


def run_all_configs(num_support_episodes=3, num_query_episodes=5):
    """
    Run evaluation for all configuration examples.
    
    Args:
        num_support_episodes (int): Episodes per task for adaptation
        num_query_episodes (int): Episodes per task for evaluation
    """
    
    configs = create_example_configs()
    all_results = {}
    
    print("\n" + "="*70)
    print("RUNNING ALL EVALUATION CONFIGURATIONS")
    print("="*70)
    
    for config_name in configs.keys():
        print(f"\n→ Running: {config_name}")
        results = run_evaluation(
            config_name=config_name,
            num_support_episodes=num_support_episodes,
            num_query_episodes=num_query_episodes
        )
        
        if results is not None:
            all_results[config_name] = results
    
    # Print comparative summary
    if all_results:
        print("\n" + "="*70)
        print("COMPARATIVE SUMMARY")
        print("="*70)
        print(f"{'Config':<15} {'Avg Cost':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10}")
        print("-" * 70)
        
        for config_name, results in all_results.items():
            print(f"{config_name:<15} "
                  f"{results['overall_avg_cost']:<12.2f} "
                  f"{results['overall_std_cost']:<12.2f} "
                  f"{results['overall_min_cost']:<10.2f} "
                  f"{results['overall_max_cost']:<10.2f}")
        
        print("="*70)
    
    return all_results


def run_shot_progression(model_name='train_1', max_shots=20, num_query_episodes=5, num_test_tasks=10):
    """
    Run evaluation with increasing number of support episodes (shots).
    
    This evaluates how performance improves as the number of few-shot
    adaptation episodes increases (0-shot, 1-shot, 2-shot, ..., 20-shot).
    
    Args:
        model_name (str): Model to evaluate ('train_1' for ProMP or 'train_2' for VPG_MAML)
        max_shots (int): Maximum number of support episodes to test
        num_query_episodes (int): Number of query episodes per task
        num_test_tasks (int): Number of test tasks
    """
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    
    base_dir = Path(__file__).parent.parent
    
    # Determine model path and log directory based on model name
    model_name_lower = model_name.lower().replace('-', '_')
    if model_name_lower in ['train_1', 'train1', 'promp']:
        model_path = base_dir / 'Saved_Model' / 'Train_1' / 'saved_model'
        log_prefix = 'Shot_Progression_ProMP'
        model_type = 'ProMP (Train_1)'
    elif model_name_lower in ['train_2', 'train2', 'vpg_maml', 'vpgmaml']:
        model_path = base_dir / 'Saved_Model' / 'Train_2' / 'saved_model'
        log_prefix = 'Shot_Progression_VPG_MAML'
        model_type = 'VPG_MAML (Train_2)'
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'train_1' (ProMP) or 'train_2' (VPG_MAML)")
    
    # Create evaluator
    evaluator = FewShotEvaluator(
        model_path=str(model_path),
        log_dir=str(base_dir / 'Tensorboard_logs' / log_prefix),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Load model
    evaluator.load_model({
        'Layers': [64, 64, 64],
        'num_task': 5,
        'learn_std': True
    })
    
    # Create test tasks (same for all shot counts)
    test_tasks = evaluator.create_custom_task_distributions(
        num_test_tasks=num_test_tasks,
        demand_ranges=[(5, 10), (10, 15), (15, 20)],
        leadtime_ranges=[(1, 2), (2, 3), (3, 5)]
    )
    
    # Run evaluations for each shot count
    print("\n" + "="*70)
    print(f"FEW-SHOT LEARNING: SHOT PROGRESSION ANALYSIS - {model_type}")
    print("="*70)
    print(f"Max Shots: {max_shots}")
    print(f"Query Episodes: {num_query_episodes}")
    print(f"Test Tasks: {num_test_tasks}")
    print("="*70)
    
    shot_results = {}
    avg_costs_per_shot = {}
    
    for num_shots in range(0, max_shots + 1):
        print(f"\n📊 Evaluating with {num_shots}-shot (support_episodes={num_shots})")
        
        try:
            results = evaluator.evaluate_on_task_distribution(
                test_tasks=test_tasks,
                num_support_episodes=num_shots,
                num_query_episodes=num_query_episodes,
                run_name=f"Shot_{num_shots}"
            )
            
            shot_results[num_shots] = results
            avg_costs_per_shot[num_shots] = results['overall_avg_cost']
            
            print(f"  ✓ Average Cost: {results['overall_avg_cost']:.2f}")
            print(f"    Std Dev: {results['overall_std_cost']:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
            shot_results[num_shots] = None
    
    evaluator.close()
    
    # Print summary
    print("\n" + "="*70)
    print("SHOT PROGRESSION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Shots':<8} {'Avg Cost':<15} {'Change vs 0-shot':<20}")
    print("-" * 70)
    
    baseline_cost = avg_costs_per_shot.get(0, None)
    
    for num_shots in sorted(avg_costs_per_shot.keys()):
        avg_cost = avg_costs_per_shot[num_shots]
        if baseline_cost is not None:
            change = avg_cost - baseline_cost
            change_pct = (change / baseline_cost * 100) if baseline_cost != 0 else 0
            print(f"{num_shots:<8} {avg_cost:<15.2f} {change:+.2f} ({change_pct:+.1f}%)")
        else:
            print(f"{num_shots:<8} {avg_cost:<15.2f} {'N/A':<20}")
    
    print("="*70)
    print("\n✓ Evaluation complete. TensorBoard logs saved to:")
    print(f"  {evaluator.log_dir}")
    print("\nView with:")
    log_path = evaluator.log_dir
    print(f"  tensorboard --logdir=\"{log_path}\"")
    
    return shot_results, avg_costs_per_shot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-Shot Learning Evaluation Examples"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='standard',
        choices=['standard', 'shot_progression'],
        help='Evaluation mode (default: standard)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='mixed',
        choices=['low_demand', 'high_demand', 'mixed', 'extreme', 'all'],
        help='Configuration to run (default: mixed)'
    )
    parser.add_argument(
        '--support-episodes',
        type=int,
        default=5,
        help='Number of support episodes per task (default: 5)'
    )
    parser.add_argument(
        '--query-episodes',
        type=int,
        default=10,
        help='Number of query episodes per task (default: 10)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model (default: Saved_Model/Train_1/saved_model)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='train_1',
        choices=['train_1', 'train_2', 'train1', 'train2', 'promp', 'vpg_maml'],
        help='Model to evaluate: train_1/promp (ProMP) or train_2/vpg_maml (VPG_MAML) (default: train_1)'
    )
    parser.add_argument(
        '--max-shots',
        type=int,
        default=20,
        help='Maximum number of shots for shot_progression mode (default: 20)'
    )
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=10,
        help='Number of test tasks (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation based on mode
    if args.mode == 'shot_progression':
        # Run shot progression analysis
        shot_results, avg_costs = run_shot_progression(
            model_name=args.model,
            max_shots=args.max_shots,
            num_query_episodes=args.query_episodes,
            num_test_tasks=args.num_tasks
        )
    else:
        # Run standard evaluation
        if args.config == 'all':
            results = run_all_configs(
                num_support_episodes=args.support_episodes,
                num_query_episodes=args.query_episodes
            )
        else:
            results = run_evaluation(
                config_name=args.config,
                num_support_episodes=args.support_episodes,
                num_query_episodes=args.query_episodes
            )
        
        if results is not None:
            print("\n✓ Evaluation completed successfully!")
            print("\nView results with TensorBoard:")
            print("  tensorboard --logdir=./Tensorboard_logs/Few_Shot_Eval")

