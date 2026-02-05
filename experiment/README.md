# Few-Shot Learning Evaluation Guide

## Overview

`few_shot_learning.py` is a comprehensive module for evaluating the **generalization performance** of trained meta-RL models. It simulates few-shot learning scenarios where the policy must quickly adapt to new task distributions.

## Key Features

### 1. **Model Loading** 
- Loads pre-trained meta-learning models from saved checkpoints
- Supports custom model architectures

### 2. **Custom Task Distributions**
- Create test tasks with different demand ranges (e.g., low/medium/high)
- Configure leadtime distributions independently per material
- Full control over test scenario parameters

### 3. **Few-Shot Adaptation Protocol**
- **Support Phase**: Collect episodes on a new task
- **Adaptation Phase**: Update policy parameters for the specific task
- **Query Phase**: Evaluate performance on adapted policy
- Metrics tracking at multiple levels (per-task, overall)

### 4. **TensorBoard Visualization**
- Real-time cost monitoring
- Per-task performance comparison
- Cost component breakdown (Holding, Delivery, Order, Process, Shortage)
- Overall generalization metrics

## File Structure

```
experiment/
├── __init__.py
└── few_shot_learning.py          # Main evaluation module
```

## Usage

### Basic Usage (Default Configuration)

```python
from experiment.few_shot_learning import main

# Run evaluation with default settings
results = main()
```

### Custom Configuration

```python
from experiment.few_shot_learning import FewShotEvaluator
import torch

# Define custom evaluation configuration
eval_config = {
    # Model path
    'model_path': 'path/to/saved_model',
    
    # Model architecture (must match training config)
    'model_params': {
        'Layers': [64, 64, 64],           # Hidden layer sizes
        'num_task': 5,                     # Number of parallel tasks
        'learn_std': True                  # Learn policy std dev
    },
    
    # Evaluation parameters
    'num_test_tasks': 15,                 # Number of different test tasks
    'num_support_episodes': 5,            # Episodes for few-shot adaptation
    'num_query_episodes': 10,             # Episodes to evaluate performance
    
    # Custom demand distributions
    'demand_ranges': [
        (5, 10),                          # Low demand (min, max units)
        (10, 15),                         # Medium demand
        (15, 20),                         # High demand
        (20, 25),                         # Very high demand (new!)
    ],
    
    # Custom leadtime distributions
    'leadtime_ranges': [
        (1, 2),                           # Short leadtime (min, max days)
        (2, 3),                           # Medium leadtime
        (3, 5),                           # Long leadtime
    ],
    
    # Device
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Initialize evaluator
evaluator = FewShotEvaluator(
    model_path=eval_config['model_path'],
    device=eval_config['device']
)

# Load model with specified architecture
evaluator.load_model(eval_config['model_params'])

# Create custom test tasks
test_tasks = evaluator.create_custom_task_distributions(
    num_test_tasks=eval_config['num_test_tasks'],
    demand_ranges=eval_config['demand_ranges'],
    leadtime_ranges=eval_config['leadtime_ranges']
)

# Run full evaluation
results = evaluator.evaluate_on_task_distribution(
    test_tasks=test_tasks,
    num_support_episodes=eval_config['num_support_episodes'],
    num_query_episodes=eval_config['num_query_episodes'],
    run_name="Custom_Eval"
)

evaluator.close()
```

### Advanced: Manual Task Control

```python
from experiment.few_shot_learning import FewShotEvaluator

evaluator = FewShotEvaluator(model_path='path/to/model')
evaluator.load_model(model_params)

# Create a specific test task
custom_task = {
    'DEMAND': {'Dist_Type': 'UNIFORM', 'min': 8, 'max': 12},
    'LEADTIME': [
        {'Dist_Type': 'UNIFORM', 'min': 1, 'max': 3},  # Material 1
        {'Dist_Type': 'UNIFORM', 'min': 2, 'max': 4},  # Material 2
        # ... (MAT_COUNT materials)
    ],
    'task_id': 'custom_001'
}

# Evaluate on single task
episode_data = evaluator.run_episode(
    task=custom_task,
    deterministic=False
)

print(f"Total cost: {episode_data['total_cost']}")
print(f"Cost breakdown: {episode_data['cost_dict']}")
```

## Configuration Parameters

### Model Parameters (`model_params`)
- `Layers` (list): MLP hidden layer sizes. Must match training configuration.
- `num_task` (int): Number of parallel tasks. Must match training.
- `learn_std` (bool): Whether policy learns log_std. Must match training.

### Evaluation Parameters
- `num_test_tasks` (int): Number of different test scenarios
  - Larger values provide better generalization estimate
  - Recommended: 10-20
  
- `num_support_episodes` (int): Episodes collected per task for adaptation
  - More episodes → better task-specific adaptation
  - Recommended: 3-10
  
- `num_query_episodes` (int): Episodes to evaluate performance
  - More episodes → more stable evaluation metrics
  - Recommended: 10-20

### Task Distribution Parameters
- `demand_ranges`: List of (min, max) demand tuples
  - Each tuple defines a demand level to test
  - Example: `[(5,10), (10,15), (15,20)]` tests 3 demand levels
  
- `leadtime_ranges`: List of (min, max) leadtime tuples
  - Each task randomly samples from these ranges
  - Example: `[(1,2), (2,3), (3,5)]` tests 3 leadtime levels

## Output & Metrics

### Returned Results Dictionary
```python
results = {
    'overall_avg_cost': float,              # Mean cost across all queries
    'overall_std_cost': float,              # Std dev of costs
    'overall_min_cost': float,              # Best episode cost
    'overall_max_cost': float,              # Worst episode cost
    'num_test_tasks': int,                  # Number of test tasks
    'num_query_episodes': int,              # Episodes per task
    'per_task_results': dict,               # Per-task metrics
    'all_query_costs': list,                # All individual costs
}
```

### TensorBoard Metrics Logged

#### Per-Task Metrics (for each task)
- `Task_{id}/Support_Phase/Avg_Cost`: Average cost during support phase
- `Task_{id}/Query_Phase/Avg_Cost`: Average cost during query phase
- `Task_{id}/Query_Phase/Std_Cost`: Std deviation during query phase
- `Task_{id}/Cost_Components/{type}`: Cost breakdown by type

#### Overall Metrics
- `Overall/Avg_Query_Cost`: Mean query cost across all tasks
- `Overall/Std_Query_Cost`: Std dev of query costs
- `Overall/Min_Query_Cost`: Best query cost
- `Overall/Max_Query_Cost`: Worst query cost
- `Overall/Cost_Components/{type}_Mean`: Mean cost per component
- `Overall/Cost_Components/{type}_Std`: Std dev per component

### Cost Components
- **Holding cost**: Cost of maintaining inventory
- **Delivery cost**: Cost of deliveries
- **Order cost**: Cost of placing orders
- **Process cost**: Cost of processing
- **Shortage cost**: Cost of inventory shortage

## TensorBoard Visualization

View results with TensorBoard:

```bash
# From the project root directory
tensorboard --logdir=./Tensorboard_logs/Few_Shot_Eval

# Or view all logs
tensorboard --logdir=./Tensorboard_logs
```

Navigate to `http://localhost:6006` in your browser.

## Integration with Main Training Pipeline

The `few_shot_learning.py` module is **standalone** and doesn't modify existing code:
- ✓ Uses `MetaEnv` from the main pipeline
- ✓ Loads models saved by training scripts
- ✓ Writes to separate TensorBoard directory
- ✓ No changes to `AIIS_META`, `envs`, or other core modules

## Advanced Features

### Adaptation Strategy
The current implementation includes a placeholder for task-specific adaptation:
```python
adapted_params = evaluator.adapt_policy(
    support_episodes=support_episodes,
    inner_lr=0.01,
    num_adaptation_steps=1
)
```

To implement full gradient-based adaptation, integrate with the ProMP/MAML algorithms from `AIIS_META.Algos`.

### Custom Episode Parameters
```python
# Run episode with custom parameters
episode_data = evaluator.run_episode(
    task=task,
    params=custom_params,           # Use custom parameters
    max_steps=100,                   # Custom max steps
    deterministic=True               # Deterministic action selection
)
```

## Troubleshooting

### Model Loading Issues
- Verify `model_path` exists and points to correct checkpoint
- Ensure `model_params` (Layers, num_task, learn_std) match training config
- Check that device (CPU/GPU) has sufficient memory

### Environment Issues
- Ensure environment config files (`config_SimPy.py`, `config_RL.py`) are properly set up
- Verify `MAT_COUNT` and `SIM_TIME` match your scenario

### TensorBoard Not Showing
- Check that log directory is created: `Tensorboard_logs/Few_Shot_Eval/`
- Restart TensorBoard to refresh
- Verify `writer.flush()` is called after logging

## Example: Complete Workflow

```python
import torch
from experiment.few_shot_learning import main

# Configuration
eval_config = {
    'model_path': 'Saved_Model/Train_1/saved_model',
    'model_params': {
        'Layers': [64, 64, 64],
        'num_task': 5,
        'learn_std': True
    },
    'num_test_tasks': 10,
    'num_support_episodes': 5,
    'num_query_episodes': 10,
    'demand_ranges': [(5, 10), (10, 15), (15, 20)],
    'leadtime_ranges': [(1, 2), (2, 3), (3, 5)],
    'device': torch.device('cuda')
}

# Run evaluation
results = main(eval_config)

# Print results
print(f"Average Cost: {results['overall_avg_cost']:.2f}")
print(f"Std Dev: {results['overall_std_cost']:.2f}")
print(f"Test Success - {results['num_test_tasks']} tasks completed")
```

## Notes

- The module preserves original code - no modifications to core algorithms
- Each evaluation creates a new TensorBoard log directory with timestamp
- Support episodes are used for potential future adaptation (currently placeholder)
- Results are suitable for analyzing generalization across different demand/leadtime distributions
