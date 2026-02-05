# Few-Shot Learning Evaluation - Quick Reference

## 🚀 Get Started in 30 Seconds

### Minimal Setup
```bash
cd c:\Github\AIIS_META_RL
python experiment/eval_example.py
```

### View Results
```bash
tensorboard --logdir=./Tensorboard_logs/Few_Shot_Eval
# Check at http://localhost:6006
```

---

## 📋 Common Use Cases

### 1️⃣ Evaluate Pre-trained Model (Default)
```bash
python experiment/eval_example.py --config mixed
```
- 10 test tasks
- 5 + 10 episodes per task
- Demand: 5~20, Leadtime: 1~5

### 2️⃣ Test Low-Demand Scenarios Only
```bash
python experiment/eval_example.py --config low_demand
```
- 8 test tasks (demand 3~8)
- Fast execution, expect lower costs

### 3️⃣ Test High-Demand Scenarios Only
```bash
python experiment/eval_example.py --config high_demand
```
- 8 test tasks (demand 18~25)
- Expect higher costs

### 4️⃣ Test Extreme Scenarios
```bash
python experiment/eval_example.py --config extreme
```
- 12 test tasks (demand 2~30)
- Assess worst-case scenarios

### 5️⃣ Run All Configurations Automatically
```bash
python experiment/eval_example.py --config all
```
- Execute all 4 configurations
- Generate comprehensive comparison report

### 6️⃣ Advanced Customization
```bash
python experiment/eval_example.py \
    --config mixed \
    --support-episodes 10 \
    --query-episodes 20
```

---

## 🎯 Use Directly from Python Code

### Simple Evaluation
```python
from experiment.few_shot_learning import main

results = main()
print(f"Average Cost: {results['overall_avg_cost']:.2f}")
```

### Custom Configuration
```python
from experiment.few_shot_learning import FewShotEvaluator
import torch

# 1. Create evaluator
evaluator = FewShotEvaluator(
    model_path='Saved_Model/Train_1/saved_model',
    device=torch.device('cuda')
)

# 2. Load model
evaluator.load_model({
    'Layers': [64, 64, 64],
    'num_task': 5,
    'learn_std': True
})

# 3. Create custom tasks
test_tasks = evaluator.create_custom_task_distributions(
    num_test_tasks=5,
    demand_ranges=[(8, 12), (15, 18)],  # 2 demand levels
    leadtime_ranges=[(1, 3)]             # 1 leadtime range
)

# 4. Run evaluation
results = evaluator.evaluate_on_task_distribution(
    test_tasks=test_tasks,
    num_support_episodes=3,
    num_query_episodes=5
)

# 5. Cleanup
evaluator.close()

# 6. Check results
print(f"Average Cost: {results['overall_avg_cost']:.2f}")
print(f"Std Dev: {results['overall_std_cost']:.2f}")
```

---

## 📊 Interpreting Results

### Key Metrics

| Metric | Meaning | Lower is Better |
|--------|---------|-----------------|
| **overall_avg_cost** | Average cost | ✅ Yes |
| **overall_std_cost** | Cost variability | ✅ Yes (Stability) |
| **overall_min_cost** | Minimum cost | ℹ️ Reference |
| **overall_max_cost** | Maximum cost | ❌ Lower is better |

### Cost Components
```
Total Cost = Holding + Delivery + Order + Process + Shortage
```

- **Holding Cost**: Excess inventory (minimize)
- **Shortage Cost**: Unmet demand (minimize)
- **Delivery Cost**: Transportation cost
- **Order Cost**: Ordering cost
- **Process Cost**: Operating cost

---

## 🔍 Output Directory Structure

```
Tensorboard_logs/
└── Few_Shot_Eval/
    ├── events.out.tfevents... (log file)
    └── [Run tensorboard from same path]
```

---

## ⚙️ Parameter Configuration Guide

### Evaluation Parameters

```python
num_test_tasks=10           # Number of test tasks
                            # ↑ More tasks = ↑ reliability, ↑ time

num_support_episodes=5      # Episodes for adaptation
                            # ↑ More episodes = ↑ adaptation, ↑ time

num_query_episodes=10       # Episodes for evaluation
                            # ↑ More episodes = ↑ reliability, ↑ time
```

### Recommended Settings

| Scenario | Settings | Time |
|----------|----------|------|
| Quick test | tasks=5, support=2, query=3 | ~5 min |
| Standard eval | tasks=10, support=5, query=10 | ~30 min |
| Detailed analysis | tasks=20, support=10, query=15 | ~90 min |

---

## 🛠️ Troubleshooting

### ❌ "Model not found"
```python
# Solution: Verify correct path
model_path = 'Saved_Model/Train_1/saved_model'
# Or
model_path = 'Saved_Model/Train_2/saved_model'
```

### ❌ CUDA Out of Memory
```python
# Solution 1: Reduce number of episodes
--query-episodes 5

# Solution 2: Use CPU
device=torch.device('cpu')
```

### ❌ TensorBoard logs not showing
```bash
# Solution 1: Terminate and restart process
# Solution 2: Verify log directory
ls -la Tensorboard_logs/Few_Shot_Eval/

# Solution 3: Use alternate port
tensorboard --logdir=Tensorboard_logs --port 6007
```

---

## 📝 Example Commands

### Basic Evaluation
```bash
# Run with default settings
python experiment/eval_example.py

# Run with mixed config (explicit)
python experiment/eval_example.py --config mixed
```

### Quick Test
```bash
# Test with minimal parameters
python experiment/eval_example.py \
    --support-episodes 2 \
    --query-episodes 3
```

### Detailed Evaluation
```bash
# Run many episodes per task
python experiment/eval_example.py \
    --config mixed \
    --support-episodes 10 \
    --query-episodes 20
```

### Extreme Scenarios
```bash
python experiment/eval_example.py --config extreme
```

### Run and Compare All
```bash
python experiment/eval_example.py --config all
```

---

## 📈 Using TensorBoard

### Check Scalar Values
```
Scalars tab → Select Overall/Avg_Query_Cost
```

### Check Cost Distribution
```
Distributions tab → View Overall/Cost_Components
```

### Compare Per-Task Metrics
```
Scalars tab → Select Task_* to compare all tasks
```

---

## 🎓 Concept Explanation

### Few-Shot Learning Flow
```
1. Support Phase
   └─ Collect few episodes on new task

2. Adaptation Phase
   └─ Update policy using collected data

3. Query Phase
   └─ Evaluate adapted policy performance
```

### Generalization Evaluation
```
Unseen demand/leadtime distributions from training
         ↓
Create new test tasks (test_tasks)
         ↓
Measure few-shot adaptation performance
         ↓
Assess meta-learning capability
```

---

## 🎁 Bonus: Custom Task Definition

```python
from experiment.few_shot_learning import FewShotEvaluator

evaluator = FewShotEvaluator('Saved_Model/Train_1/saved_model')
evaluator.load_model({'Layers': [64, 64, 64], 'num_task': 5, 'learn_std': True})

# Fully custom task
custom_task = {
    'DEMAND': {'Dist_Type': 'UNIFORM', 'min': 12, 'max': 15},
    'LEADTIME': [
        {'Dist_Type': 'UNIFORM', 'min': 2, 'max': 4},
        {'Dist_Type': 'UNIFORM', 'min': 1, 'max': 3},
        # ... (MAT_COUNT times)
    ],
    'task_id': 'my_custom_task'
}

# Test single episode
episode = evaluator.run_episode(custom_task, deterministic=True)
print(f"Total cost: {episode['total_cost']}")

evaluator.close()
```

---

## 📞 Help

```bash
python experiment/eval_example.py --help
```

Output:
```
usage: eval_example.py [-h] [--config {low_demand,high_demand,mixed,extreme,all}]
                       [--support-episodes SUPPORT_EPISODES]
                       [--query-episodes QUERY_EPISODES]
                       [--model-path MODEL_PATH]

Few-Shot Learning Evaluation Examples

optional arguments:
  -h, --help            show this help message and exit
  --config {low_demand,high_demand,mixed,extreme,all}
                        Configuration to run (default: mixed)
  --support-episodes SUPPORT_EPISODES
                        Number of support episodes per task (default: 5)
  --query-episodes QUERY_EPISODES
                        Number of query episodes per task (default: 10)
  --model-path MODEL_PATH
                        Path to trained model (default: Saved_Model/Train_1/saved_model)
```

---

**Last Updated**: 2025-01-26  
**Status**: ✅ Ready to Use
