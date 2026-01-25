# AIIS_META_RL

## 📌 Overview

This project is based on the original implementation of **[ProMP (Proximal Meta-Policy Search)](https://promp.readthedocs.io/en/latest/index.html)** and has been updated with modern deep learning libraries (PyTorch, TensorBoard, etc.) for meta-reinforcement learning.

### Key Features
- **Meta-Reinforcement Learning**: ProMP algorithm implementation
- **Modern Library Support**: Compatible with latest libraries including PyTorch 2.x and TensorBoard
- **Inventory Management Simulation**: Uses [SimPy-based Inventory Management Simulator](https://github.com/Ha-An/SimPy_IMS) as sample environment

---

## 🚀 Quick Start

### 1. Virtual Environment Setup
```bash
# Create Conda virtual environment
conda create -n aiis_meta_rl python=3.9
conda activate aiis_meta_rl

# Install required libraries
pip install -r requirements.txt
```

### 2. Run the Code
```bash
python main.py
```

### 3. View Results with TensorBoard
```bash
# Run in a new terminal
tensorboard --logdir=./Tensorboard_logs
```
Then access `http://localhost:6006` in your web browser

---

## 📁 Project Structure

```
AIIS_META_RL/
├── main.py                          # Main execution file
├── requirements.txt                 # Library dependencies
├── AIIS_META/                       # Meta RL algorithm implementation
│   ├── Agents/                      # Policy networks (MLP, Gaussian)
│   ├── Algos/                       # Algorithm implementations (ProMP, MAML)
│   ├── Baselines/                   # Gradient estimation baselines
│   ├── Sampler/                     # Sample collection and processing
│   └── Utils/                       # Utility functions
├── envs/                            # Inventory management environment
│   ├── config_SimPy.py              # SimPy simulation configuration
│   ├── config_RL.py                 # RL algorithm configuration
│   ├── environment.py               # Environment class definitions
│   ├── promp_env.py                 # MetaEnv (meta environment)
│   └── scenarios.py                 # Scenario generation
├── Tensorboard_logs/                # Training logs (TensorBoard)
├── Saved_Model/                     # Trained model storage
└── README.md                        # Project documentation
```

---

## 📊 FlowChart

<img width="2048" height="1428" alt="image" src="https://github.com/user-attachments/assets/82df120f-57af-4177-ac45-c75ab4f2a3c4" />

---

## 🔗 References

- **ProMP**: [Proximal Meta-Policy Search Documentation](https://promp.readthedocs.io/en/latest/index.html)
- **Environment**: [SimPy-based Inventory Management Simulator](https://github.com/Ha-An/SimPy_IMS)
- **PyTorch**: [Deep Learning Framework](https://pytorch.org/)
- **SimPy**: [Discrete Event Simulation](https://simpy.readthedocs.io/)

---

## 📝 License

This project is based on ProMP and updates it with modern deep learning libraries.
