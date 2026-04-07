# MJLab Experiments

Collection of MJLab experiments.

## 🛠️ Setup

This package is managed with conda. To create an environment:

```bash
conda env create -f environment.yml
conda activate mjlab_crawler
```

## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- Train: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- Play: Replay trained policies to verify expected behavior.
- Sim2Real: Deploy trained policies to physical robots for real-world execution.

## 🚀 Deploy

Visualize the robot:
```bash
python crawler/config.py
```

Start training:
```bash
python utils/train.py crawler_velocity --wandb.project crawler_velocity
```

Observe the policy:
```bash
python src/play.py Mjlab-Velocity-Flat-Crawler --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```