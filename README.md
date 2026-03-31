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

`Train` → `Play` → `Sim2Rea`

- Train: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- Play: Replay trained policies to verify expected behavior.
- Sim2Real: Deploy trained policies to physical robots for real-world execution.

## 🚀 Deploy

TODO

```bash
python scripts/train.py --robot <robot> --env <env> --policy <policy> --env.scene.num-envs=4096
python mjlab_experiments/scripts/train.py crawler-velocity-tracking --env.scene.num-envs 4096
```

```bash
python scripts/play.py --robot <robot> --env <env> --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

## 📂 Registry



## ⚠️ TODO

- [ ] Add robot_constants path to registry