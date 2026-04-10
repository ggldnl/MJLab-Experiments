# MJLab Experiments

A collection of experiments built with MJLab, focused on legged robotics.

## 🧪 Current Experiments

### Crawler – Velocity Tracking

This experiment implements velocity tracking for a quadruped robot named 'Crawler'. 
The robot follows a crawler-type morphology with a coxa–femur–tibia leg configuration.

The objective is to train a control policy that tracks target linear velocities while maintaining base stability and efficient gait patterns.

Below is a sample rollout after training:

<p align="center">
  <img src="media/crawler_velocity_tracking.gif" alt="Results" width=100%>
</p>

## 🚀️ Deploy

Visualize the robot in its init position:
```bash
python play.py <task_id> --mode none
```

Visualize the robot with a random policy to check if everything works as expected:
```bash
python play.py <task_id> --mode random
```

Start training:
```bash
python train.py <task_id> --env.scene.num-envs 2048
```

Observe the policy after trainig:
```bash
python play.py <task_id> --mode policy
```
The `play` script will automatically search for the latest checkpoint of the latest experiment
available for that particular task on the folder created by the `train` script. If you want to 
check out a particular checkpoint, simply use the `--checkpoint` flag followed by the path 
to the checkpoint.

## ⭐ Support

If you find this repository useful, consider giving it a star. It helps visibility and supports further development.
