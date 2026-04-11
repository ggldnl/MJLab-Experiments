"""
Defines observation groups, typically one for the actor (noisy, sim-to-real-safe sensors only)
and one for the critic (privileged information that's only available in sim, like true linear
velocity or ground truth foot contact forces). The critic group usually starts with the
actor_terms and adds privileged terms on top.
"""

from mjlab.envs.mdp.observations import (
  builtin_sensor,
  generated_commands,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
  height_scan
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity.mdp.observations import (
  foot_air_time,
  foot_contact,
  foot_contact_forces,
  foot_height
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
import torch

from experiments.robots.crawler.constants import CRAWLER_FOOT_GEOM_NAMES
from experiments.robots.crawler.sensors import TERRAIN_SCAN


# Number of past frames to stack for proprioceptive terms.
# At 100 Hz with decimation=4, we have a 25 Hz policy,
# so 10 frames = 400ms window, enough to observe
# roughly one full stride cycle
_PROPRIOCEPTIVE_HISTORY = 10

# Four legs, diagonal pairs offset by pi so legs 0/1 and 2/3 are anti-phase.
# This directly encodes a trot pattern into the input space.
_LEG_PHASE_OFFSETS = torch.tensor([0.0, torch.pi, 0.0, torch.pi])

def gait_phase_clock(
  env: ManagerBasedRlEnv,
  frequency: float = 1.5,  # Hz, roughly one stride per second
) -> torch.Tensor:
  # Advance a per-env phase counter lazily
  if not hasattr(env, "_phase_clock"):
    env._phase_clock = torch.zeros(env.num_envs, device=env.device)

  # Reset on new episode
  env._phase_clock[env.episode_length_buf <= 1] = 0.0

  dt = env.physics_dt * env.cfg.decimation  # policy dt
  env._phase_clock += 2.0 * torch.pi * frequency * dt

  # [B, 4] per-leg phases: diagonal pair offset by pi
  offsets = _LEG_PHASE_OFFSETS.to(env.device)
  phases = env._phase_clock.unsqueeze(1) + offsets.unsqueeze(0)  # [B, 4]

  # sin and cos together encode both phase and rate without discontinuity
  return torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)  # [B, 8]


actor_proprioceptive_terms = {
  "base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_lin_vel"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
    history_length=_PROPRIOCEPTIVE_HISTORY,
  ),
  "base_ang_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_ang_vel"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
    history_length=_PROPRIOCEPTIVE_HISTORY,
  ),
  # No history: gravity vector changes only when the robot tips, which is a
  # slow signal; a single frame is sufficient
  "projected_gravity": ObservationTermCfg(
    func=projected_gravity,
    noise=Unoise(n_min=-0.05, n_max=0.05),
  ),
  # Having joint positions across frames encode leg phase implicitly:
  # the policy can infer whether each leg is mid-swing or mid-stance without
  # a separate phase signal
  "joint_pos": ObservationTermCfg(
    func=joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    history_length=_PROPRIOCEPTIVE_HISTORY,
  ),
  "joint_vel": ObservationTermCfg(
    func=joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
    history_length=_PROPRIOCEPTIVE_HISTORY,
  ),
  # Past actions let the policy detect its own oscillation pattern
  # and self-correct without needing an explicit oscillation penalty
  "actions": ObservationTermCfg(
    func=last_action,
    history_length=_PROPRIOCEPTIVE_HISTORY,
  ),
  # No history: command is constant within a resampling window, stacking it`
  # would just repeat the same vector n times and waste input dimensions
  "command": ObservationTermCfg(
    func=generated_commands,
    params={"command_name": "twist"},
  ),
  # Tells the policy when to move
  "gait_phase": ObservationTermCfg(
    func=gait_phase_clock,
    params={"frequency": 1.5},
  ),
}

actor_exteroceptive_terms = {
  # No history: terrain geometry changes slowly relative to the stride cycle;
  # the scan is also the largest term (many grid points), stacking it would
  # inflate the input dimension significantly with no benefit
  "height_scan": ObservationTermCfg(
    func=height_scan,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    scale=1 / TERRAIN_SCAN.max_distance,
  ),
}

actor_terms = {
  **actor_proprioceptive_terms,
  **actor_exteroceptive_terms
}

# Critic takes all the terms of the actor + observations from environment e.g. foot height, ...
# Clean ground truth versions of the noisy actor terms. No history needed
# since the critic sees exact state and doesn't need to infer trends

critic_terms = {
  **actor_terms,
  "true_base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_lin_vel"},
    # Critic sees clean signal
  ),
  "true_base_ang_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_ang_vel"},
  ),
  "true_joint_pos": ObservationTermCfg(
    func=joint_pos_rel,
  ),
  "true_joint_vel": ObservationTermCfg(
    func=joint_vel_rel,
  ),
  "height_scan": ObservationTermCfg(
    func=height_scan,
    params={"sensor_name": "terrain_scan"},
    scale=1 / TERRAIN_SCAN.max_distance,
  ),
  "feet_contact": ObservationTermCfg(
      func=foot_contact,
      params={"sensor_name": "feet_ground_contact"},
  ),
  "feet_air_time": ObservationTermCfg(
      func=foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
  ),
  "feet_contact_forces": ObservationTermCfg(
      func=foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
  ),
  "feet_height": ObservationTermCfg(
    func=foot_height,
    params={"asset_cfg": SceneEntityCfg("robot", geom_names=CRAWLER_FOOT_GEOM_NAMES)},
  ),
}

observations = {
  "actor": ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}