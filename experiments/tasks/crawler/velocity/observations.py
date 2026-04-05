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
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity.mdp.observations import (
  foot_air_time,
  foot_contact,
  foot_contact_forces,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


actor_terms = {
  "base_ang_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
  ),
  "projected_gravity": ObservationTermCfg(
    func=projected_gravity,
    noise=Unoise(n_min=-0.05, n_max=0.05),
  ),
  "joint_pos": ObservationTermCfg(
    func=joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
  ),
  "joint_vel": ObservationTermCfg(
    func=joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
  ),
  "actions": ObservationTermCfg(
    func=last_action
  ),
  "command": ObservationTermCfg(
    func=generated_commands,
    params={"command_name": "twist"},
  ),
}

# Critic takes all the terms of the actor + observations from environment e.g. foot height, ...

critic_terms = {
  **actor_terms,
  "base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
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
