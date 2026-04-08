"""
Commands, Actions, Curriculum, Terminations
Four tightly related things that define the agent's interface with the task:
- Commands: what goals are sampled and sent to the policy each step
    (e.g. target velocity [vx, vy, ωz])
- Actions: how the policy output maps onto the robot
    (e.g. joint position targets with a scale factor)
- Curriculum: how difficulty ramps over training
    (e.g. expanding velocity ranges at step thresholds)
- Terminations: episode-ending conditions
    (timeout, robot fell over)

These four belong together because they all answer the question:
what is the task contract between environment and policy?
"""

import math
from typing import Dict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

from experiments.robots.crawler.actuators import CRAWLER_ACTION_SCALES


commands: Dict[str, CommandTermCfg] = {
  "twist": UniformVelocityCommandCfg(
    entity_name="robot",
    rel_standing_envs=0.1,
    rel_heading_envs=0.3,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    resampling_time_range=(0.5, 5.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.8, 0.8),  # hard ceiling matches actuator limits
      lin_vel_y=(-0.8, 0.8),
      ang_vel_z=(-0.5, 0.5),
      heading=(-math.pi, math.pi),
    ),
  )
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=CRAWLER_ACTION_SCALES,
    use_default_offset=True,
  )
}

curriculum = {
  "command_vel": CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": (-0.25, 0.25),  # comfortable walking range
          "lin_vel_y": (-0.25, 0.25),
          "ang_vel_z": (-0.2, 0.2),
        },
        {
          "step": 5000 * 24,
          "lin_vel_x": (-0.5, 0.5),  # nominal trot
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.4, 0.4),
        },
        {
          "step": 10000 * 24,
          "lin_vel_x": (-0.8, 0.8),  # aggressive, near physical ceiling
          "lin_vel_y": (-0.8, 0.8),
          "ang_vel_z": (-0.5, 0.5),
        },
      ],
    },
  ),
}


def _get_counter(env: ManagerBasedRlEnv, attr: str) -> torch.Tensor:
  # Lazily initialize a per-env step counter
  if not hasattr(env, attr):
    setattr(env, attr, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
  counter = getattr(env, attr)

  # Reset counter for envs that just started a new episode
  counter[env.episode_length_buf <= 1] = 0
  return counter


def stand_still_termination(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  velocity_threshold: float = 0.05,
  max_still_steps: int = 150,
) -> torch.Tensor:

  # Terminate if the robot hasn't moved for too long while commanded
  counter = _get_counter(env, "_still_termination_counter")

  asset = env.scene["robot"]
  speed = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)

  command = env.command_manager.get_command(command_name)
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  command_active = total_command > command_threshold

  is_still = command_active & (speed < velocity_threshold)
  counter[is_still] += 1
  counter[~is_still] = 0  # reset on any movement

  return counter >= max_still_steps


def poor_velocity_tracking_termination(
  env: ManagerBasedRlEnv,
  command_name: str,
  error_threshold: float = 0.8,
  command_threshold: float = 0.1,
  max_bad_steps: int = 200,
) -> torch.Tensor:
  counter = _get_counter(env, "_tracking_termination_counter")

  asset = env.scene["robot"]
  command = env.command_manager.get_command(command_name)

  # root_link_vel_w is [B, 6]: [:, :3] linear, [:, 3:] angular — world frame
  lin_vel_w = asset.data.root_link_vel_w[:, :2]
  ang_vel_w = asset.data.root_link_vel_w[:, 5]  # yaw rate

  lin_vel_error = torch.norm(command[:, :2], dim=1) - torch.norm(lin_vel_w, dim=1)
  ang_vel_error = torch.abs(command[:, 2] - ang_vel_w)
  total_error = torch.abs(lin_vel_error) + ang_vel_error

  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  command_active = total_command > command_threshold

  tracking_bad = command_active & (total_error > error_threshold)
  counter[tracking_bad] += 1
  counter[~tracking_bad] = 0

  return counter >= max_bad_steps


terminations = {
  "time_out": TerminationTermCfg(
    func=time_out,
    time_out=True,
  ),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "stand_still": TerminationTermCfg(
    func=stand_still_termination,
    params={
      "command_name": "twist",
      "command_threshold": 0.05,
      "velocity_threshold": 0.05,
      "max_still_steps": 300,  # ~3s at 100Hz
    },
  ),
  "poor_tracking": TerminationTermCfg(
    func=poor_velocity_tracking_termination,
    params={
      "command_name": "twist",
      "error_threshold": 0.8,
      "command_threshold": 0.05,
      "max_bad_steps": 300,  # ~3s tolerance before killing the episode
    },
  ),
}
