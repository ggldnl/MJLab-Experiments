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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, reward_weight
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


# Curriculum
# steps_per_iteration = common_step_counter / iterations
# steps_per_iteration_per_env = steps_per_iteration / num_envs
#
# _STEPS_PER_ITER = _TOTAL_STEPS_AT_ITER_X / _ITER_X  # measure from logs
# _Sx = y * _STEPS_PER_ITER  # at iteration y we trigger something
# _TOTAL_STEPS_AT_ITER_X = 4964352
# _ITER_X = 100
# _STEPS_PER_ITER = _TOTAL_STEPS_AT_ITER_X / _ITER_X = 49643 ~ 50k

_S0 = 0
_S1 = 500    # basic locomotion
_S2 = 2000   # introduce posture
_S3 = 3000   # strengthen posture, expand velocity
_S4 = 4500   # gait quality: stability, smoothness
_S5 = 6000   # full velocity + final polish

curriculum = {

  # Velocity expands in three steps so the policy always has
  # a comfortable margin above what it already mastered
  "command_vel": CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": _S0, "lin_vel_x": (-0.18, 0.18), "lin_vel_y": (-0.18, 0.18), "ang_vel_z": (-0.10, 0.10)},
        {"step": _S1, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-0.15, 0.15)},
        {"step": _S3, "lin_vel_x": (-0.40, 0.40), "lin_vel_y": (-0.40, 0.40), "ang_vel_z": (-0.30, 0.30)},
        {"step": _S5, "lin_vel_x": (-0.50, 0.50), "lin_vel_y": (-0.50, 0.50), "ang_vel_z": (-0.40, 0.40)},
      ],
    },
  ),

  # Phase 1 -> 2: behavioral penalties ramp together with the first
  # velocity expansion, so the harder task comes with stricter rules

  # stand_still: relax gradually but never to zero
  "w_stand_still": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "stand_still",
      "weight_stages": [
        {"step": _S0, "weight": 0.0},
        {"step": _S3, "weight": -1.0},
        {"step": _S4, "weight": -1.5},
      ],
    },
  ),

  "w_nonfeet_ground_contact": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "nonfeet_ground_contact",
      "weight_stages": [
        {"step": _S0, "weight": -0.5},
        {"step": _S1, "weight": -1.0},
        {"step": _S3, "weight": -1.5},
        {"step": _S4, "weight": -2.0},
      ],
    },
  ),

  # Phase 2 -> 3: posture pair introduced together at _S2.
  # Upright and base_height are the same conceptual signal
  # (body pose), so changing them at different steps gains nothing

  # Introduce upright orientation reward once basic locomotion exists.
  # Low weight at first so it shapes without dominating velocity tracking.
  "w_upright": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "upright",
      "weight_stages": [
        {"step": _S0, "weight": 0.0},
        {"step": _S2, "weight": 0.5},
        {"step": _S3, "weight": 1.0},
        {"step": _S4, "weight": 1.5},
      ],
    },
  ),

  # Base height reward
  "w_base_height": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "base_height",
      "weight_stages": [
        {"step": _S0, "weight": 0.0},
        {"step": _S2, "weight": 0.5},
        {"step": _S3, "weight": 1.0},
        {"step": _S4, "weight": 1.5},
      ],
    },
  ),

  # Phase 3 -> 4: gait quality terms.
  # foot_slip starts light at _S2 alongside posture (both concern
  # contact quality), then tightens at _S4 with the stability terms

  "w_foot_slip": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "foot_slip",
      "weight_stages": [
        {"step": _S0, "weight":  0.0},
        {"step": _S2, "weight": -0.2},
        {"step": _S4, "weight": -0.5},
      ],
    },
  ),

  # Introduce base stability only after posture rewards have had time to
  # establish a stable base. Too early and it fights locomotion exploration.
  "w_base_stability": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "base_stability",
      "weight_stages": [
        {"step": _S0, "weight":  0.0},
        {"step": _S4, "weight": -0.3},
        {"step": _S5, "weight": -0.6},
      ],
    },
  ),

  # Phase 4 -> 5: smoothness.
  # action_rate stays flat until _S4 so early exploration
  # is not suppressed; joint_vel only at _S5 as final polish

  # Penalizes the change in action between consecutive steps
  "w_action_rate_l2": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "action_rate_l2",
      "weight_stages": [
        {"step": _S0, "weight": -0.05},
        {"step": _S4, "weight": -0.15},
        {"step": _S5, "weight": -0.30},
      ],
    },
  ),

  "w_joint_vel_l2": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "joint_vel_l2",
      "weight_stages": [
        {"step": _S0, "weight":  0.000},
        {"step": _S5, "weight": -0.003},
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
  command_threshold: float = 0.05,
  displacement_threshold: float = 0.05,  # meters of net travel required
  window_steps: int = 100, # ~1s at 100Hz
  max_still_steps: int = 300,
) -> torch.Tensor:
  counter = _get_counter(env, "_still_termination_counter")

  # Lazily init position buffer: stores root XY at the start of each window
  if not hasattr(env, "_still_term_anchor_pos"):
    asset = env.scene["robot"]
    env._still_term_anchor_pos = asset.data.root_link_pos_w[:, :2].clone()
    env._still_term_anchor_step = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  asset = env.scene["robot"]
  current_pos = asset.data.root_link_pos_w[:, :2]
  current_step = env.episode_length_buf

  # Reset anchor on new episode
  new_episode = current_step <= 1
  env._still_term_anchor_pos[new_episode] = current_pos[new_episode]
  env._still_term_anchor_step[new_episode] = 0

  # Refresh anchor every window_steps
  window_elapsed = current_step - env._still_term_anchor_step
  refresh = window_elapsed >= window_steps
  env._still_term_anchor_pos[refresh] = current_pos[refresh]
  env._still_term_anchor_step[refresh] = current_step[refresh]

  # Net displacement since last anchor
  net_displacement = torch.norm(current_pos - env._still_term_anchor_pos, dim=1)

  command = env.command_manager.get_command(command_name)
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  command_active = total_command > command_threshold

  # Count steps where displacement is insufficient despite command
  is_still = command_active & (net_displacement < displacement_threshold)
  counter[is_still] += 1
  counter[~is_still] = 0

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
    params={"limit_angle": math.radians(90.0)},
  ),
  # "stand_still": TerminationTermCfg(
  #   func=stand_still_termination,
  #   params={
  #     "command_name": "twist",
  #   },
  # ),
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
