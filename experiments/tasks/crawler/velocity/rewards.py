"""
A flat dictionary of RewardTermCfg entries, each with a func, a scalar weight, and params.
Positive weights encourage behaviors; negative weights penalize them. Reward tuning is
iterative and messy, so having it on a dedicated file could be useful.
"""

import math
import torch

from mjlab.envs.mdp import (
  action_rate_l2,
  joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  body_angular_velocity_penalty,
  feet_air_time,
  feet_slip,
  self_collision_cost,
  soft_landing,
  track_angular_velocity,
  track_linear_velocity,
  feet_clearance,
  feet_swing_height,
  angular_momentum_penalty
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from experiments.robots.crawler.constants import CRAWLER_BASE_NAME, CRAWLER_FOOT_SITE_NAMES


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
) -> torch.Tensor:
  """
  Reward flat base orientation (robot being upright).
  """

  asset = env.scene["robot"]
  base_id = asset.find_bodies(CRAWLER_BASE_NAME)[0]

  body_quat_w = asset.data.body_link_quat_w[:, base_id, :]  # [B, 4]
  gravity_w = asset.data.gravity_vec_w  # [3]

  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
  xy_squared = torch.sum(projected_gravity_b[:, :2] ** 2, dim=1)

  return torch.exp(-xy_squared / std**2)


rewards = {
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=2.0,
    params={
      "command_name": "twist",
      "std": math.sqrt(0.25)
    },
  ),
  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=2.0,
    params={
      "command_name": "twist",
      "std": math.sqrt(0.5)
    },
  ),
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
    },
  ),
  # TODO add pose reward
  "body_ang_vel": RewardTermCfg(
    func=body_angular_velocity_penalty,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=CRAWLER_BASE_NAME)
    },
  ),
  "angular_momentum": RewardTermCfg(
    func=angular_momentum_penalty,
    weight=-0.5,
    params={"sensor_name": "root_angmom"},
  ),
  "dof_pos_limits": RewardTermCfg(
    func=joint_pos_limits,
    weight=-1.0
  ),
  "action_rate_l2": RewardTermCfg(
    func=action_rate_l2,
    weight=-0.1
  ),
  "air_time": RewardTermCfg(
    func=feet_air_time,
    weight=0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "threshold_max": 0.5,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  ),
  "feet_clearance": RewardTermCfg(
    func=feet_clearance,
    weight=-2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=CRAWLER_FOOT_SITE_NAMES),
      "target_height": 0.025,  # cm
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "feet_swing_height": RewardTermCfg(
    func=feet_swing_height,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=CRAWLER_FOOT_SITE_NAMES),
      "target_height": 0.025,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=CRAWLER_FOOT_SITE_NAMES),
    },
  ),
  "soft_landing": RewardTermCfg(
    func=soft_landing,  # Penalize high foot impact forces
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-0.5,
    params={
      "sensor_name": "self_collision",
      "force_threshold": 1.0
    },
  ),
}

"""
# Rationale for std values:
# Running values are ~1.5-2x walking values to accommodate larger motion range.
# Patterns use (?i) for case-insensitive matching.
rewards["pose"].params["std_standing"] = {
  ".*": 0.05
}
rewards["pose"].params["std_walking"] = {
  r"(?i).*_coxa$": 0.15,   # tighter
  r"(?i).*_femur$": 0.35,  # looser
  r"(?i).*_tibia$": 0.4,   # largest variation
}
rewards["pose"].params["std_running"] = {
  r"(?i).*_coxa$": 0.25,
  r"(?i).*_femur$": 0.6,
  r"(?i).*_tibia$": 0.7,
}
"""
