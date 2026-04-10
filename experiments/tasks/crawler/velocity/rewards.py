"""
A flat dictionary of RewardTermCfg entries, each with a func, a scalar weight, and params.
Positive weights encourage behaviors; negative weights penalize them. Reward tuning is
iterative and messy, so having it on a dedicated file could be useful.

The general rule is: no penalty term should ever produce a per-episode magnitude larger
than the primary reward term's weight. The primary reward is track_linear_velocity
at weight 4.0. Any penalty that routinely produces -10, -40 etc. will always win
the gradient competition.
"""

import math
import torch

from mjlab.envs.mdp import (
  action_rate_l2,
  joint_vel_l2,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  feet_air_time,
  feet_slip,
  self_collision_cost,
  track_angular_velocity,
  track_linear_velocity,
  is_terminated,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from experiments.robots.crawler.constants import CRAWLER_BASE_NAME, CRAWLER_FOOT_SITE_NAMES, CRAWLER_LEG_DIAGONAL_PAIRS


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", joint_names=".*")


def trot_stability(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
  mode: str = "two_legs"
) -> torch.Tensor:
  """
  Reward trot stability. Two modes can be used to ensure stability:
  - diagonal leg pairs being in the same contact phase (both grounded or both airborne) results
    in a highly dynamic gait which is often unfeasible in reality
  - only one leg at a time could swing, the other three stay on the ground and propel the
    body forward
  """

  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found.reshape(env.num_envs, -1).float()  # [B, num_feet]

  reward = torch.zeros(env.num_envs, device=env.device)
  if mode == "two_legs":
    for i, j in CRAWLER_LEG_DIAGONAL_PAIRS:
      # 1.0 when both grounded or both airborne, 0.0 when mismatched
      same_phase = 1.0 - torch.abs(found[:, i] - found[:, j])
      reward += same_phase
    reward /= len(CRAWLER_LEG_DIAGONAL_PAIRS)  # normalize to [0, 1]
  elif mode == "three_legs":
    num_grounded = torch.sum(found, dim=1)
    reward = 1.0 - torch.abs(num_grounded - 3.0) / 3.0
    reward = torch.clamp(reward, min=0.0)

  command = env.command_manager.get_command(command_name)
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  scale = (total_command > command_threshold).float()

  return reward * scale


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
) -> torch.Tensor:
  """
  Reward flat base orientation (robot being upright) measuring the robot's current tilt angle.
  It projects gravity into the body frame: if the robot is perfectly upright, gravity points
  straight down in body frame and the x/y components are zero. If the robot is tilted, those
  components grow. It's a position-domain signal.
  """

  asset = env.scene["robot"]
  base_id = asset.find_bodies(CRAWLER_BASE_NAME)[0]

  body_quat_w = asset.data.body_link_quat_w[:, base_id, :]  # [B, 4]
  gravity_w = asset.data.gravity_vec_w  # [3]

  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
  xy_squared = torch.sum(projected_gravity_b[:, :2] ** 2, dim=1)

  return torch.exp(-xy_squared / std**2)

def base_height(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float
) -> torch.Tensor:
  """
  Reward staying close to a target base height.
  Uses a Gaussian kernel so small deviations are tolerated but large ones are penalized.
  Adjust target_height to match your robot's nominal standing height.
  """

  asset = env.scene["robot"]
  base_id = asset.find_bodies(CRAWLER_BASE_NAME)[0]

  height = asset.data.body_link_pos_w[:, base_id, 2].squeeze(-1)  # Z coordinate [B]
  height_error_sq = (height - target_height) ** 2

  return torch.exp(-height_error_sq / std ** 2)

def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  std: float = 0.05,            # meters, not rad/s — tighter now
  window_steps: int = 20,       # ~0.2s rolling window
) -> torch.Tensor:
  # Lazily init short-window anchor for reward
  if not hasattr(env, "_stand_still_anchor"):
    asset = env.scene["robot"]
    env._stand_still_anchor = asset.data.root_link_pos_w[:, :2].clone()
    env._stand_still_anchor_step = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  asset = env.scene["robot"]
  current_pos = asset.data.root_link_pos_w[:, :2]
  current_step = env.episode_length_buf

  new_episode = current_step <= 1
  env._stand_still_anchor[new_episode] = current_pos[new_episode]
  env._stand_still_anchor_step[new_episode] = 0

  refresh = (current_step - env._stand_still_anchor_step) >= window_steps
  env._stand_still_anchor[refresh] = current_pos[refresh]
  env._stand_still_anchor_step[refresh] = current_step[refresh]

  net_displacement = torch.norm(current_pos - env._stand_still_anchor, dim=1)
  stillness = torch.exp(-net_displacement / std)  # near 1.0 when not actually moving

  command = env.command_manager.get_command(command_name)
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  scale = (total_command > command_threshold).float()

  return stillness * scale


def nonfeet_ground_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "nonfeet_ground_contact",
) -> torch.Tensor:
  """
  Penalize any contact between non-foot geoms and the terrain.
  """

  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found.reshape(env.num_envs, -1)  # [B, N]
  return found.float().sum(dim=1)


def base_stability(
  env: ManagerBasedRlEnv,
  std: float = 0.5,
) -> torch.Tensor:
  """
  Penalize roll and pitch angular velocity of the base during locomotion.
  This targets dynamic wobble and tilting while walking, which the static
  orientation term (posture) cannot capture: a robot can be momentarily
  upright but still oscillating heavily.
  Uses a Gaussian so small wobble is tolerated and only large rates are
  penalized strongly.
  It's a velocity-domain signal.
  """

  asset = env.scene["robot"]

  # root_link_vel_w is [B, 6]: [:3] linear, [3:] angular in world frame
  # indices 3 and 4 are roll rate and pitch rate respectively
  roll_pitch_vel = asset.data.root_link_vel_w[:, 3:5]  # [B, 2]
  wobble_sq = torch.sum(roll_pitch_vel ** 2, dim=1)  # [B]

  return torch.exp(-wobble_sq / std**2)  # 1.0 when still, 0.0 at large wobble


rewards = {

  # Positive signals: define what the robot should do

  # Phase 1: robot must learn to move and explore different strategies

  # Primary task: velocity tracking.
  # Weights are high and stay high throughout training.
  # std=0.25 m/s for linear: at command=0.1 m/s a stationary robot gets
  # exp(-0.04/0.0625)=0.53, giving a meaningful gradient toward motion.
  # std=0.2 rad/s for angular: same reasoning.
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=5.0,
    params={
      "command_name": "twist",
      "std": 0.25,
    },
  ),
  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=2.0,  # lower than linear, yaw is secondary
    params={
      "command_name": "twist",
      "std": 0.25,
    },
  ),

  # Penalizes non-foot body parts touching the terrain.
  "nonfeet_ground_contact": RewardTermCfg(
    func=nonfeet_ground_contact,
    weight=-0.5,
    params={
      "sensor_name": "nonfeet_ground_contact",
    },
  ),

  # Discourages standing still when commanded to move.
  "stand_still": RewardTermCfg(
    func=stand_still,
    weight=-3.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.05,
      "std": 0.2,
    },
  ),

  # With positive weight, incentivizes foot lifting, exactly what we want during gait
  "air_time": RewardTermCfg(
    func=feet_air_time,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "threshold_max": 0.3,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),

  # Hard termination penalty.
  "is_terminated": RewardTermCfg(
    func=is_terminated,
    weight=-200.0,
  ),

  # Everything else starts from 0 and gets enabled by the curriculum
  # Phase 1: Robot must move (only velocity + termination active)
  # Phase 2: Robot must move well (gait quality added)
  # Phase 3: Robot must move well and look good (posture added)
  # Phase 4: Robot must move well, look good, and be efficient (smoothness added)

  # Posture: upright orientation.
  # std=0.5 rad (~28 deg): tolerates normal walking lean.
  # Weight starts at 0, introduced by curriculum once the robot walks.
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=0.0,
    params={
      "std": 0.5,
    },
  ),

  # Posture: base height.
  # std=0.025 m (25 mm): tolerates terrain bounce, penalizes collapse.
  # Weight starts at 0, introduced by curriculum once the robot walks.
  "base_height": RewardTermCfg(
    func=base_height,
    weight=0.0,
    params={
      "target_height": 0.035,  # 3/4 cm from the ground
      "std": 0.025,
    },
  ),

  # Posture: dynamic base stability (roll/pitch rate).
  # std=0.5 rad/s: tolerates smooth gait, penalizes stumbling.
  # Weight starts at 0, introduced by curriculum after posture is established.
  # This is a velocity-domain signal and only penalizes dynamical wobbling.
  "base_stability": RewardTermCfg(
    func=base_stability,
    weight=0.0,
    params={
      "std": 0.5,  # tolerates ~0.5 rad/s roll/pitch rate before significant penalty
    },
  ),

  # Penalizes foot sliding.
  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=CRAWLER_FOOT_SITE_NAMES),
    },
  ),

  # Discourages jerky joint commands.
  "action_rate_l2": RewardTermCfg(
    func=action_rate_l2,
    weight=-0.05,
  ),

  # Penalize fast joint motion. Weight must be small enough that walking-speed
  # joint velocities don't swamp the positive tracking signal.
  # At 12 joints with typical walking vel ~2 rad/s each, raw L2 ~ 12*(2²) = 48.
  # At -0.1 that's already -4.8 per step, which over an episode completely
  # dominates the +4.0 tracking reward. For this reason we keep this term
  # very small compared to others.
  "joint_vel_l2": RewardTermCfg(
    func=joint_vel_l2,
    weight=0.0,
  ),

  # Penalizes self-collision.
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=0.0,
    params={
      "sensor_name": "self_collision",
      "force_threshold": 2.5,
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