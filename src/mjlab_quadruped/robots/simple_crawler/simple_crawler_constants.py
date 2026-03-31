"""Simple Crawler constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# MJCF and assets

_HERE = Path(__file__).parent
SIMPLE_CRAWLER_XML: Path = _HERE / "assets" / "simple_crawler.urdf"
assert SIMPLE_CRAWLER_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, SIMPLE_CRAWLER_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SIMPLE_CRAWLER_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# Actuator config, assuming all the actuators are the same

EFFORT_LIMIT = 0.18

# Random small armature since we don't know the real value.
ARMATURE = 0.001

# PD gains derived from armature, targeting 10 Hz natural frequency.
NATURAL_FREQ = 20 * 2.0 * 3.1415926535  # 20 Hz
DAMPING_RATIO = 5.0

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

SIMPLE_CRAWLER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*coxa", ".*femur", ".*tibia"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)

# Keyframes
# TODO fix these

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.08),
  
  # Same leg configuration copy-pasted
  joint_pos={
    ".*coxa": 0.0,
    ".*femur": 0.0,
    ".*tibia": 0.0
  },
  
  joint_vel={".*": 0.0},
)

"""
# Use this if the legs have a different configuration
# e.g. mirrored about x/y planes
joint_pos={
  "leg_1_coxa": 0.0,
  "leg_1_femur": -0.4,
  "leg_1_tibia": 0.6,
  "leg_2_coxa": 0.0,
  "leg_2_femur": -0.4,
  "leg_2_tibia": 0.6,
  "leg_3_coxa": 0.0,
  "leg_3_femur": -0.4,
  "leg_3_tibia": 0.6,
  "leg_4_coxa": 0.0,
  "leg_4_femur": -0.4,
  "leg_4_tibia": 0.6,
}
"""

# Collision config

_foot_regex = r"^leg_[1-4]_tibia$"

# .*mesh matches all geometry (base, coxa, femur, tibia, ...)
# _foot_regex gets the special soft contact for stability
SIMPLE_CRAWLER_COLLISION = CollisionCfg(
  geom_names_expr=(".*mesh", _foot_regex),
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp={_foot_regex: (0.015, 1, 0.03)},
)

# Final config

SIMPLE_CRAWLER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(SIMPLE_CRAWLER_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,  # joints stop 10% before their MJCF limits (safety margin)
)


def get_simple_crawler_robot_cfg() -> EntityCfg:
  """Get a fresh SimpleCrawler robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(SIMPLE_CRAWLER_COLLISION,),
    spec_fn=get_spec,
    articulation=SIMPLE_CRAWLER_ARTICULATION,
  )


SIMPLE_CRAWLER_SCALE: dict[str, float] = {}
for _a in SIMPLE_CRAWLER_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _d = _a.damping
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    SIMPLE_CRAWLER_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":

  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_simple_crawler_robot_cfg())

  viewer.launch(robot.spec.compile())