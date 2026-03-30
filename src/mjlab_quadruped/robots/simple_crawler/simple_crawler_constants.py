"""ANYbotics ANYmal C constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

# MJCF and assets

_HERE = Path(__file__).parent
SIMPLE_CRAWLER_XML: Path = _HERE / "assets" / "simple_crawler.xml"
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
  pos=(0.0, 0.0, 0.05),
  joint_pos={
    ".*HAA": 0.0,
    "LF_HFE": 0.4,
    "RF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RH_HFE": -0.4,
    "LF_KFE": -0.8,
    "RF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RH_KFE": 0.8,
  },
  joint_vel={".*": 0.0},
)

# Collision config
# TODO fix these

_foot_regex = r"^[LR][FH]_foot$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", _foot_regex),
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
    collisions=(FULL_COLLISION,),
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