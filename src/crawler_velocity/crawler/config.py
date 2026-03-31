"""Crawler constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


# MJCF and assets

_HERE = Path(__file__).parent
CRAWLER_DESCRIPTION_PATH: Path = _HERE / "assets" / "crawler.xml"
assert CRAWLER_DESCRIPTION_PATH.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, CRAWLER_DESCRIPTION_PATH.parent / "meshes", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CRAWLER_DESCRIPTION_PATH))
    spec.assets = get_assets(spec.meshdir)
    return spec


# Actuator config

EFFORT_LIMIT = 0.18
ARMATURE = 0.001

NATURAL_FREQ = 20 * 2.0 * 3.1415926535  # 20 Hz
DAMPING_RATIO = 2.0

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

CRAWLER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*coxa", ".*femur", ".*tibia"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

# Keyframes

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.08),
    joint_pos={
        ".*coxa": 0.0,
        ".*femur": -0.25,
        ".*tibia": -1.75,
    },
    joint_vel={".*": 0.0},
)

# Collision config

# NOTE: These must match the actual geom names produced by your URDF.
# Inspect with: [model.geom(i).name for i in range(model.ngeom)]
# Tibia geoms are expected to be the terminal contact surface.
# Adjust the regex below once you've confirmed the actual names.
_tibia_geom_regex = r"^leg_[1-4]_tibia(_collision\d*)?$"

CRAWLER_COLLISION = CollisionCfg(
    geom_names_expr=(".*mesh", _tibia_geom_regex),
    condim=3,
    priority=1,
    friction=(0.6,),
    # Soft contact only on tibia tips for stability; remove if unwanted.
    solimp={_tibia_geom_regex: (0.015, 1, 0.03)},
)

# Articulation config

CRAWLER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(CRAWLER_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)

# Action scale: maps each actuator pattern to a scaled effort/stiffness ratio.
# Exported as CRAWLER_ACTION_SCALE to match the env config import.
CRAWLER_ACTION_SCALE: dict[str, float] = {}
for _a in CRAWLER_ARTICULATION.actuators:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    _e = _a.effort_limit
    _s = _a.stiffness
    assert _e is not None
    for _n in _a.target_names_expr:
        CRAWLER_ACTION_SCALE[_n] = 0.25 * _e / _s


# Leg geometry/site names used by the env config.
# TODO it would be better to define a foot in the model, so that we avoid
#  unintended collisions with the rest of the tibia
CRAWLER_FOOT_GEOM_NAMES = (
    "leg_1_tibia_geom",
    "leg_2_tibia_geom",
    "leg_3_tibia_geom",
    "leg_4_tibia_geom",
)

CRAWLER_FOOT_SITE_NAMES = (
    "leg_1_foot",
    "leg_2_foot",
    "leg_3_foot",
    "leg_4_foot",
)

CRAWLER_BASE_NAME = "base"

def get_crawler_robot_cfg() -> EntityCfg:
    """Get a fresh Crawler robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(CRAWLER_COLLISION,),
        spec_fn=get_spec,
        articulation=CRAWLER_ARTICULATION,
    )


if __name__ == "__main__":

    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_crawler_robot_cfg())

    # Useful for debugging geom/site names:
    model = robot.spec.compile()
    print("Geom names:", [model.geom(i).name for i in range(model.ngeom)])
    print("Site names:", [model.site(i).name for i in range(model.nsite)])

    viewer.launch(model)