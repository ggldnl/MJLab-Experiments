"""Crawler actuators and articulations."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg


EFFORT_LIMIT = 0.18
ARMATURE = 0.001

HZ = 20
NATURAL_FREQ = HZ * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

CRAWLER_ACTUATORS = BuiltinPositionActuatorCfg(
    target_names_expr=(".*coxa", ".*femur", ".*tibia"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

CRAWLER_ARTICULATIONS = EntityArticulationInfoCfg(
    actuators=(CRAWLER_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)

# Action scale: maps each actuator pattern to a scaled effort/stiffness ratio
# TODO check this
CRAWLER_ACTION_SCALES: dict[str, float] = {}
for _a in CRAWLER_ARTICULATIONS.actuators:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    _e = _a.effort_limit
    _s = _a.stiffness
    assert _e is not None
    for _n in _a.target_names_expr:
        CRAWLER_ACTION_SCALES[_n] = 0.25 * _e / _s