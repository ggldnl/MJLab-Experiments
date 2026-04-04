"""Crawler constants."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets


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

# Leg geometry/site names used by the env config
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