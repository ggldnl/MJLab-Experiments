"""Crawler collisions."""

from mjlab.utils.spec_config import CollisionCfg


# NOTE: These must match the actual geom names in the XML/URDF.
# Inspect with: [model.geom(i).name for i in range(model.ngeom)]
# Tibia geoms are expected to be the terminal contact surface
_tibia_geom_regex = r"^leg_[1-4]_tibia(_geom\d*)?$"

CRAWLER_COLLISIONS = CollisionCfg(
    geom_names_expr=(".*mesh", _tibia_geom_regex),
    condim=3,
    priority=1,
    friction=(0.6,),
    # Soft contact only on tibia tips for stability
    solimp={_tibia_geom_regex: (0.015, 1, 0.03)},
)