"""Crawler velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


from crawler_velocity.crawler.config import (
    CRAWLER_ACTION_SCALE,
    CRAWLER_FOOT_SITE_NAMES,
    CRAWLER_FOOT_GEOM_NAMES,
    CRAWLER_BASE_NAME,
    get_crawler_robot_cfg,
)


def crawler_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Crawler rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 50

    cfg.scene.entities = {"robot": get_crawler_robot_cfg()}

    # remove observations that depend on imu or other sensors that don't exist
    if "imu" in cfg.observations["actor"].terms:
        del cfg.observations["actor"].terms["imu"]
    if "imu_lin_vel" in cfg.observations["actor"].terms:
        del cfg.observations["actor"].terms["imu_lin_vel"]
    if "height_scan" in cfg.observations["actor"].terms:
        del cfg.observations["actor"].terms["height_scan"]

    # Set raycast sensor frame to base, if existing
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "base"

    site_names = CRAWLER_FOOT_SITE_NAMES
    geom_names = CRAWLER_FOOT_GEOM_NAMES

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            # Grab all geoms (anything that ends with "_geom")
            pattern=r".*_geom$",
            # Except for the foot geoms.
            exclude=tuple(geom_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = CRAWLER_ACTION_SCALE

    cfg.viewer.body_name = CRAWLER_BASE_NAME
    cfg.viewer.distance = 2.0
    cfg.viewer.elevation = -10.0

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

    # Rewards based on distance of the joints from the reference
    cfg.rewards["pose"].params["std_standing"] = {
        ".*coxa": 0.1,
        ".*femur": 0.1,
        ".*tibia": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        ".*coxa": 0.2,      # tight
        ".*femur": 0.25,    # moderate
        ".*tibia": 0.4,     # loose
    }
    cfg.rewards["pose"].params["std_running"] = {
        ".*coxa": 0.3,
        ".*femur": 0.35,
        ".*tibia": 0.6,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name},
    )

    cmd = cfg.commands["twist"]
    assert isinstance(cmd, UniformVelocityCommandCfg)
    cmd.viz.z_offset = 0.5

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def crawler_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Crawler flat terrain velocity configuration."""
    cfg = crawler_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan (no terrain to scan).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Disable terrain curriculum.
    cfg.curriculum.pop("terrain_levels", None)

    return cfg