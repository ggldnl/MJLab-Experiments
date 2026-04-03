from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from crawler.velocity.terrains import training_terrain_cfg
from crawler.velocity.task import crawler_velocity_env_cfg
from crawler.velocity.algorithms import crawler_ppo_cfg

register_mjlab_task(
  task_id="crawler_velocity",
  env_cfg=crawler_velocity_env_cfg(),
  play_env_cfg=crawler_velocity_env_cfg(play=True),
  rl_cfg=crawler_ppo_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)