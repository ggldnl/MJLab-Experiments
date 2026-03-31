"""RL configuration for Crawler velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def crawler_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Crawler velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name="crawler_velocity_ppo",
    max_iterations=10_000,
  )