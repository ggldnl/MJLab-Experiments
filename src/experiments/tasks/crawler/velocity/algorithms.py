"""
Defines the learning algorithm. Here we can set PPO hyperparameters
(learning rate, clip range, GAE lambda, mini-batches, etc.) and the
neural network architecture (hidden layers, activation functions).
It's fully decoupled from the environment — we could swap in SAC
or any other algorithm without touching anything else.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def crawler_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Crawler velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
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