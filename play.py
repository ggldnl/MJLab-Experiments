import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import experiments.tasks  # noqa: F401


@dataclass(frozen=True)
class PlayConfig:
    checkpoint: Path | None = None  # if None, a random policy is used
    viewer: Literal["native", "viser"] = "viser"


def main() -> None:
    all_tasks = list_tasks()
    task_id, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(PlayConfig, args=remaining, prog=f"{sys.argv[0]} {task_id}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)

    if cfg.checkpoint is None:
        action_shape = env.unwrapped.action_space.shape
        policy = lambda obs: 2 * torch.rand(action_shape, device=device) - 1  # random actions in [-1, 1]
    else:
        agent_cfg = load_rl_cfg(task_id)
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(cfg.checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

    if cfg.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()