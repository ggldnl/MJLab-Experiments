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
    mode: Literal["policy", "random", "none"] = "random"
    checkpoint: Path | None = None
    viewer: Literal["native", "viser"] = "viser"


def _find_latest_checkpoint(log_dir: Path, experiment_name: str) -> Path:
    # Walk logs/<experiment_name>/ and pick the checkpoint with the highest step count
    runs_root = log_dir / experiment_name
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs found for experiment '{experiment_name}' in {log_dir}")

    checkpoints = sorted(runs_root.glob("**/model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {runs_root}")

    # model_<step>.pt — sort numerically by step, not lexicographically
    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=_step)


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
    env_cfg.scene.num_envs = 1024

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)

    action_shape = env.unwrapped.action_space.shape

    if cfg.mode == "policy":
        agent_cfg = load_rl_cfg(task_id)
        checkpoint = cfg.checkpoint or _find_latest_checkpoint(
            Path("logs", "rsl_rl"), agent_cfg.experiment_name
        )
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
    elif cfg.mode == "random":
        policy = lambda obs: 2 * torch.rand(action_shape, device=device) - 1  # random actions in [-1, 1]
    else:
        policy = lambda obs: torch.zeros(action_shape, device=device)  # zero actions, robot holds init pose

    if cfg.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()