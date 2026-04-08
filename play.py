import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import experiments.tasks  # noqa: F401


@dataclass(frozen=True)
class PlayConfig:
    mode: Literal["policy", "random", "none"] = "random"
    checkpoint: Path | None = None
    viewer: Literal["native", "viser"] = "viser"
    video: bool = False
    video_length: int = 500
    video_height: int | None = None
    video_width: int | None = None
    video_dir: Path = Path("videos", "rsl_rl")
    logs_dir: Path = Path("logs", "rsl_rl")


def _find_latest_checkpoint(log_dir: Path, experiment_name: str) -> Path:
    runs_root = log_dir / experiment_name
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs found for experiment '{experiment_name}' in {log_dir}")

    # Get all run directories
    run_dirs = [d for d in runs_root.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")

    # Select latest run by folder name (timestamp string sorts correctly)
    latest_run = max(run_dirs, key=lambda d: d.name)

    # Find checkpoints only inside latest run
    checkpoints = list(latest_run.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {latest_run}")

    def _step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    checkpoint = max(checkpoints, key=_step)
    print(f"Selected checkpoint: {checkpoint}")
    return checkpoint


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
    env_cfg.scene.num_envs = 64

    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (cfg.mode == "policy" and cfg.video) else None
    if cfg.video and cfg.mode != "policy":
        print("[WARN] Video recording is only supported in policy mode.")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if cfg.mode == "policy" and cfg.video:
        print("[INFO] Recording video during play")
        env = VideoRecorder(
            env,
            video_folder=cfg.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)  # always before runner

    action_shape = env.unwrapped.action_space.shape

    if cfg.mode == "policy":
        agent_cfg = load_rl_cfg(task_id)
        checkpoint = cfg.checkpoint or _find_latest_checkpoint(
            cfg.logs_dir, agent_cfg.experiment_name
        )
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)  # env already has RslRlVecEnvWrapper
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
    elif cfg.mode == "random":
        policy = lambda obs: 2 * torch.rand(action_shape, device=device) - 1
    else:
        policy = lambda obs: torch.zeros(action_shape, device=device)

    if cfg.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()