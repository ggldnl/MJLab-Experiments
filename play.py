"""
Run a trained policy on a registered mjlab task.

Usage:
    python play.py my-robot-velocity --checkpoint logs/.../checkpoints/model_5000.pt
    python play.py my-robot-velocity --checkpoint ... --video --video-length 1000
"""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from loguru import logger

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class PlayConfig:
    checkpoint: Path
    num_envs: int = 1
    use_cuda: bool = True
    video: bool = False
    video_dir: Path = Path("logs/videos")
    video_length: int = 500


def _load_play_env_cfg(task_id: str):
    """Load the play-specific env config.

    Tries the task registry first (which calls booster_t1_velocity_env_cfg(play=True)
    and disables noise, pushes, and curriculum). Falls back to the train config
    with a warning if the registry does not expose get_task().
    """
    try:
        from mjlab.tasks.registry import get_task
        return get_task(task_id).play_env_cfg
    except (ImportError, AttributeError):
        logger.warning("Registry does not expose get_task() — using train env config as fallback.")
        return load_env_cfg(task_id)


def main() -> None:
    all_tasks = list_tasks()
    task_id, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(PlayConfig, args=remaining, prog=f"{sys.argv[0]} {task_id}")

    if not cfg.checkpoint.exists():
        logger.error(f"Checkpoint not found: {cfg.checkpoint}")
        sys.exit(1)

    device = "cuda:0" if cfg.use_cuda and torch.cuda.is_available() else "cpu"
    os.environ["MUJOCO_GL"] = "egl"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info(f"Task: {task_id}  |  Device: {device}")
    logger.info(f"Checkpoint: {cfg.checkpoint}")

    env_cfg = _load_play_env_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")

    if cfg.video:
        cfg.video_dir.mkdir(parents=True, exist_ok=True)
        env = VideoRecorder(
            env,
            video_folder=cfg.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Use cfg.checkpoint.parent as the runner log dir so tensorboard artifacts stay tidy
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(cfg.checkpoint.parent), device)
    runner.load(str(cfg.checkpoint))
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    logger.success("Running. Press Ctrl+C to stop.")
    try:
        while True:
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        logger.info("Stopped.")

    env.close()


if __name__ == "__main__":
    main()