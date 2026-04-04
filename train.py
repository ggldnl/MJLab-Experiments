"""
Train an RL agent on a registered mjlab task.

Usage:
    python train.py crawler-velocity
    python train.py crawler-velocity --env.scene.num-envs 2048
    python train.py crawler-velocity --env.episode-length-s 30
    python train.py crawler-velocity --agent.learning-rate 3e-4
    python train.py crawler-velocity --wandb.project my-project
    python train.py crawler-velocity --checkpoint logs/.../checkpoints/model_1000.pt
    python train.py crawler-velocity --gpu-ids 0 1
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
import wandb
from loguru import logger

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import experiments.tasks  # noqa: F401 - triggers _auto_import_submodules, populates registry


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    project: str = "mjlab"
    entity: str | None = None
    group: str | None = None  # defaults to task_id when None
    log_interval: int = 1


@dataclass(frozen=True)
class TrainConfig:
    # Populated with registry defaults in _make_train_cfg(); all fields are
    # individually overridable from the CLI (e.g. --env.scene.num-envs 2048).
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_dir: Path = Path("logs/rsl_rl")
    seed: int = 0
    checkpoint: Path | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    gpu_ids: list[int] | Literal["all"] | None = None

    @staticmethod
    def from_task(task_id: str) -> TrainConfig:
        return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def _make_run_dir(log_dir: Path, experiment_name: str) -> Path:
    root = log_dir / experiment_name
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _init_wandb(
    task_id: str, cfg: TrainConfig, run_dir: Path
) -> wandb.sdk.wandb_run.Run | None:
    if not cfg.wandb.enabled:
        return None
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group or task_id,
        name=run_dir.name,
        dir=str(run_dir),
        config={"task": task_id, "seed": cfg.seed, **asdict(cfg.env), **asdict(cfg.agent)},
        sync_tensorboard=True,  # RSL-RL writes TB summaries; this syncs them automatically
    )


def run_train(task_id: str, cfg: TrainConfig, run_dir: Path) -> None:
    """Training body — runs directly for single-GPU, via torchrunx for multi-GPU."""
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        device, rank, seed = "cpu", 0, cfg.seed
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = cfg.seed + local_rank  # per-process diversity

    configure_torch_backends()

    # Apply seed to the configs that came in through CLI (already overridden by the user)
    env_cfg = cfg.env
    agent_cfg = cfg.agent
    env_cfg.seed = seed
    agent_cfg.seed = seed

    if cfg.enable_nan_guard:
        env_cfg.sim.nan_guard.enabled = True

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if rank == 0:
        logger.add(run_dir / "train.log", level="DEBUG", rotation="50 MB")
        wandb_run = _init_wandb(task_id, cfg, run_dir)
        logger.info(f"Task:      {task_id}")
        logger.info(f"Seed:      {seed}  |  Num envs: {env_cfg.scene.num_envs}")
        logger.info(f"Device:    {device}  |  Run dir: {run_dir}")
    else:
        wandb_run = None

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if cfg.video else None,
    )

    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=run_dir / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(run_dir), device)
    runner.add_git_repo_to_log(__file__)

    if rank == 0:
        dump_yaml(run_dir / "params" / "env.yaml", asdict(env_cfg))
        dump_yaml(run_dir / "params" / "agent.yaml", asdict(agent_cfg))

    if cfg.checkpoint is not None:
        if not cfg.checkpoint.exists():
            logger.error(f"Checkpoint not found: {cfg.checkpoint}")
            sys.exit(1)
        logger.info(f"Resuming from: {cfg.checkpoint}")
        runner.load(str(cfg.checkpoint))
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(
            run_dir.parent.parent, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        logger.info(f"Resuming from: {resume_path}")
        runner.load(str(resume_path))

    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    except KeyboardInterrupt:
        if rank == 0:
            logger.warning("Interrupted — saving checkpoint...")
            interrupt_path = run_dir / "checkpoints" / "interrupted.pt"
            interrupt_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                runner.save(str(interrupt_path))
                logger.success(f"Saved: {interrupt_path}")
            except Exception as exc:
                logger.error(f"Failed to save checkpoint: {exc}")

    if wandb_run is not None:
        wandb_run.finish()

    env.close()
    if rank == 0:
        logger.success("Training complete.")


def launch(task_id: str, cfg: TrainConfig) -> None:
    run_dir = _make_run_dir(cfg.log_dir, cfg.agent.experiment_name)

    selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    )
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        run_train(task_id, cfg, run_dir)
        return

    import torchrunx

    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("TORCHRUNX_LOG_DIR", str(run_dir / "torchrunx"))
    logger.info(f"Launching with {num_gpus} GPUs")
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=num_gpus,
        backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, cfg, run_dir)


def main() -> None:
    all_tasks = list_tasks()
    task_id, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    cfg = tyro.cli(
        TrainConfig,
        args=remaining,
        default=TrainConfig.from_task(task_id),  # registry defaults, then CLI overrides
        prog=f"{sys.argv[0]} {task_id}",
    )
    launch(task_id, cfg)


if __name__ == "__main__":
    main()