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
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from experiments.utils.video_logger import RunnerWithEvalVideo, EvalVideoLogger

import experiments.tasks  # noqa: F401 - triggers _auto_import_submodules, populates registry


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    project: str = "mjlab"
    entity: str | None = None
    group: str | None = None  # defaults to task_id when None
    log_interval: int = 1
    eval_video_interval: int = 500
    eval_video_length: int = 200


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


def _init_wandb(task_id: str, cfg: TrainConfig, run_dir: Path) -> wandb.sdk.wandb_run.Run | None:
    if not cfg.wandb.enabled:
        return None
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group or task_id,
        name=run_dir.name,
        dir=str(run_dir),
        config={"task": task_id, "seed": cfg.seed, **asdict(cfg.env), **asdict(cfg.agent)},
        sync_tensorboard=True,  # RSL-RL writes TB summaries; wandb syncs them automatically
    )


def run_train(task_id: str, cfg: TrainConfig) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    configure_torch_backends()

    env_cfg = cfg.env
    agent_cfg = cfg.agent
    env_cfg.seed = cfg.seed
    agent_cfg.seed = cfg.seed

    if cfg.enable_nan_guard:
        env_cfg.sim.nan_guard.enabled = True

    run_dir = _make_run_dir(cfg.log_dir, agent_cfg.experiment_name)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(run_dir / "train.log", level="DEBUG", rotation="50 MB")

    wandb_run = _init_wandb(task_id, cfg, run_dir)

    logger.info(f"Task:         {task_id}")
    logger.info(f"Seed:         {cfg.seed}")
    logger.info(f"Num envs:     {env_cfg.scene.num_envs}")
    logger.info(f"Device:       {device}")
    logger.info(f"Run dir:      {run_dir}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Use eval-video runner subclass when wandb + a non-zero interval are both enabled
    use_eval_video = cfg.wandb.enabled and cfg.wandb.eval_video_interval > 0
    runner_cls = load_runner_cls(task_id) or (RunnerWithEvalVideo if use_eval_video else MjlabOnPolicyRunner)
    runner = runner_cls(env, asdict(agent_cfg), str(run_dir), device)
    runner.add_git_repo_to_log(__file__)

    if use_eval_video and isinstance(runner, RunnerWithEvalVideo):
        runner.eval_video_logger = EvalVideoLogger(
            env_cfg=env_cfg,
            device=device,
            video_length=cfg.wandb.eval_video_length,
            interval=cfg.wandb.eval_video_interval,
        )

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
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
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
    logger.success("Training complete.")


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
    run_train(task_id, cfg)


if __name__ == "__main__":
    main()