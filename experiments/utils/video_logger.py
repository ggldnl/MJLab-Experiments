import copy

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from loguru import logger
import numpy as np
import torch
import wandb


class EvalVideoLogger:
    """Runs a short deterministic rollout and logs the result to wandb."""

    def __init__(
        self,
        env_cfg: ManagerBasedRlEnvCfg,
        device: str,
        video_length: int,
        interval: int,
        clip_actions: float | None = None,
    ):
        self.interval = interval
        self._env_cfg = env_cfg
        self._device = device
        self._video_length = video_length
        self._clip_actions = clip_actions

    def maybe_log(self, runner: MjlabOnPolicyRunner, iteration: int) -> None:
        if iteration % self.interval != 0:
            return

        # Deep-copy so we never mutate the live training config
        eval_cfg = copy.deepcopy(self._env_cfg)
        eval_cfg.scene.num_envs = 1

        # Build a separate env in rgb_array mode
        base_env = ManagerBasedRlEnv(cfg=eval_cfg, device=self._device, render_mode="rgb_array")
        eval_env = RslRlVecEnvWrapper(base_env, clip_actions=self._clip_actions)

        policy = runner.get_inference_policy(device=self._device)

        frames: list[np.ndarray] = []
        obs, _ = eval_env.get_observations()

        with torch.no_grad():
            for _ in range(self._video_length):
                actions = policy(obs)
                obs, _, _, _ = eval_env.step(actions)

                # Call render() on the base env, not the wrapper
                frame = base_env.render()  # (H, W, 3) uint8
                if frame is not None:
                    frames.append(frame)

        eval_env.close()

        if not frames:
            logger.warning(f"Eval video at iteration {iteration} produced no frames — skipping.")
            return

        # wandb expects (T, C, H, W)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        wandb.log(
            {"eval/policy": wandb.Video(video_array, fps=30, format="mp4")},
            step=iteration,
            commit=True,
        )
        logger.info(f"Logged eval video at iteration {iteration} ({len(frames)} frames).")


class RunnerWithEvalVideo(MjlabOnPolicyRunner):
    """Thin subclass that fires EvalVideoLogger on every log() call."""

    eval_video_logger: EvalVideoLogger | None = None

    def log(self, locs: dict, *args, **kwargs) -> None:
        super().log(locs, *args, **kwargs)
        if self.eval_video_logger is not None:
            it = locs.get("it", self.current_learning_iteration)
            self.eval_video_logger.maybe_log(self, it)