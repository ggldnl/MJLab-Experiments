from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from loguru import logger
import numpy as np
import torch
import wandb


class EvalVideoLogger:
    """Runs a short deterministic rollout and logs the result to wandb."""

    def __init__(self, env_cfg: ManagerBasedRlEnvCfg, device: str, video_length: int, interval: int):
        self.interval = interval
        self._env_cfg = env_cfg
        self._device = device
        self._video_length = video_length

    def maybe_log(self, runner: MjlabOnPolicyRunner, iteration: int) -> None:
        if iteration % self.interval != 0:
            return

        # Single-env copy of the config for eval — override num_envs to avoid wasted memory.
        # If ManagerBasedRlEnvCfg is frozen you may need dataclasses.replace() instead.
        eval_cfg = self._env_cfg
        eval_cfg.scene.num_envs = 1

        eval_env = ManagerBasedRlEnv(cfg=eval_cfg, device=self._device, render_mode="rgb_array")
        eval_env = RslRlVecEnvWrapper(eval_env)

        policy = runner.get_inference_policy(device=self._device) # returns a callable (obs -> actions)

        frames: list[np.ndarray] = []
        obs, _ = eval_env.get_observations()

        with torch.no_grad():
            for _ in range(self._video_length):
                actions = policy(obs)
                obs, _, _, _ = eval_env.step(actions)
                frame = eval_env.render() # (H, W, 3) uint8
                if frame is not None:
                    frames.append(frame)

        eval_env.close()

        if not frames:
            logger.warning(f"Eval video at iteration {iteration} produced no frames — skipping.")
            return

        video_array = np.stack(frames).transpose(0, 3, 1, 2) # (T, C, H, W) as wandb expects
        wandb.log({"eval/policy": wandb.Video(video_array, fps=30, format="mp4")}, step=iteration, commit=True)
        logger.info(f"Logged eval video at iteration {iteration}.")


class RunnerWithEvalVideo(MjlabOnPolicyRunner):
    """Thin subclass that fires EvalVideoLogger on every log() call."""

    eval_video_logger: EvalVideoLogger | None = None  # set after construction

    def log(self, locs: dict, *args, **kwargs) -> None:
        super().log(locs, *args, **kwargs)
        if self.eval_video_logger is not None:
            it = locs.get("it", self.current_learning_iteration)
            self.eval_video_logger.maybe_log(self, it)