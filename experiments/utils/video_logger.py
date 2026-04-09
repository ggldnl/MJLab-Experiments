from __future__ import annotations

import numpy as np
import wandb

from mjlab.envs import ManagerBasedRlEnv


class WandbVideoLogger:
    # Captures a short rollout from env and uploads it to wandb as a video panel entry

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        run: wandb.sdk.wandb_run.Run,
        fps: int,
        length_seconds: float,
        frequency: int,
    ) -> None:
        self._env = env
        self._run = run
        self._fps = fps
        self._num_frames = int(fps * length_seconds)
        self._frequency = frequency

    def __call__(self, iteration: int, runner: object) -> None:
        if iteration % self._frequency != 0:
            return

        frames = self._capture(runner)
        if frames is None:
            return

        # wandb.Video expects (T, C, H, W) uint8
        video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        self._run.log({"video": wandb.Video(video, fps=self._fps, format="mp4")}, step=iteration)

    def _capture(self, runner: object) -> list[np.ndarray] | None:
        # Use the runner policy in eval mode, then restore training mode
        policy = runner.get_inference_policy(device=self._env.device)

        obs, _ = self._env.reset()
        frames: list[np.ndarray] = []

        for _ in range(self._num_frames):
            with __import__("torch").no_grad():
                actions = policy(obs)
            obs, _, _, _, _ = self._env.step(actions)
            frame = self._env.render()  # expects (H, W, 3) uint8 from your render backend
            if frame is None:
                return None
            frames.append(frame)

        return frames

    def close(self):
        pass