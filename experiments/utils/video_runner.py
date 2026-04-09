from __future__ import annotations

from typing import Callable, Type

from mjlab.rl import MjlabOnPolicyRunner


def make_video_runner_cls(base_cls: Type[MjlabOnPolicyRunner] | Type) -> Type[MjlabOnPolicyRunner]:

    class _VideoRunner(base_cls):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._iteration_callbacks: list[Callable] = []

        def add_on_iteration_callback(self, fn: Callable) -> None:
            # fn signature: (iteration: int, runner) -> None
            self._iteration_callbacks.append(fn)

        def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False, **kwargs):
            for i in range(num_learning_iterations):
                super().learn(
                    num_learning_iterations=1,
                    init_at_random_ep_len=init_at_random_ep_len and i == 0,  # only on first iter
                )
                iteration = self.current_learning_iteration
                for callback in self._iteration_callbacks:
                    callback(iteration, self)

    _VideoRunner.__name__ = f"Video{base_cls.__name__}"
    return _VideoRunner