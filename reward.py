from typing import Any
from collections import deque
from gym_super_mario_bros import SuperMarioBrosEnv
from gymnasium import Wrapper


class MarioReward(Wrapper):
    """
    A wrapper for the SuperMarioBrosEnv that focuses on speed running, which encourages the agent to sprint forward (right) as fast as possible.
    It also penalizes death and stuck, which cause early termination.
    """

    def __init__(
        self,
        env: SuperMarioBrosEnv,
        terminate_on_stuck: bool = True,
        queue_length: int = 15,
    ):
        """
        Initialize the wrapper.
        args:
            env: The environment to wrap.
            terminate_on_stuck: If True, the episode will terminate if the agent is stuck.
            queue_length: The number of frames to keep track of for the stuck detection.
        """
        super().__init__(env)
        self.terminate_on_stuck = terminate_on_stuck
        self._n_frames = 0
        self._x_displacements = deque(maxlen=queue_length)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple:
        self._last_score = 0
        self._last_x_position = 0
        obs, info = self.env.reset(seed=seed, options=options)

        self._last_score = info["score"]
        self._last_x_position = info["x_pos"]
        self._last_time = info["time"]
        # displacement tracking for stuck detection
        self._x_displacements.clear()

        return obs, info

    def step(self, action: int) -> tuple:
        # call to the step but ignore the reward because we are going to replace it
        obs, _, terminated, truncated, info = self.env.step(action)

        # struck detection
        if self.terminate_on_stuck and (not info["flag_get"]):
            self._x_displacements.append(info["x_pos"] - self._last_x_position)
            if (
                len(self._x_displacements) >= self._x_displacements.maxlen
                and sum(self._x_displacements) / len(self._x_displacements) < 2
            ):
                # stuck detected, terminate the episode and return negative reward
                return obs, -10, True, truncated, info

        # get the reward
        reward = self._get_reward(info)

        return obs, reward, terminated or truncated, truncated, info

    def _get_reward(self, info: dict) -> float:
        x_pos, score, flag_get, time, is_dead = tuple(
            info[k]
            for k in (
                "x_pos",
                "score",
                "flag_get",
                "time",
                "is_dead",
            )
        )

        if is_dead:
            return -36 * 4

        # x position reward
        x_reward = min(5, max(-5, x_pos - self._last_x_position)) * 2
        self._last_x_position = max(x_pos, self._last_x_position)

        # flag reward
        flag_reward = 500 if flag_get else 0

        # time penalty
        time_penalty = max(0, (time - self._last_time))
        self._last_time = time

        # put it all together
        return x_reward + flag_reward + time_penalty
