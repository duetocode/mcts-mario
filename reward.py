from typing import Any
from collections import deque
from gym_super_mario_bros import SuperMarioBrosEnv
from gymnasium import Wrapper


class MarioReward(Wrapper):

    def __init__(
        self,
        env: SuperMarioBrosEnv,
        terminate_on_stuck: bool = True,
        queue_length: int = 15,
    ):
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
                and sum(self._x_displacements) / len(self._x_displacements) < 30
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
            return -36

        # x position reward
        x_reward = min(5, max(-5, x_pos - self._last_x_position))
        self._last_x_position = max(x_pos, self._last_x_position)

        # score reward a bit
        # score_reward = min(1, max(0, score - self._last_score))
        # self._last_score = score

        # flag reward
        flag_reward = 250 if flag_get else 0

        # time penalty
        time_penalty = max(0, (time - self._last_time))
        self._last_time = time

        # put it all together
        return x_reward + flag_reward + time_penalty
