from typing import Any
from gym_super_mario_bros import SuperMarioBrosEnv
from gymnasium import Wrapper, Env


class MarioReward(Wrapper):
    """
    A wrapper for the SuperMarioBrosEnv that focuses on speed running, which encourages the agent to sprint forward (right) as fast as possible.
    It also penalizes death and stuck, which cause early termination.
    """

    def __init__(
        self,
        env: Env,
        max_stuck_frames: int = 8,
    ):
        """
        Initialize the wrapper.
        args:
            env: The environment to wrap.
            max_stuck_frame: int The maximum number of frames the agent can be stuck before the episode is terminated.
        """
        super().__init__(env)
        self.max_stuck_frames = max_stuck_frames

        # the progress of the gamy
        self._progress = 0
        # the number of frames the agent has been stuck
        self._n_stuck_frames = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple:
        # call the original reset
        obs, info = self.env.reset(seed=seed, options=options)

        # reset the internal variables with the new episode
        self._progress = info["x_pos"]
        self._n_stuck_frames = 0

        return obs, info

    def step(self, action: int) -> tuple:
        # call to the step but ignore the reward because we are going to replace it
        obs, _, terminated, truncated, info = self.env.step(action)

        # get the information from the environment
        x_pos, y_pos = info["x_pos"], info["y_pos"]
        flag_get, is_dead = info["flag_get"], info["is_dead"]

        # death penalty and
        if is_dead or y_pos < 75:
            return obs, -50, True, truncated, info

        # stuck penalty
        x_displacement = x_pos - self._progress
        self._progress = x_pos
        if x_displacement >= -5 and x_displacement <= 5:
            if x_displacement < 1:
                self._n_stuck_frames += 1
            else:
                self._n_stuck_frames = 0
            stuck_penalty = -self._n_stuck_frames
            # early termination if stuck
            if self._n_stuck_frames >= self.max_stuck_frames:
                # return obs, stuck_penalty, True, truncated, info
                return obs, stuck_penalty, True, truncated, info

            # x position reward
            x_reward = x_displacement
        else:
            x_reward = 0
            stuck_penalty = 0

        # flag reward
        if flag_get:
            flag_reward = 50
            # encourage to reach the bottom of the pole
            flag_reward += 240 - y_pos
        else:
            flag_reward = 0

        # put it all together
        reward = stuck_penalty + x_reward + flag_reward

        # return obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info
