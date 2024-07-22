from gymnasium import Wrapper, Env


class FrameSkip(Wrapper):
    """A wrapper that skips a number of frames wth the specified action and returns the accumulated reward."""

    def __init__(self, env: Env, frame_skip: int = 4):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action: int) -> tuple:
        """Execute the action for a number of frames and return the accumulated reward."""
        # accumulated reward
        total_reward = 0
        # step the environment for the specified number of frames
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # accumulate the reward
            total_reward += reward
            # break if the game is ended
            if terminated or truncated:
                break

        # return the latest data with the accumulated reward
        return obs, total_reward, terminated, truncated, info
