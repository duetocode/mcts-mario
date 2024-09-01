from gymnasium import Env, make
from nes_py.wrappers import JoypadSpace
from mario_reward import MarioReward
from frame_skipping import FrameSkip
from action_space import FAST_MOVE


def create_env(
    frame_skip: int = 8,
    headless: bool = False,
    with_reward: bool = False,
    render_mode: str = "rgb_array",
) -> Env:
    """
    Create the environment that host the game.

    Parameters
    ----------
    frame_skip : int
        The number of frames to skip between each step.
    headless : bool
        Whether to run the game in headless mode (skip the actual rendering in VRAM)
    with_reward : bool
        Whether to use the `MarioReward` wrapper for the environment.
    render_mode : str
        The mode to render the game for the environment.
    """
    # create the basic environment
    env = make("SuperMarioBros-4-1-v0", render_mode=render_mode, headless=headless)

    # define the action space with the FAST_MOVE for speedrunning
    env = JoypadSpace(env, FAST_MOVE)

    # the reward for speedrunning
    if with_reward:
        env = MarioReward(env)

    # the frame skip to save computational costs
    if frame_skip > 0:
        env = FrameSkip(env, frame_skip)

    return env
