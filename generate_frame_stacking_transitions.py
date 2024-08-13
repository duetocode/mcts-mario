"""This program generates the frame stacking transitions from the game play of the agent."""

from pathlib import Path
from run import create_env
import json
import shutil
import numpy as np
import gymnasium as gym

from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
)
from frame_skipping import FrameSkip
from nes_py.wrappers.joypad_space import JoypadSpace
import h5py

FAST_MOVE = [
    ["NOOP"],
    ["right", "B"],
    ["right", "A", "B"],
    ["left"],
]


def replay():
    # get the latest directory
    saved_dir = sorted(Path("data").iterdir(), reverse=True)[0]
    if not saved_dir.exists():
        raise ValueError("No saved game play found")
    print(f"Using the latest directory: {saved_dir}")

    if not (saved_dir.exists() and saved_dir.is_dir()):
        raise ValueError(f"{saved_dir} is not a valid directory")

    # initialize the data
    sample_data = Path(saved_dir, "samples")
    shutil.rmtree(sample_data, ignore_errors=True)
    sample_data.mkdir(parents=True)

    # initialize the environment without frame skipping
    env = gym.make("SuperMarioBros-4-1-v0", render_mode="rgb_array", headless=False)
    env = JoypadSpace(env, FAST_MOVE)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (120, 128))
    env = FrameStack(env, num_stack=8)
    env.reset()

    # run the saved game play
    done, index, frames = False, 0, 0
    while not done:
        # read the step data
        step_info = json.loads(Path(saved_dir, f"{index:04d}.json").read_text())
        # run the game with 4 steps
        for _ in range(8):
            # step
            obs, reward, terminated, truncated, _ = env.step(step_info["action"])
            # save the transition
            sample_name = sample_data / f"{index:04d}.npz"
            np.savez_compressed(sample_name, obs=obs)
            # render the game
            frames += 1
            # determine if the game is ended
            if terminated or truncated:
                done = True
                break

        # increase the index
        index += 1

    print(f"Total frames: {frames}")


if __name__ == "__main__":
    replay()
