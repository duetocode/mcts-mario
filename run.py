import multiprocessing as mp
import datetime as dt
import time
import os

import gymnasium as gym

from nes_py.wrappers import JoypadSpace

from agent_kane import AgentKane
from mario_reward import MarioReward
from frame_skipping import FrameSkip
from game_play_recorder import GamePlayRecorder

FAST_MOVE = [
    ["NOOP"],
    ["right", "B"],
    ["right", "A", "B"],
    ["left"],
]


def create_env(
    frame_skip: int = 8,
    headless: bool = False,
    with_reward: bool = False,
    render_mode: str = "rgb_array",
):
    env = gym.make("SuperMarioBros-4-1-v0", render_mode=render_mode, headless=headless)
    env = JoypadSpace(env, FAST_MOVE)

    if with_reward:
        env = MarioReward(env)

    if frame_skip > 0:
        env = FrameSkip(env, frame_skip)

    return env


def run():
    env = create_env(render_mode="human")
    state, _ = env.reset()

    agent = AgentKane(env_provider=create_env, num_workers=int(os.cpu_count() * 2))
    recorder = GamePlayRecorder(f"data/{dt.datetime.now().isoformat()}")

    done = False

    while not done:
        t_0 = time.time()
        action, tree = agent.act(env, state)
        t_1 = time.time()
        _, reward, terminated, truncated, _ = env.step(action)
        env.render()
        recorder.record(
            {
                "action": action,
                "reward": reward,
                "time": t_1 - t_0,
            },
            env.serialize(),
            tree,
        )

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    run()
