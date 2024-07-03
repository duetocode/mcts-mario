import gymnasium as gym
import time
from agent_kane import AgentKane
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from reward import MarioReward
import multiprocessing as mp


def create_env(headless: bool = False, with_reward=False, render_mode="rgb_array"):
    env = gym.make("SuperMarioBros-1-1-v0", render_mode=render_mode, headless=headless)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    if with_reward:
        env = MarioReward(env)

    return env


def run():
    env = create_env(render_mode="human")
    state, _ = env.reset()

    agent = AgentKane(env_provider=create_env)

    done = False

    while not done:
        action = agent.act(env, state)
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            env.render()
            if done:
                break

    env.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    run()
