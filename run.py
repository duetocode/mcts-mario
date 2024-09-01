import multiprocessing as mp
import datetime as dt
import time
import os

import gymnasium as gym

from nes_py.wrappers import JoypadSpace

from agent_kane import AgentKane
from game_play_recorder import GamePlayRecorder
from environment import create_env


def run():
    """Run the game."""
    # Prepare the environment
    env = create_env(render_mode="human")
    state, _ = env.reset()

    # and the environments that will be used for the simulations of the MCTS
    agent = AgentKane(env_provider=create_env, num_workers=int(os.cpu_count() * 2))

    # Record the gameplay steps. These data can be renders to actual game play with the `replay.py`
    recorder = GamePlayRecorder(f"data/{dt.datetime.now().isoformat()}")

    # The game play loop
    done = False
    while not done:
        # make decision based on the observation
        t_0 = time.time()
        action, tree = agent.act(env, state)
        t_1 = time.time()

        # execute the decision
        _, reward, terminated, truncated, _ = env.step(action)

        # render for visualisation
        env.render()

        # recording the information about the step
        recorder.record(
            {
                "action": action,
                "reward": reward,
                "time": t_1 - t_0,
            },
            env.serialize(),
            tree,
        )

        # check if game is ended
        if terminated or truncated:
            break

    # clean up
    env.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    run()
