from pathlib import Path
from run import create_env
import json
import cv2


def replay(saved_gameplay: str):
    saved_dir = Path(saved_gameplay)
    if not (saved_dir.exists() and saved_dir.is_dir()):
        raise ValueError(f"{saved_gameplay} is not a valid directory")

    # initialize the mp4 video writer
    video_writer = cv2.VideoWriter(
        "gameplay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (256, 240), True
    )

    # initialize the environment without frame skipping
    env = create_env(frame_skip=0, render_mode="rgb_array")
    env.reset()

    # run the saved game play
    done, index, frames = False, 0, 0
    while not done:
        # read the step data
        step_info = json.loads(Path(saved_dir, f"{index:04d}.json").read_text())
        # run the game with 4 steps
        for _ in range(4):
            # step
            _, _, terminated, truncated, _ = env.step(step_info["action"])
            # render the game
            screen = env.render()
            # render the video
            video_writer.write(cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
            frames += 1
            # determine if the game is ended
            if terminated or truncated:
                done = True
                break

        # increase the index
        index += 1

    print(f"Total frames: {frames}")
    # finish the game play
    video_writer.release()


if __name__ == "__main__":
    replay("data/2024-07-22T15:36:33.156997")
