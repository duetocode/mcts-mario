from pathlib import Path
from run import create_env
import json
import cv2


def replay(data_dir: str | None):
    """
    This function replays the recorded gameplay from the given data directory into a mp4 file.

    Parameters
    ----------
    data_dir : str | None
        The directory contains the gameplay data created by the `run.py`
    """
    # Check the data directory
    if data_dir is None:
        # get the latest directory
        saved_dir = sorted(Path("data").iterdir(), reverse=True)[0]
        if not saved_dir.exists():
            raise ValueError("No saved game play found")
        print(f"Using the latest directory: {saved_dir}")
    else:
        saved_dir = Path(data_dir)

    if not (saved_dir.exists() and saved_dir.is_dir()):
        raise ValueError(f"{saved_dir} is not a valid directory")

    # initialize the mp4 video writer
    video_file = Path(saved_dir, "gameplay.mp4")
    video_writer = cv2.VideoWriter(
        str(video_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        60.0,
        (256, 240),
        True,
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
        for _ in range(8):
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
    print(str(video_file))
    # finish the game play
    video_writer.release()
    # also save the number of the frames
    (Path(saved_dir) / "frames.txt").write_text(str(frames))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render the gameplay data into a video."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        help="The directory containing the saved game play data",
    )
    args = parser.parse_args()
    replay(data_dir=args.data_dir)
