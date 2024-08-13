from pathlib import Path
import random
import lzma
from gymnasium import Wrapper, Env


class RandomEpisode(Wrapper):
    """Begin the episode with a random checkpoint."""

    def __init__(self, env: Env, data_dir: Path):
        super().__init__(env)

        self._data_dir = data_dir

        # iterate the checkpoints
        if not (data_dir.exists() and data_dir.is_dir()):
            raise ValueError(f"Invalid data directory: {data_dir}")

        self._checkpoints = sorted(data_dir.glob("*.state.xz"))

    def reset(self):
        # reset the environment
        super().reset()

        # randomly select a checkpoint
        checkpoint = random.choice(self._checkpoints)

        # load the checkpoint
        saved_state = lzma.decompress(checkpoint.read_bytes())
        self.deserialize(saved_state)
