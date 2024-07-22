import numpy as np
from PIL import Image
from pathlib import Path
from monte_carlo_tree_search import Node
import pickle
import lzma
import json


class GamePlayRecorder:

    def __init__(self, recording_name: str):
        self._output_dir = Path(recording_name)
        self._output_dir.mkdir(parents=True)
        self._index = 0

    def record(self, play_info: dict, state: bytes, tree: Node):
        # the stem for the data
        file_stem = Path(self._output_dir, f"{self._index:04d}")

        # save the meata data
        json.dump(play_info, file_stem.with_suffix(".json").open("w"))

        # save the state
        file_stem.with_suffix(".state.xz").write_bytes(lzma.compress(state))
        # increase the index
        self._index += 1

        # save the tree
        def to_dict(node: Node) -> dict:
            return {
                "action": node.action,
                "visits": node.visits,
                "value": node.value,
                "is_terminal": node.is_terminal,
                "is_victory": node.is_victory,
                "children": [to_dict(child) for child in node.children],
            }

        tree_data = to_dict(tree)
        file_stem.with_suffix(".tree.json").write_bytes(pickle.dumps(tree_data))
