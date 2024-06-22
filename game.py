from typing import Any


class GameState:
    """The state of the simulation for the game."""

    def __init__(self, state: Any = None):
        self._state = state

    def clone(self) -> "GameState":
        """Return a copy of the current game state."""
        return GameState(self._state)
