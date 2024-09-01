# Agent Kane

Agent Kane can speedrun the Super Mario Bros. (NES) with Monte Carlo Tree Search algorithm. It searches the possible future states based on its current state and makes a decision for the action about to take. The agent is oberseved to be able to finish the level 4-1 in 20.217 seconds, which is 1213 frames, surpassing the recorded human player performance in the level. 

If you feel lucky, please try to run the game to see if you can encounter a better gameplay.

## Requirements

- Python 3.10 or higher
- A C++ compiler that supports C++17

The project has been tested on an Apple M1 chip.

## Installation

Before running the programes, please install the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run the game

To run the game, please simply call the following command:

```bash
python run.py
```

It will create a data directory with current time under the `data` directory. All the game data will be stored in the directory. The speed of the tree search algorithm depends on the speed of the computer. The programme shows the gameplay with the frameskip. For smooth gameplay, please use the `replay.py` after finished the game to produce smooth a version:

```bash
python replay.py <gameplay_directory>
```

The about will produce a mp4 file in the specified directory. The mp4 file is a smooth version of the gameplay without frame skipping.

## Machine learning support

The gameplay recordings can be used to boost the other agents that are based on machine learning techniques:

## Initmation learning

The Behaviour Cloning method can use supervised learning to boost the Reinforcement Learning models. The `generate_frame_stacking_transitions.py` can generate frame stacking observations, which can combined with the action decisions produced by the `run.py` as a labelled dataset.

The `run.py` also saves the states of the game environment, which can be loaded back to the environment to restore the state. These recordings allows the ML agents to do the random checkpoint learning, alleviate the bais towards the begining of the game.