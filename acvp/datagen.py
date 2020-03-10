from typing import Dict

import click
import numpy as np
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset

from rainy.agents import Agent
from rainy.envs import EnvExt, EnvTransition
from rainy.lib.hooks import EvalHook
from rainy.prelude import Action, State


class ImageWriterHook(EvalHook):
    def __init__(self, image_shape: str = "HWC", out_dir: str = "dataset") -> None:
        self.logdir = None
        self.writer = None
        self.episode_id = 0
        self.total_steps = 0
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self._w_index = image_shape.find("W")
        self._h_index = image_shape.find("H")
        self._state_buffer = []
        self._action_buffer = []
        if self._w_index < 0 or self._h_index < 0:
            raise ValueError(f"Invalid shape: {image_shape}")
        c_index = image_shape.find("C")
        if c_index < 0:
            self._transpose = self._h_index, self._w_index
        else:
            self._transpose = self._h_index, self._w_index, c_index

    def reset(self, _agent: Agent, env: EnvExt, _initial_state: State) -> None:
        self.episode_id += 1
        image = env.render(mode="rgb_array")
        self._state_buffer.append(np.transpose(image, self._transpose))

    def step(
        self,
        env: EnvExt,
        action: Action,
        transition: EnvTransition,
        _net_outputs: Dict[str, Tensor],
    ) -> None:
        image = env.render(mode="rgb_array")
        self._state_buffer.append(np.transpose(image, self._transpose))
        self._action_buffer.append(action)
        if transition.terminal:
            out = self.out_dir.joinpath(f"ep{self.episode_id}.npz")
            states = np.stack(self._state_buffer)
            actions = np.stack(self._action_buffer)
            np.savez_compressed(out, states=states, actions=actions)
            self._state_buffer.clear()
            self._action_buffer.clear()


class AtariData(Dataset):
    def __init__(self, file_name: str) -> None:
        pass


@click.command()
def datagen_main():
    pass


if __name__ == "__main__":
    datagen_main()
