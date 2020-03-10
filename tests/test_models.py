import torch

import gym
from acvp import models
from rainy.utils import Device


def test_ffmodel_for_atari() -> None:
    atari = gym.make("BreakoutNoFrameskip-v0")
    acvp_netfn = models.prepare_ff()
    d = Device()
    acvp_net = acvp_netfn((3, 210, 160), 4, d)
    states, actions = [], []
    atari.reset()
    for _ in range(10):
        s, _, _, _ = atari.step(0)
        states.append(s.transpose(2, 0, 1))
        actions.append(0)
    states = d.tensor(states)
    actions = d.tensor(actions, dtype=torch.long)
    s_decoded = acvp_net(states, actions)
    assert tuple(s_decoded.shape) == (10, 3, 210, 160)
