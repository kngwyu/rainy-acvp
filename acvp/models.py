from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rainy.net import calc_cnn_hidden
from rainy.net.init import Initializer, orthogonal
from rainy.net.prelude import NetFn
from rainy.utils import Device


class FFAcvpNet(nn.Module):
    def __init__(
        self,
        input_dim: Sequence[int],
        action_dim: int,
        hidden_dim: int = 2048,
        conv_channels: List[int] = [64, 128, 128, 128],
        encoder_args: List[tuple] = [(8, 2, (0, 1)), (6, 2, 1), (6, 2, 1), (4, 2)],
        decoder_args: List[tuple] = [(4, 2), (6, 2, 1), (6, 2, 1), (8, 2, (0, 1))],
        device: Device = Device(),
        init: Initializer = Initializer(orthogonal(nonlinearity="relu")),
    ) -> None:
        super().__init__()

        in_channel, height, width = input_dim
        channels = [in_channel] + conv_channels
        self.conv = init.make_list(
            [
                nn.Conv2d(channels[i], channels[i + 1], *encoder_args[i])
                for i in range(len(channels) - 1)
            ]
        )

        conved_dim = (
            np.prod(calc_cnn_hidden(encoder_args, height, width)) * channels[-1]
        )
        self.fc_enc = nn.Linear(conved_dim, hidden_dim)

        self.w_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_action = nn.Linear(action_dim, hidden_dim, bias=False)

        self.fc_action_trans = nn.Linear(hidden_dim, hidden_dim)

        self.fc_dec = nn.Linear(hidden_dim, conved_dim)

        channels.reverse()
        self.deconv = init.make_list(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], *decoder_args[i])
                for i in range(len(channels) - 1)
            ]
        )
        self.action_dim = action_dim
        self.device = device
        self.to(device.unwrapped)

    def _onehot(self, actions: Tensor) -> Tensor:
        batch_size = actions.size(0)
        indices = torch.arange(batch_size, device=self.device.unwrapped)
        res = self.device.zeros((batch_size, self.action_dim))
        res[indices, actions] = 1.0
        return res

    def forward(self, x: Tensor, actions: Tensor) -> Tensor:
        batch_size = x.size(0)
        for conv in self.conv:
            x = F.relu_(conv(x))
        conved_size = x.shape
        x = F.relu_(self.fc_enc(x.view(batch_size, -1)))

        onehot_actions = self._onehot(actions)
        x = self.w_enc(x) * self.w_action(onehot_actions)
        x = self.fc_action_trans(x)

        x = F.relu_(self.fc_dec(x)).view(conved_size)
        for deconv in self.deconv:
            x = F.relu_(deconv(x))
        return x


def prepare_ff(
    conv_channels: List[int] = [64, 128, 128, 128],
    encoder_args: List[tuple] = [(8, 2, (0, 1)), (6, 2, 1), (6, 2, 1), (4, 2)],
    decoder_args: List[tuple] = [(4, 2), (6, 2, 1), (6, 2, 1), (8, 2, (0, 1))],
    hidden_dim: int = 2048,
    **kwargs,
) -> NetFn:
    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> FFAcvpNet:
        return FFAcvpNet(
            state_dim,
            action_dim,
            hidden_dim,
            conv_channels,
            encoder_args,
            decoder_args,
            device,
        )

    return _net  # type: ignore
