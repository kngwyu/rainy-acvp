from torch import nn, Tensor
from torch.nn import functional as F
from rainy.net import calc_cnn_hidden
from rainy.net.init import Initializer

from typing import List, Sequence


class AcvpNet(nn.Module):
    def __init__(
            self,
            input_dim: Sequence[int],
            action_dim: int,
            hidden_dim: int = 2048,
            conv_channels: List[int] = [64, 128, 128, 128],
            encoder_args: List[tuple] = [(8, 2, 1), (6, 2, 1), (6, 2, 1), (4, 2)],
            decoder_args: List[tuple] = [(4, 2), (6, 2, 1), (6, 2, 1), (8, 2, 1)],
            init: Initializer = Initializer(nonlinearity='relu')
    ):
        super().__init__()

        in_channel = input_dim[0]
        channels = [in_channel] + conv_channels
        self.conv = init.make_list([
            nn.Conv2d(channels[i], channels[i + 1], *encoder_args[i])
            for i in range(len(channels) - 1)
        ])

        conved_dim = calc_cnn_hidden(encoder_args)
        self.fc_enc = nn.Linear(conved_dim, hidden_dim)

        self.w_enc = nn.Linear(conved_dim, hidden_dim, bias=False)
        self.w_action = nn.Linear(action_dim, hidden_dim, bias=False)

        self.fc_action_trans = nn.Linear(hidden_dim, hidden_dim)

        self.fc_dec = nn.Linear(hidden_dim, conved_dim)

        channels.reverse()
        self.deconv = init.make_list([
            nn.ConvTranspose2d(channels[i], channels[i + 1], *decoder_args[i])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x: Tensor, action: Tensor):
        batch_size = x.size(0)
        for conv in self.conv:
            x = F.relu_(conv(x))
        conved_size = x.shape
        x = F.relu_(self.fc_enc(x.view(batch_size, -1)))

        x = self.w_env(x) * self.w_action(action)
        x = self.fc_action_trans(x)

        x = F.relu_(self.fc_dec(x)).view(conved_size)
        for deconv in self.deconv:
            x = F.relu_(deconv(x))
        return x
