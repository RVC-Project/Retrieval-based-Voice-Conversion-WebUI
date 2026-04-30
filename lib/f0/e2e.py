from typing import Final, cast

import torch.nn as nn

from .deepunet import DeepUnet

N_MELS: Final = cast(int, getattr(nn, "N_MELS", 128))
N_CLASS: Final = cast(int, getattr(nn, "N_CLASS", 360))


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: tuple[int, int],
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()

        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                self.BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

    class BiGRU(nn.Module):
        def __init__(
            self,
            input_features: int,
            hidden_features: int,
            num_layers: int,
        ):
            super().__init__()
            self.gru = nn.GRU(
                input_features,
                hidden_features,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        def forward(self, x):
            return self.gru(x)[0]
