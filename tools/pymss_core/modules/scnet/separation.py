import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM


class FeatureConversion(nn.Module):
    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse, self.channels = inverse, channels

    def forward(self, x):
        x = x.float()
        if self.inverse:
            return torch.fft.irfft(torch.complex(x[:, : self.channels // 2], x[:, self.channels // 2 :]), dim=3, norm="ortho")
        x = torch.fft.rfft(x, dim=3, norm="ortho")
        return torch.cat([x.real, x.imag], dim=1)


class DualPathRNN(nn.Module):
    def __init__(self, d_model, expand, bidirectional=True):
        super().__init__()
        hidden_size = d_model * expand
        self.lstm_layers = nn.ModuleList(
            [LSTM(d_model, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True) for _ in range(2)]
        )
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size * 2, d_model) for _ in range(2)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])

    def forward(self, x):
        B, C, F, T = x.shape

        x = (
            self.linear_layers[0](self.lstm_layers[0](self.norm_layers[0](x).transpose(1, 3).contiguous().view(B * T, F, C))[0])
            .view(B, T, F, C)
            .transpose(1, 3)
            + x
        )

        x = (
            self.linear_layers[1](
                self.lstm_layers[1](self.norm_layers[1](x).transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2))[0]
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, F, C, T)
            .transpose(1, 2)
            + x
        )

        return x


class SeparationNet(nn.Module):
    def __init__(self, channels, expand=1, num_layers=6):
        super().__init__()
        self.dp_modules = nn.ModuleList([DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand) for i in range(num_layers)])
        self.feature_conversion = nn.ModuleList(
            [FeatureConversion(channels * 2, inverse=i % 2 == 1) for i in range(num_layers)]
        )

    def forward(self, x):
        for dp_module, feature_conversion in zip(self.dp_modules, self.feature_conversion):
            x = feature_conversion(dp_module(x))
        return x
