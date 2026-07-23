import torch
from torch import nn
import torch.nn.functional as F


def crop_center(h1, h2):
    h1_time, h2_time = h1.size(3), h2.size(3)
    if h1_time == h2_time:
        return h1
    if h1_time < h2_time:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    start = (h1_time - h2_time) // 2
    return h1[:, :, :, start : start + h2_time]


class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def forward(self, input_tensor):
        return self.conv(input_tensor)


class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()

        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def forward(self, input_tensor):
        hidden = self.conv1(input_tensor)
        return self.conv2(hidden)


class Decoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super(Decoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, input_tensor, skip=None):
        input_tensor = F.interpolate(input_tensor, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            input_tensor = torch.cat([input_tensor, crop_center(skip, input_tensor)], dim=1)

        hidden = self.conv1(input_tensor)
        return self.dropout(hidden) if self.dropout is not None else hidden


class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ))
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, input_tensor):
        _, _, h, w = input_tensor.size()

        out = self.bottleneck(
            torch.cat(
                (
                    F.interpolate(self.conv1(input_tensor), size=(h, w), mode="bilinear", align_corners=True),
                    self.conv2(input_tensor),
                    self.conv3(input_tensor),
                    self.conv4(input_tensor),
                    self.conv5(input_tensor),
                ),
                dim=1,
            )
        )
        return self.dropout(out) if self.dropout is not None else out


class LSTMModule(nn.Module):
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)

        self.lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True)

        self.dense = nn.Sequential(nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU())

    def forward(self, input_tensor):
        N, _, nbins, nframes = input_tensor.size()

        hidden, _ = self.lstm(self.conv(input_tensor)[:, 0].permute(2, 0, 1))
        return self.dense(hidden.reshape(-1, hidden.size(-1))).reshape(nframes, N, 1, nbins).permute(1, 2, 3, 0)
