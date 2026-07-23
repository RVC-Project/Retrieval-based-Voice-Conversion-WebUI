import torch
from torch import nn
import torch.nn.functional as F
from . import layers_new as layers


class BaseNet(nn.Module):
    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = layers.Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(nout * 6, nout * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)

        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def forward(self, input_tensor):
        encoded1 = self.enc1(input_tensor)
        encoded2 = self.enc2(encoded1)
        encoded3 = self.enc3(encoded2)
        encoded4 = self.enc4(encoded3)
        encoded5 = self.enc5(encoded4)

        bottleneck = self.aspp(encoded5)

        bottleneck = self.dec4(bottleneck, encoded4)
        bottleneck = self.dec3(bottleneck, encoded3)
        bottleneck = self.dec2(bottleneck, encoded2)
        bottleneck = torch.cat([bottleneck, self.lstm_dec2(bottleneck)], dim=1)
        bottleneck = self.dec1(bottleneck, encoded1)

        return bottleneck


class CascadedNet(nn.Module):
    def __init__(self, n_fft, nn_arch_size=51000, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64
        nout = 64 if nn_arch_size == 218409 else nout

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm), layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0)
        )
        self.stg1_high_band_net = BaseNet(2, nout // 4, self.nin_lstm // 2, nout_lstm // 2)

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm), layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0)
        )
        self.stg2_high_band_net = BaseNet(nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2)

        self.stg3_full_band_net = BaseNet(3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm)

        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, input_tensor):
        input_tensor = input_tensor[:, :, : self.max_bin]

        bandw = input_tensor.size()[2] // 2
        l1_in = input_tensor[:, :, :bandw]
        h1_in = input_tensor[:, :, bandw:]

        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)

        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)

        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)

        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([input_tensor, aux1, aux2], dim=1)

        f3 = self.stg3_full_band_net(f3_in)

        mask = torch.sigmoid(self.out(f3))

        mask = F.pad(input=mask, pad=(0, 0, 0, self.output_bin - mask.size()[2]), mode="replicate")

        if self.training:
            aux = torch.sigmoid(self.aux_out(torch.cat([aux1, aux2], dim=1)))
            aux = F.pad(input=aux, pad=(0, 0, 0, self.output_bin - aux.size()[2]), mode="replicate")
            return mask, aux
        return mask

    def predict_mask(self, input_tensor):
        mask = self.forward(input_tensor)

        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
            assert mask.size()[3] > 0

        return mask

    def predict(self, input_tensor):
        mask = self.forward(input_tensor)
        pred_mag = input_tensor * mask

        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag
