import torch
from torch import nn
import torch.nn.functional as F

from . import layers


class BaseASPPNet(nn.Module):
    def __init__(self, nn_architecture, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.nn_architecture = nn_architecture

        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        if self.nn_architecture == 129605:
            self.enc5 = layers.Encoder(ch * 8, ch * 16, 3, 2, 1)
            self.aspp = layers.ASPPModule(nn_architecture, ch * 16, ch * 32, dilations)
            self.dec5 = layers.Decoder(ch * (16 + 32), ch * 16, 3, 1, 1)
        else:
            self.aspp = layers.ASPPModule(nn_architecture, ch * 8, ch * 16, dilations)

        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def forward(self, input_tensor):
        hidden_state, encoder_output1 = self.enc1(input_tensor)
        hidden_state, encoder_output2 = self.enc2(hidden_state)
        hidden_state, encoder_output3 = self.enc3(hidden_state)
        hidden_state, encoder_output4 = self.enc4(hidden_state)

        if self.nn_architecture == 129605:
            hidden_state, encoder_output5 = self.enc5(hidden_state)
            hidden_state = self.aspp(hidden_state)
            hidden_state = self.dec5(hidden_state, encoder_output5)
        else:
            hidden_state = self.aspp(hidden_state)

        hidden_state = self.dec4(hidden_state, encoder_output4)
        hidden_state = self.dec3(hidden_state, encoder_output3)
        hidden_state = self.dec2(hidden_state, encoder_output2)
        hidden_state = self.dec1(hidden_state, encoder_output1)

        return hidden_state


def determine_model_capacity(n_fft_bins, nn_architecture):
    sp_model_arch = [31191, 33966, 129605]
    hp_model_arch = [123821, 123812]
    hp2_model_arch = [537238, 537227]

    if nn_architecture in sp_model_arch:
        model_capacity_data = [
            (2, 16),
            (2, 16),
            (18, 8, 1, 1, 0),
            (8, 16),
            (34, 16, 1, 1, 0),
            (16, 32),
            (32, 2, 1),
            (16, 2, 1),
            (16, 2, 1),
        ]

    if nn_architecture in hp_model_arch:
        model_capacity_data = [
            (2, 32),
            (2, 32),
            (34, 16, 1, 1, 0),
            (16, 32),
            (66, 32, 1, 1, 0),
            (32, 64),
            (64, 2, 1),
            (32, 2, 1),
            (32, 2, 1),
        ]

    if nn_architecture in hp2_model_arch:
        model_capacity_data = [
            (2, 64),
            (2, 64),
            (66, 32, 1, 1, 0),
            (32, 64),
            (130, 64, 1, 1, 0),
            (64, 128),
            (128, 2, 1),
            (64, 2, 1),
            (64, 2, 1),
        ]

    return CascadedASPPNet(n_fft_bins, model_capacity_data, nn_architecture)


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft, model_capacity_data, nn_architecture):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[0])
        self.stg1_high_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[1])

        self.stg2_bridge = layers.Conv2DBNActiv(*model_capacity_data[2])
        self.stg2_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[3])

        self.stg3_bridge = layers.Conv2DBNActiv(*model_capacity_data[4])
        self.stg3_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[5])

        self.out = nn.Conv2d(*model_capacity_data[6], bias=False)
        self.aux1_out = nn.Conv2d(*model_capacity_data[7], bias=False)
        self.aux2_out = nn.Conv2d(*model_capacity_data[8], bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, input_tensor):
        mix = input_tensor.detach()
        input_tensor = input_tensor.clone()

        input_tensor = input_tensor[:, :, : self.max_bin]

        bandwidth = input_tensor.size()[2] // 2
        aux1 = torch.cat(
            [self.stg1_low_band_net(input_tensor[:, :, :bandwidth]), self.stg1_high_band_net(input_tensor[:, :, bandwidth:])],
            dim=2,
        )

        hidden_state = torch.cat([input_tensor, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(hidden_state))

        hidden_state = torch.cat([input_tensor, aux1, aux2], dim=1)
        hidden_state = self.stg3_full_band_net(self.stg3_bridge(hidden_state))

        mask = torch.sigmoid(self.out(hidden_state))

        mask = F.pad(input=mask, pad=(0, 0, 0, self.output_bin - mask.size()[2]), mode="replicate")

        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(input=aux1, pad=(0, 0, 0, self.output_bin - aux1.size()[2]), mode="replicate")
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(input=aux2, pad=(0, 0, 0, self.output_bin - aux2.size()[2]), mode="replicate")
            return mask * mix, aux1 * mix, aux2 * mix
        return mask

    def predict_mask(self, input_tensor):
        mask = self.forward(input_tensor)

        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]

        return mask
