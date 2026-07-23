import torch
import torch.nn as nn


class SubbandSTFT:
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        channels, length = x.shape[-2:]
        x = torch.stft(
            x.reshape(-1, length),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x).permute(0, 3, 1, 2)
        x = x.reshape(*batch_dims, channels * 2, -1, x.shape[-1])
        return x[..., : self.dim_f, :]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        channels, freq_bins, time_bins = x.shape[-3:]
        full_freq_bins = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, channels, full_freq_bins - freq_bins, time_bins]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape(-1, 2, full_freq_bins, time_bins).permute(0, 2, 3, 1)
        x = x[..., 0] + x[..., 1] * 1.0j
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
        )
        return x.reshape([*batch_dims, 2, -1])


def get_activation(act_type):
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "relu":
        return nn.ReLU()
    if act_type[:3] == "elu":
        alpha = float(act_type.replace("elu", ""))
        return nn.ELU(alpha)
    raise Exception


def cac_to_cws(x, num_subbands):
    batch, channels, freq_bins, time_bins = x.shape
    return x.reshape(batch, channels * num_subbands, freq_bins // num_subbands, time_bins)


def cws_to_cac(x, num_subbands):
    batch, channels, freq_bins, time_bins = x.shape
    return x.reshape(batch, channels // num_subbands, freq_bins * num_subbands, time_bins)


def forward_subband_mask_model(module, x, core_fn):
    x = module.stft(x)

    mix = x = cac_to_cws(x, module.num_subbands)

    first_conv_out = x = module.first_conv(x)

    x = core_fn(x.transpose(-1, -2)).transpose(-1, -2)

    x = x * first_conv_out

    x = module.final_conv(torch.cat([mix, x], 1))

    x = cws_to_cac(x, module.num_subbands)

    if module.num_target_instruments > 1:
        batch, channels, freq_bins, time_bins = x.shape
        x = x.reshape(batch, module.num_target_instruments, -1, freq_bins, time_bins)

    return module.stft.inverse(x)
