import sys
sys.path.append('D:/GitHub/Retrieval-based-Voice-Conversion-WebUI')


from io import BytesIO
import os
from typing import List, Optional, Tuple
import numpy as np
import torch

from infer.lib import jit


import torch.nn as nn
import torch.nn.functional as F
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window

import logging

logger = logging.getLogger(__name__)


class STFT(torch.nn.Module):
    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        assert filter_length >= self.win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(self, input_data, return_phase=False):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)
        """
        input_data = F.pad(
            input_data,
            (self.pad_amount, self.pad_amount),
            mode="reflect",
        )
        forward_transform = input_data.unfold(
            1, self.filter_length, self.hop_length
        ).permute(0, 2, 1)
        forward_transform = torch.matmul(self.forward_basis, forward_transform)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def inverse(self, magnitude, phase):
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced
        by the ```transform``` function.

        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        cat = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        fold = torch.nn.Fold(
            output_size=(1, (cat.size(-1) - 1) * self.hop_length + self.filter_length),
            kernel_size=(1, self.filter_length),
            stride=(1, self.hop_length),
        )
        inverse_transform = torch.matmul(self.inverse_basis, cat)
        inverse_transform = fold(inverse_transform)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        window_square_sum = (
            self.fft_window.pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0)
        )
        window_square_sum = fold(window_square_sum)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        inverse_transform /= window_square_sum
        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        self.magnitude, self.phase = self.transform(input_data, return_phase=True)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


from time import time as ttime


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        # self.shortcut:Optional[nn.Module] = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Intermediate(nn.Module):  #
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,    
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
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
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        # print(mel.shape)
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        # print(x.shape)
        return x


from librosa.filters import mel


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )
        if "privateuseone" in str(audio.device):
            if not hasattr(self, "stft"):
                self.stft = STFT(
                    filter_length=n_fft_new,
                    hop_length=hop_length_new,
                    win_length=win_length_new,
                    window="hann",
                ).to(audio.device)
            magnitude = self.stft.transform(audio)
        else:
            fft = torch.stft(
                audio,
                n_fft=n_fft_new,
                hop_length=hop_length_new,
                win_length=win_length_new,
                window=self.hann_window[keyshift_key],
                center=center,
                return_complex=True,
            )
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half == True:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    def __init__(self, model_path: str, is_half, device=None, use_jit=False):
        self.resample_kernel = {}
        self.resample_kernel = {}
        self.is_half = is_half
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mel_extractor = MelSpectrogram(
            is_half, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)
        if "privateuseone" in str(device):
            import onnxruntime as ort

            ort_session = ort.InferenceSession(
                "%s/rmvpe.onnx" % os.environ["rmvpe_root"],
                providers=["DmlExecutionProvider"],
            )
            self.model = ort_session
        else:
            if str(self.device) == "cuda":
                self.device = torch.device("cuda:0")

            def get_jit_model():
                jit_model_path = model_path.rstrip(".pth")
                jit_model_path += ".half.jit" if is_half else ".jit"
                reload = False
                if os.path.exists(jit_model_path):
                    ckpt = jit.load(jit_model_path)
                    model_device = ckpt["device"]
                    if model_device != str(self.device):
                        reload = True
                else:
                    reload = True

                if reload:
                    ckpt = jit.rmvpe_jit_export(
                        model_path=model_path,
                        mode="script",
                        inputs_path=None,
                        save_path=jit_model_path,
                        device=device,
                        is_half=is_half,
                    )
                model = torch.jit.load(BytesIO(ckpt["model"]), map_location=device)
                return model

            def get_default_model():
                model = E2E(4, 1, (2, 2))
                ckpt = torch.load(model_path, map_location="cpu")
                model.load_state_dict(ckpt)
                model.eval()
                if is_half:
                    model = model.half()
                else:
                    model = model.float()
                return model

            if use_jit:
                if is_half and "cpu" in str(self.device):
                    logger.warning(
                        "Use default rmvpe model. \
                                 Jit is not supported on the CPU for half floating point"
                    )
                    self.model = get_default_model()
                else:
                    self.model = get_jit_model()
            else:
                self.model = get_default_model()

            self.model = self.model.to(device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            if "privateuseone" in str(self.device):
                onnx_input_name = self.model.get_inputs()[0].name
                onnx_outputs_names = self.model.get_outputs()[0].name
                hidden = self.model.run(
                    [onnx_outputs_names],
                    input_feed={onnx_input_name: mel.cpu().numpy()},
                )[0]
            else:
                mel = mel.half() if self.is_half else mel.float()
                hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        # torch.cuda.synchronize()
        # t0 = ttime()
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        mel = self.mel_extractor(
            audio.float().to(self.device).unsqueeze(0), center=True
        )
        # print(123123123,mel.device.type)
        # torch.cuda.synchronize()
        # t1 = ttime()
        hidden = self.mel2hidden(mel)
        # torch.cuda.synchronize()
        # t2 = ttime()
        # print(234234,hidden.device.type)
        if "privateuseone" not in str(self.device):
            hidden = hidden.squeeze(0).cpu().numpy()
        else:
            hidden = hidden[0]
        if self.is_half == True:
            hidden = hidden.astype("float32")

        f0 = self.decode(hidden, thred=thred)
        # torch.cuda.synchronize()
        # t3 = ttime()
        # print("hmvpe:%s\t%s\t%s\t%s"%(t1-t0,t2-t1,t3-t2,t3-t0))
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        # t0 = ttime()
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        # t1 = ttime()
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        # t2 = ttime()
        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        # t3 = ttime()
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= thred] = 0
        # t4 = ttime()
        # print("decode:%s\t%s\t%s\t%s" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return devided






# Mel 频谱计算类（TorchScript 兼容）
class MelSpectrogramTS(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels=128,
        sampling_rate=16000,
        win_length=1024,
        hop_length=160,
        n_fft=1024,
        mel_fmin=30,
        mel_fmax=8000,
        clamp=1e-5
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        
        # 计算 Mel 滤波器组
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        
        # 预计算汉宁窗
        hann_window = torch.hann_window(win_length)
        self.register_buffer("hann_window", hann_window)

    def forward(self, audio):
        # 确保窗口与输入在同一设备上
        window = self.hann_window.to(audio.device)

        # STFT 计算
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        
        # Mel 频谱计算
        mel_basis = self.mel_basis.to(audio.device)  # 确保 mel_basis 也在正确设备上
        mel_output = torch.matmul(mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec

# 完整的 RMVPE 模型，包含 Mel 计算和 F0 提取
class RMVPEWithMel(nn.Module):
    def __init__(self, model_path: str, is_half: bool = False, device: torch.device = torch.device("cpu"), use_jit: bool = False):
        super().__init__()
        self.resample_kernel = {}
        self.is_half = is_half
        self.device = device
        self.mel_extractor = MelSpectrogramTS().to(device)

        def get_default_model():
            # The exact model constructor arguments may need adjustment to match checkpoint.
            model = E2E(n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16)
            ckpt = torch.load(model_path, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]

            model.load_state_dict(ckpt, strict=False)
            model.eval()
            model = model.to(device)
            return model

        self.model = get_default_model().to(device)
        # 注册 cents_mapping 为缓冲区，使其成为模型的一部分
        cents_mapping = 20 * torch.arange(360).float() + 1997.3794084376191
        # pad as in decompiled code
        cents_mapping = F.pad(cents_mapping, (4, 4))
        self.register_buffer("cents_mapping", cents_mapping.to(device)) # 缓冲区也要移动到设备

    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        # mel expected shape: (batch, n_mels, time)
        with torch.no_grad():
            n_frames = mel.size(-1)
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            #if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant", value=0.0)
            
            # 简化设备判断，专注于 TorchScript 导出
            if self.is_half:
                hidden = self.model(mel.half())
                hidden = hidden.float()
            else:
                hidden = self.model(mel.float())
            
            # 如果模型输出是 [batch, time, 360]，则转置为 [batch, 360, time]
            
            hidden = hidden.transpose(1, 2)

            # cut back to original frames
            hidden = hidden[..., :n_frames]
            return hidden

    def decode(self, hidden: torch.Tensor, thred: float = 0.03) -> torch.Tensor:
        cents_pred = self.to_local_average_cents(hidden, thred)
        f0 = 10 * 2 ** (cents_pred / 1200.0)
        mask = (f0 != 10).float()
        f0 = f0 * mask
        return f0

    def infer_from_audio(self, audio: torch.Tensor, thred: float = 0.03) -> torch.Tensor:
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        audio = audio.to(self.device)
        mel = self.mel_extractor(audio.float().unsqueeze(0))  # (1, n_mels, t)
        hidden = self.mel2hidden(mel)
        #hidden = hidden[0]  # 移除批次维度
        if self.is_half:
            hidden = hidden.float()
        f0 = self.decode(hidden, thred)
        #return f0
        return f0.squeeze(0)  # 移除批次维度，返回形状为 [n_time]

    def to_local_average_cents(self, salience: torch.Tensor, thred: float = 0.05):
        # salience: (batch, n_bins, time)
        batch = salience.shape[0]
        n_time = salience.shape[2]
        
        # 找到每个帧的最大值位置
        center = torch.argmax(salience, dim=1)  # (batch, time)
        
        # 填充 salience
        salience_p = F.pad(salience, (0, 0, 4, 4))  # 在频率维度上填充，左右各4
        
        # 使用向量化操作替代循环
        # 创建索引张量
        batch_indices = torch.arange(batch, device=salience.device).view(batch, 1, 1)
        time_indices = torch.arange(n_time, device=salience.device).view(1, n_time, 1)
        
        # 调整 center 索引以考虑填充
        center = center + 4
        
        # 创建窗口索引
        window_indices = torch.arange(9, device=salience.device).view(1, 1, 9)
        indices = center.unsqueeze(2) - 4 + window_indices  # (batch, time, 9)
        
        # 限制索引范围
        indices = torch.clamp(indices, 0, self.cents_mapping.size(0) - 1)
        
        # 收集 salience 值
        todo_salience = salience_p[batch_indices, indices, time_indices]  # (batch, time, 9)
        
        # 收集 cents_mapping 值
        todo_cents = self.cents_mapping[indices]  # (batch, time, 9)
        
        # 计算加权平均
        product_sum = torch.sum(todo_salience * todo_cents, dim=2)  # (batch, time)
        weight_sum = torch.sum(todo_salience, dim=2)  # (batch, time)
        
        # 避免除以零
        weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
        devided = product_sum / weight_sum
        
        # 应用阈值
        maxx = torch.max(salience, dim=1)[0]  # (batch, time)
        devided[maxx <= thred] = 0
        
        return devided

    def forward(self, audio: torch.Tensor, thred: float = 0.03):
        return self.infer_from_audio(audio, thred)

# 导出函数
def export_rmvpe_with_mel(model_path, output_path, is_half=False):
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ 使用 CUDA 设备导出模型")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA 不可用，使用 CPU 设备导出模型")

    # 创建模型
    model = RMVPEWithMel(model_path, is_half, device)
    model.eval()
    
    # 使用真实音频进行测试
    import soundfile as sf
    import librosa
    path = r"C:/Users/Administrator/Desktop/VC/NeuVC/NeuVC/exports/recorded_audio_1.wav"
    audio, sampling_rate = sf.read(path)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    
    # 将 NumPy 数组转换为 PyTorch 张量并移动到正确设备
    audio_tensor = torch.from_numpy(audio).float().to(device)
    
    # 跟踪模型
    traced_model = torch.jit.trace(model, (audio_tensor,))
    
    # 保存 TorchScript 模型
    traced_model.save(output_path)
    print(f"✅ TorchScript 模型已保存到: {output_path}")
    
    # 测试模型
    with torch.no_grad():
        f0 = traced_model(audio_tensor)
        print(f"✅ 模型测试成功，输出 F0 形状: {f0.shape}")
        print(f"✅ F0 范围: {f0.min().item():.2f} - {f0.max().item():.2f} Hz")
        
        # 输出一些非零 F0 值的统计信息
        non_zero_f0 = f0[f0 > 0]
        if len(non_zero_f0) > 0:
            print(f"✅ 非零 F0 值数量: {len(non_zero_f0)}")
            print(f"✅ 非零 F0 值范围: {non_zero_f0.min().item():.2f} - {non_zero_f0.max().item():.2f} Hz")
            print(f"✅ 非零 F0 值平均值: {non_zero_f0.mean().item():.2f} Hz")
        else:
            print("⚠️ 没有检测到非零 F0 值")

if __name__ == "__main__":
    # 导出模型
    model_path = "C:/Users/Administrator/Desktop/Model/rmvpe.pt"
    output_path = "C:/Users/Administrator/Desktop/Model/rmvpe_with_mel.pt"
    export_rmvpe_with_mel(model_path, output_path)

