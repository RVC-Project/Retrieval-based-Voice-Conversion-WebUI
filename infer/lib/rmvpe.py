from io import BytesIO
import os
from typing import Literal, TypeAlias, cast, overload
import numpy as np
import torch

from infer.lib import jit

try:
    # Fix "Torch not compiled with CUDA enabled"
    import intel_extension_for_pytorch as ipex  # type: ignore[missing-import]  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import torch.nn as nn
import torch.nn.functional as F
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window

import logging

logger = logging.getLogger(__name__)
HiddenArray: TypeAlias = np.ndarray
AudioInput: TypeAlias = torch.Tensor | np.ndarray


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

    @overload
    def transform(
        self, input_data: torch.Tensor, return_phase: Literal[False] = False
    ) -> torch.Tensor: ...

    @overload
    def transform(
        self, input_data: torch.Tensor, return_phase: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def transform(self, input_data: torch.Tensor, return_phase: bool = False):
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
        forward_basis = cast(torch.Tensor, self.forward_basis)
        forward_transform = torch.matmul(forward_basis, forward_transform)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
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
        inverse_basis = cast(torch.Tensor, self.inverse_basis)
        inverse_transform = torch.matmul(inverse_basis, cat)
        inverse_transform = fold(inverse_transform)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        window_square_sum = (
            cast(torch.Tensor, self.fft_window).pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0)
        )
        window_square_sum = fold(window_square_sum)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        inverse_transform /= window_square_sum
        return inverse_transform

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor):
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
        concat_tensors: list[torch.Tensor] = []
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

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]):
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
        n_blocks: int,
        n_gru: int,
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
                nn.Linear(3 * 128, 360), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel: torch.Tensor):
        # print(mel.shape)
        # mel = mel.transpose(-1, -2).unsqueeze(1)
        # x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        # x = x.contiguous()
        # x = self.fc(x)
        # print(x.shape)
        # return x
        # print("Initial mel shape:", mel.shape)
        mel = mel.transpose(-1, -2).unsqueeze(1)
        # print("Mel after transpose and unsqueeze:", mel.shape)

        unet_output = self.unet(mel)
        # print("Unet output shape:", unet_output.shape)

        x = self.cnn(unet_output)
        # print("CNN output shape:", x.shape)

        x = x.transpose(1, 2).flatten(-2)
        # print("After transpose and flatten:", x.shape)  # This is the input to GRU

        # x = x.contiguous()
        # print("After contiguous:", x.shape, x.is_contiguous())  # Verify contiguity

        x = self.fc(x)
        # print("Final x shape:", x.shape)
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

    def forward(
        self, audio: torch.Tensor, keyshift: float = 0, speed: float = 1, center: bool = True
    ) -> torch.Tensor:
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
        mel_output = torch.matmul(cast(torch.Tensor, self.mel_basis), magnitude)
        if self.is_half == True:
            try:
                mel_output = mel_output.half()
            except Exception:
                mel_output = mel_output.float()
                print(
                    "Warning: could not convert mel spectrogram to half — keeping float32."
                )
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    def __init__(
        self,
        model_path: str,
        is_half: bool,
        device: str | torch.device | None = None,
        use_jit: bool = False,
    ) -> None:
        self.resample_kernel = {}
        self.resample_kernel = {}
        self.is_half = is_half
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model: object
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
                device = self.device

            def get_jit_model() -> torch.jit.ScriptModule:
                jit_model_path = model_path.rstrip(".pth")
                jit_model_path += ".half.jit" if is_half else ".jit"
                reload = False
                ckpt = None
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
                        inputs_path=None,  # type: ignore[bad-argument-type]
                        save_path=jit_model_path,
                        device=device,
                        is_half=is_half,
                    )
                assert ckpt is not None
                model = torch.jit.load(BytesIO(ckpt["model"]), map_location=device)
                return model

            def get_default_model() -> E2E:
                model = E2E(4, 1, (2, 2))
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
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

            self.model = cast(nn.Module, self.model).to(device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor | np.ndarray:
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            if "privateuseone" in str(self.device):
                onnx_model = self.model
                onnx_input_name = onnx_model.get_inputs()[0].name  # type: ignore[attr-defined]
                onnx_outputs_names = onnx_model.get_outputs()[0].name  # type: ignore[attr-defined]
                hidden = cast(
                    np.ndarray,
                    onnx_model.run(  # type: ignore[attr-defined]
                    [onnx_outputs_names],
                    input_feed={onnx_input_name: mel.cpu().numpy()},
                    )[0],
                )
                return hidden[:, :n_frames]
            else:
                try:
                    mel = mel.half() if self.is_half else mel.float()
                except Exception:
                    mel = mel.float()
                    print(
                        "Warning: could not convert mel spectrogram to half — keeping float32."
                    )
                hidden = cast(torch.Tensor, cast(nn.Module, self.model)(mel))
                return hidden[:, :n_frames]

    def decode(self, hidden: np.ndarray, thred: float = 0.03) -> np.ndarray:
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0

    # def infer_from_audio(self, audio, thred=0.03):
    #     # torch.cuda.synchronize()
    #     # t0 = ttime()
    #     if not torch.is_tensor(audio):
    #         audio = torch.from_numpy(audio)
    #     mel = self.mel_extractor(
    #         audio.float().to(self.device).unsqueeze(0), center=True
    #     )
    #     # print(123123123,mel.device.type)
    #     # torch.cuda.synchronize()
    #     # t1 = ttime()
    #     hidden = self.mel2hidden(mel)
    #     # torch.cuda.synchronize()
    #     # t2 = ttime()
    #     # print(234234,hidden.device.type)
    #     if "privateuseone" not in str(self.device):
    #         hidden = hidden.squeeze(0).cpu().numpy()
    #     else:
    #         hidden = hidden[0]
    #     if self.is_half == True:
    #         hidden = hidden.astype("float32")

    #     f0 = self.decode(hidden, thred=thred)
    #     # torch.cuda.synchronize()
    #     # t3 = ttime()
    #     # print("hmvpe:%s\t%s\t%s\t%s"%(t1-t0,t2-t1,t3-t2,t3-t0))
    #     return f0
    def infer_from_audio(
        self,
        audio: AudioInput,
        thred: float = 0.03,
        chunk_size_seconds: int = 60,
        overlap_seconds: int = 2,
    ) -> np.ndarray:
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        # Calculate frame-based chunk and overlap sizes
        # Based on your mel_extractor's hop_length (160 samples at 16000 Hz)
        # hop_length_ms = 160 / 16000 * 1000 = 10 ms
        # So, 1 second = 100 frames

        # Convert chunk and overlap from seconds to mel frames
        sampling_rate = 16000  # Your mel_extractor's sampling rate
        hop_length = 160  # Your mel_extractor's hop_length

        chunk_frames = int(chunk_size_seconds * sampling_rate / hop_length)
        overlap_frames = int(overlap_seconds * sampling_rate / hop_length)

        all_hidden_parts = []

        # Pad audio to ensure all chunks can be processed, especially the last one
        # A more robust padding might be needed depending on how audio is handled
        audio_len = audio.size(0)

        # Simple padding to ensure full chunks. You might want to consider more sophisticated padding.
        if audio_len < chunk_frames * hop_length:
            # If audio is shorter than a single chunk, process it as is.
            mel = self.mel_extractor(
                audio.float().to(self.device).unsqueeze(0), center=True
            )
            hidden = self.mel2hidden(mel)
            if "privateuseone" not in str(self.device):
                hidden_np = cast(torch.Tensor, hidden).squeeze(0).cpu().numpy()
            else:
                hidden_np = cast(np.ndarray, hidden)[0]
            if self.is_half:
                hidden_np = hidden_np.astype("float32")
            f0 = self.decode(hidden_np, thred=thred)
            return f0

        # Determine indices for chunking
        start_sample = 0
        while start_sample < audio_len:
            end_sample = min(
                start_sample + chunk_size_seconds * sampling_rate, audio_len
            )

            # Take audio chunk
            current_audio_chunk = audio[start_sample:end_sample]

            # If the chunk is shorter than what's needed for the full mel extraction
            # (due to win_length), pad it or handle it.
            # Here, we'll ensure it's at least as long as win_length
            if len(current_audio_chunk) < self.mel_extractor.win_length:
                # Pad the end of the very last short chunk
                pad_needed = self.mel_extractor.win_length - len(current_audio_chunk)
                current_audio_chunk = F.pad(
                    current_audio_chunk, (0, pad_needed), mode="constant"
                )

            mel_chunk = self.mel_extractor(
                current_audio_chunk.float().to(self.device).unsqueeze(0), center=True
            )
            hidden_chunk = cast(
                torch.Tensor,
                self.mel2hidden(
                mel_chunk
                ),
            )  # Shape: (1, num_frames_in_chunk, 384)

            # Remove batch dimension
            hidden_chunk = hidden_chunk.squeeze(0)

            all_hidden_parts.append(hidden_chunk)

            # Move to the next chunk
            start_sample += (chunk_size_seconds - overlap_seconds) * sampling_rate
            if start_sample >= audio_len:  # Ensure we don't go past the end too early
                break

        # Combine the hidden parts with blending for overlaps
        # This is a simplified blending. For true overlap-add/blend,
        # you'd need to consider the weight of each segment.
        # For pitch, a simple concatenation might be acceptable, but blending
        # is safer if the model is sensitive to edge effects.
        combined_hidden = torch.cat(all_hidden_parts, dim=0)

        # Post-process to handle overlaps more gracefully
        # A common approach: for overlapping regions, average the predictions
        # from both chunks. This can be complex to implement perfectly for pitch.
        # For a first pass, simple concatenation might be okay if the overlap is large enough.
        # Let's refine the combining part to handle overlap explicitly:

        final_hidden_list = []
        current_idx_in_combined = 0

        # Add the first non-overlapping part
        first_chunk_non_overlap_frames = chunk_frames - overlap_frames
        final_hidden_list.append(all_hidden_parts[0][:first_chunk_non_overlap_frames])
        current_idx_in_combined += first_chunk_non_overlap_frames

        for i in range(1, len(all_hidden_parts)):
            prev_chunk = all_hidden_parts[i - 1]
            curr_chunk = all_hidden_parts[i]

            # Overlap region from previous chunk
            prev_overlap_part = prev_chunk[-overlap_frames:]
            # Overlap region from current chunk
            curr_overlap_part = curr_chunk[:overlap_frames]

            # Blend the overlap regions
            # Simple linear fade-in/fade-out for blending
            blend_weights_prev = torch.linspace(
                1, 0, overlap_frames, device=self.device
            ).unsqueeze(-1)
            blend_weights_curr = torch.linspace(
                0, 1, overlap_frames, device=self.device
            ).unsqueeze(-1)

            blended_overlap = (
                prev_overlap_part * blend_weights_prev
                + curr_overlap_part * blend_weights_curr
            )

            final_hidden_list.append(blended_overlap)

            # Add the non-overlapping part of the current chunk
            non_overlap_curr_part = curr_chunk[
                overlap_frames : chunk_frames - overlap_frames
            ]
            final_hidden_list.append(non_overlap_curr_part)
            current_idx_in_combined += chunk_frames - overlap_frames

        # Handle the very last chunk's remaining non-overlapping part if not fully covered
        # (This logic might need further adjustment based on how the last segment's
        # non-overlapping part is handled by the loop above; this is a common tricky part)
        if (
            len(all_hidden_parts) > 0
            and (audio_len % (sampling_rate * (chunk_size_seconds - overlap_seconds)))
            != 0
        ):
            last_chunk_remaining_frames = (
                audio_len
                - (
                    start_sample
                    - (chunk_size_seconds - overlap_seconds) * sampling_rate
                )
            ) // hop_length
            if last_chunk_remaining_frames > 0:
                final_hidden_list.append(
                    all_hidden_parts[-1][-last_chunk_remaining_frames:]
                )

        combined_hidden_final = torch.cat(final_hidden_list, dim=0)

        if "privateuseone" not in str(self.device):
            combined_hidden_np = combined_hidden_final.cpu().numpy()
        else:
            combined_hidden_np = cast(np.ndarray, combined_hidden_final)[
                0
            ]  # Assuming onnx returns batch dim
        if self.is_half:
            combined_hidden_np = combined_hidden_np.astype("float32")

        f0 = self.decode(combined_hidden_np, thred=thred)
        return f0

    def to_local_average_cents(self, salience: np.ndarray, thred: float = 0.05) -> np.ndarray:
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


if __name__ == "__main__":
    import librosa
    import soundfile as sf

    audio, sampling_rate = sf.read(r"C:\Users\liujing04\Desktop\Z\冬之花clip1.wav")
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    audio_bak = audio.copy()
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    model_path = r"D:\BaiduNetdiskDownload\RVC-beta-v2-0727AMD_realtime\rmvpe.pt"
    thred = 0.03  # 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmvpe = RMVPE(model_path, is_half=False, device=device)
    t0 = ttime()
    f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    t1 = ttime()
    logger.info("%s %.2f", f0.shape, t1 - t0)
