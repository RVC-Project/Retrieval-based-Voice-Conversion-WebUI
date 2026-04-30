import numpy as np
import torch
import torchcrepe

from .f0 import F0Predictor, FilterRadius, FloatArray


class CRePE(F0Predictor):
    def __init__(
        self,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        sampling_rate=44100,
        device="cpu",
    ):
        if "privateuseone" in str(device):
            device = "cpu"
        super().__init__(
            hop_length,
            f0_min,
            f0_max,
            sampling_rate,
            device,
        )

    def compute_f0(
        self,
        wav: FloatArray,
        p_len: int | None = None,
        filter_radius: FilterRadius = None,
    ) -> FloatArray:
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        wav_tensor = torch.from_numpy(wav)
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using device 'device'
        f0, pd = torchcrepe.predict(
            wav_tensor.float().to(self.device).unsqueeze(dim=0),
            self.sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]
