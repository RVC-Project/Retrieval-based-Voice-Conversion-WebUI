import torch

from .f0 import F0Predictor, FilterRadius, FloatArray


class FCPE(F0Predictor):
    def __init__(
        self,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        sampling_rate=44100,
        device="cpu",
    ):
        super().__init__(
            hop_length,
            f0_min,
            f0_max,
            sampling_rate,
            device,
        )

        from torchfcpe import (
            spawn_bundled_infer_model,
        )  # must be imported at here, or it will cause fairseq crash on training

        self.model = spawn_bundled_infer_model(self.device)

    def compute_f0(
        self,
        wav: FloatArray,
        p_len: int | None = None,
        filter_radius: FilterRadius = 0.006,
    ) -> FloatArray:
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        wav_tensor = torch.from_numpy(wav)
        threshold = float(filter_radius) if filter_radius is not None else 0.006
        f0 = (
            self.model.infer(
                wav_tensor.float().to(self.device).unsqueeze(0),
                sr=self.sampling_rate,
                decoder_mode="local_argmax",
                threshold=threshold,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]
