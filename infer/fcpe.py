import torch


def _is_directml_device(device):
    """Return whether *device* is the PrivateUse1 device registered by DirectML."""
    return getattr(device, "type", None) == "privateuseone" or "privateuseone" in str(
        device
    ).lower()


class FCPEInfer:
    """Project-local FCPE inference adapter with a DirectML execution path.

    DirectML does not support the complex tensor produced by ``torch.stft`` in
    torchfcpe's wav2mel stage.  Keep preprocessing and the small indexed
    decoder on CPU while the FCPE neural network runs on DirectML.  Other
    devices retain torchfcpe's original end-to-end inference path.
    """

    def __init__(self, device):
        from torchfcpe import spawn_bundled_infer_model

        self.device = device
        self.is_directml = _is_directml_device(device)
        if self.is_directml:
            # Loading a checkpoint directly with map_location=privateuseone is
            # not supported consistently.  Load on CPU, leave wav2mel there,
            # and move only the real-valued FCPE network to DirectML.
            self.infer_model = spawn_bundled_infer_model("cpu")
            self.infer_model.wav2mel.eval()
            self.cent_table_cpu = (
                self.infer_model.model.cent_table.detach().float().cpu().clone()
            )
            self.out_dims = int(self.infer_model.model.out_dims)
            self.infer_model.model.to(device).eval()
        else:
            self.infer_model = spawn_bundled_infer_model(device)

    def _decode_on_cpu(self, latent, decoder_mode, threshold):
        """Decode DML network logits on CPU with torchfcpe's exact formulas.

        The current DirectML backend's ``aten::gather`` returns incorrect FCPE
        bin values even though its indices and the neural-network logits match
        CPU.  Decoding is tiny compared with the model, so keep this
        compatibility boundary on CPU as well as the complex STFT.
        """
        latent = latent.detach().float().cpu()
        batch, frames, _ = latent.shape
        cents = self.cent_table_cpu[None, None, :].expand(batch, frames, -1)

        if decoder_mode == "argmax":
            confidence = torch.max(latent, dim=-1, keepdim=True).values
            decoded = torch.sum(cents * latent, dim=-1, keepdim=True) / torch.sum(
                latent, dim=-1, keepdim=True
            )
        elif decoder_mode == "local_argmax":
            confidence, max_index = torch.max(latent, dim=-1, keepdim=True)
            local_index = torch.arange(9, dtype=torch.long) + (max_index - 4)
            local_index.clamp_(0, self.out_dims - 1)
            local_cents = torch.gather(cents, -1, local_index)
            local_latent = torch.gather(latent, -1, local_index)
            decoded = torch.sum(
                local_cents * local_latent, dim=-1, keepdim=True
            ) / torch.sum(local_latent, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown FCPE decoder mode: {decoder_mode}")

        decoded = decoded.masked_fill(confidence <= threshold, float("-inf"))
        return 10.0 * torch.pow(2.0, decoded / 1200.0)

    # torch.no_grad is used instead of inference_mode because DirectML's
    # PrivateUse1 backend still updates version counters in a few operators.
    @torch.no_grad()
    def infer(
        self,
        wav,
        sr,
        decoder_mode="local_argmax",
        threshold=0.006,
    ):
        if not self.is_directml:
            return self.infer_model.infer(
                wav,
                sr=sr,
                decoder_mode=decoder_mode,
                threshold=threshold,
            )

        wav_cpu = wav.detach().to(device="cpu", dtype=torch.float32)
        mel_cpu = self.infer_model.wav2mel(wav_cpu, sr)
        mel_dml = mel_cpu.to(device=self.device, dtype=torch.float32)
        latent_dml = self.infer_model.model(mel_dml)
        return self._decode_on_cpu(
            latent_dml,
            decoder_mode=decoder_mode,
            threshold=threshold,
        )
