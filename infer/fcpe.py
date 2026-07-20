import torch

from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph


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
            if getattr(device, "type", None) == "cuda" or str(device).startswith(
                "cuda"
            ):
                # torchfcpe creates this tensor on CPU and copies it to CUDA in
                # every local-argmax decode.  Host-to-device copies are forbidden
                # during CUDA Graph capture, so keep the immutable offsets on the
                # target device and use the graph-safe equivalent decoder below.
                self.local_offsets = torch.arange(
                    9, device=device, dtype=torch.long
                ).view(1, 1, 9)

    def _graphable_model_infer(self, mel, decoder_mode, threshold):
        """Run FCPE network and an exactly equivalent capture-safe decoder."""
        model = self.infer_model.model
        latent = model(mel)
        batch, frames, _ = latent.shape
        cents = model.cent_table[None, None, :].expand(batch, frames, -1)

        if decoder_mode == "argmax":
            confidence = torch.max(latent, dim=-1, keepdim=True).values
            decoded = torch.sum(cents * latent, dim=-1, keepdim=True) / torch.sum(
                latent, dim=-1, keepdim=True
            )
        elif decoder_mode == "local_argmax":
            confidence, max_index = torch.max(latent, dim=-1, keepdim=True)
            local_index = self.local_offsets + (max_index - 4)
            local_index = local_index.clamp(0, model.out_dims - 1)
            local_cents = torch.gather(cents, -1, local_index)
            local_latent = torch.gather(latent, -1, local_index)
            decoded = torch.sum(
                local_cents * local_latent, dim=-1, keepdim=True
            ) / torch.sum(local_latent, dim=-1, keepdim=True)
        else:
            raise ValueError("Unknown FCPE decoder mode: %s" % decoder_mode)

        # Match torchfcpe's masking and cent-to-Hz formulas operation-for-operation.
        confidence_mask = torch.ones_like(confidence)
        confidence_mask.masked_fill_(confidence <= threshold, float("-inf"))
        decoded = decoded * confidence_mask
        return 10.0 * torch.pow(2.0, decoded / 1200.0)

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
            wav = wav.to(self.device)
            if cuda_graph_enabled(wav.device):
                # Wav2MelModule contains tensor-dependent Python conditionals and
                # cannot be captured.  Running it eagerly also creates/caches its
                # STFT window before the graph.  The much larger FCPE neural net
                # and decoder form the stable-shape CUDA Graph boundary.
                mel = self.infer_model.wav2mel(wav, sr)
                return run_cuda_graph(
                    self.infer_model.model,
                    "fcpe-core-%s-%s" % (decoder_mode, threshold),
                    lambda input_mel: self._graphable_model_infer(
                        input_mel,
                        decoder_mode,
                        threshold,
                    ),
                    mel,
                )
            return run_cuda_graph(
                self.infer_model,
                "fcpe-%s-%s-%s" % (sr, decoder_mode, threshold),
                lambda input_wav: self.infer_model.infer(
                    input_wav,
                    sr=sr,
                    decoder_mode=decoder_mode,
                    threshold=threshold,
                ),
                wav,
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
