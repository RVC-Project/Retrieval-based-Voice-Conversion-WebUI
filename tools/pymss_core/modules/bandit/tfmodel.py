import warnings

import torch
from torch import nn
from torch.nn.modules import rnn
from torch.utils.checkpoint import checkpoint_sequential


class TimeFrequencyModellingModule(nn.Module):
    pass


class ResidualRNN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        rnn_dim: int,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        use_batch_trick: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.norm = (
            nn.LayerNorm(emb_dim)
            if use_layer_norm
            else nn.GroupNorm(
                num_groups=emb_dim,
                num_channels=emb_dim,
            )
        )
        self.rnn = rnn.__dict__[rnn_type](
            input_size=emb_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            in_features=rnn_dim * (2 if bidirectional else 1),
            out_features=emb_dim,
        )
        self.use_batch_trick = use_batch_trick
        if not self.use_batch_trick:
            warnings.warn("NOT USING BATCH TRICK IS EXTREMELY SLOW!!")

    def forward(self, z):
        z0 = torch.clone(z)

        if self.use_layer_norm:
            z = self.norm(z)
        else:
            z = self.norm(z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        batch, n_uncrossed, n_across, emb_dim = z.shape

        if self.use_batch_trick:
            z = self.rnn(z.reshape(batch * n_uncrossed, n_across, emb_dim).contiguous())[0].reshape(
                batch, n_uncrossed, n_across, -1
            )
        else:
            z = torch.stack([self.rnn(z[:, i, :, :])[0] for i in range(n_uncrossed)], dim=1)

        return self.fc(z) + z0


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, z):
        return z.transpose(self.dim0, self.dim1)


class SeqBandModellingModule(TimeFrequencyModellingModule):
    def __init__(
        self,
        n_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        parallel_mode: bool = False,
        sequential_transpose: bool = False,
        checkpoint_segments: int | None = None,
    ) -> None:
        super().__init__()
        self.n_modules = n_modules
        self.parallel_mode = parallel_mode
        self.checkpoint_segments = checkpoint_segments

        if parallel_mode:
            self.seqband = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            ResidualRNN(
                                emb_dim=emb_dim,
                                rnn_dim=rnn_dim,
                                bidirectional=bidirectional,
                                rnn_type=rnn_type,
                            ),
                            ResidualRNN(
                                emb_dim=emb_dim,
                                rnn_dim=rnn_dim,
                                bidirectional=bidirectional,
                                rnn_type=rnn_type,
                            ),
                        ]
                    )
                    for _ in range(n_modules)
                ]
            )
        elif sequential_transpose:
            layers = []
            for _ in range(2 * n_modules):
                layers.extend(
                    [
                        ResidualRNN(
                            emb_dim=emb_dim,
                            rnn_dim=rnn_dim,
                            bidirectional=bidirectional,
                            rnn_type=rnn_type,
                        ),
                        Transpose(1, 2),
                    ]
                )
            self.seqband = nn.Sequential(*layers)
        else:
            self.seqband = nn.ModuleList(
                [
                    ResidualRNN(
                        emb_dim=emb_dim,
                        rnn_dim=rnn_dim,
                        bidirectional=bidirectional,
                        rnn_type=rnn_type,
                    )
                    for _ in range(2 * n_modules)
                ]
            )

    def forward(self, z):
        if self.parallel_mode:
            for sbm_t, sbm_f in self.seqband:
                zt = sbm_t(z)
                zf = sbm_f(z.transpose(1, 2))
                z = zt + zf.transpose(1, 2)
            return z

        if isinstance(self.seqband, nn.Sequential):
            if self.checkpoint_segments:
                return checkpoint_sequential(
                    self.seqband,
                    self.checkpoint_segments,
                    z,
                    use_reentrant=False,
                )
            return self.seqband(z)

        for sbm in self.seqband:
            z = sbm(z)
            z = z.transpose(1, 2)
        return z


class _SeqBandModellingPreset(SeqBandModellingModule):
    def __init__(
        self,
        n_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        parallel_mode: bool = False,
    ) -> None:
        super().__init__(
            n_modules=n_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            parallel_mode=parallel_mode,
            **self._preset_runtime_options(n_modules, parallel_mode),
        )

    @staticmethod
    def _preset_runtime_options(n_modules, parallel_mode):
        return {
            "sequential_transpose": False,
            "checkpoint_segments": None,
        }
