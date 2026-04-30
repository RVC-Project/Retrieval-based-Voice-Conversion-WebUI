import torch
import torch.nn as nn


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        momentum: float = 0.01,
    ):
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
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: tuple[int, int],
        n_blocks: int,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders

        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        concat_tensors: list[torch.Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class ResEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | None,
        n_blocks=1,
        momentum=0.01,
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size

        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))

        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)

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
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
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
        kernel_size: tuple[int, int],
        n_blocks: int,
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
