"""ResU-Net de la tesis, adaptada de HybridGNet (ngaggion).
Procedencia: https://github.com/ngaggion/HybridGNet/blob/main/models/unet.py
Código bajo GPL-3.0. se conserva el crédito a HybridGNet. Ver docs/THIRD_PARTY.md.
Modelo seleccionado: base=2, InstanceNorm, RGB, 323.981 parámetros.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(norm: str, num_channels: int) -> nn.Module:
    if norm == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    return nn.BatchNorm2d(num_channels, track_running_stats=False)


class ResBlock(nn.Module):
    """
    Residual block: BN→ReLU→Conv3x3→BN→ReLU→Conv3x3 + skip connection.

    Si in_channels != out_channels, el skip usa una Conv1x1+BN para proyectar.
    Diseño idéntico al `residualBlock` de HybridGNet/modelutils.py.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        self.block = nn.Sequential(
            _norm_layer(norm, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm_layer(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                _norm_layer(norm, out_channels),
            )
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.skip is not None else x
        return F.relu(self.block(x) + identity, inplace=True)


class ResUNet(nn.Module):
    """
    U-Net de 5 niveles con bloques residuales.

    Encoder:    5 ResBlocks + MaxPool2d
    Bottleneck: 1 ResBlock
    Decoder:    5 ResBlocks con bilinear upsample + skip concatenation
    Head:       Conv1x1 → num_classes

    Args:
        num_classes: Clases de salida (default 3: bg, iris, pupila).
        in_channels: Canales de entrada (1 = grayscale, 3 = RGB).
        base:        Factor de escala de canales (c en HybridGNet, default 8).
                     Canales por nivel: base*2, base*4, base*8, base*16, base*32.
        norm:        "batch" (default) o "instance" (para cross-domain LODO).
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        base: int = 8,
        norm: str = "batch",
    ):
        super().__init__()
        c = base
        s = [c * 2, c * 4, c * 8, c * 16, c * 32]  # [16, 32, 64, 128, 256] con base=8

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ResBlock(in_channels, s[0], norm)
        self.enc2 = ResBlock(s[0], s[1], norm)
        self.enc3 = ResBlock(s[1], s[2], norm)
        self.enc4 = ResBlock(s[2], s[3], norm)
        self.enc5 = ResBlock(s[3], s[4], norm)

        # Bottleneck
        self.bottleneck = ResBlock(s[4], s[4], norm)

        # Decoder
        self.dec5 = ResBlock(s[4] * 2, s[4], norm)
        self.dec4 = ResBlock(s[4] + s[3], s[3], norm)
        self.dec3 = ResBlock(s[3] + s[2], s[2], norm)
        self.dec2 = ResBlock(s[2] + s[1], s[1], norm)
        self.dec1 = ResBlock(s[1] + s[0], s[0], norm)

        # Head
        self.head = nn.Conv2d(s[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Bottleneck
        b = self.bottleneck(self.pool(e5))

        # Decoder
        def up(feat, skip):
            return torch.cat([F.interpolate(feat, skip.shape[2:], mode="bilinear", align_corners=False), skip], dim=1)

        d5 = self.dec5(up(b, e5))
        d4 = self.dec4(up(d5, e4))
        d3 = self.dec3(up(d4, e3))
        d2 = self.dec2(up(d3, e2))
        d1 = self.dec1(up(d2, e1))

        return self.head(d1)
