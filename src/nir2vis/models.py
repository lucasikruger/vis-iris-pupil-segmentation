"""
pix2pix generator (U-Net) and PatchGAN discriminator.

Reference: Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks" (CVPR 2017). Defaults match the original pix2pix
implementation but expose channel counts so smaller/larger variants can
be instantiated.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class UNetSkipBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: "UNetSkipBlock | None" = None,
        outermost: bool = False,
        innermost: bool = False,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up.append(nn.Dropout(0.5))
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UNetGenerator(nn.Module):
    """8-stage U-Net generator (256×256 → 256×256 by default).

    For grayscale NIR→VIS use input_nc=1, output_nc=1.
    """

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        num_downs: int = 8,
        ngf: int = 64,
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        block = UNetSkipBlock(ngf * 8, ngf * 8, submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            block = UNetSkipBlock(ngf * 8, ngf * 8, submodule=block, use_dropout=use_dropout)
        block = UNetSkipBlock(ngf * 4, ngf * 8, submodule=block)
        block = UNetSkipBlock(ngf * 2, ngf * 4, submodule=block)
        block = UNetSkipBlock(ngf, ngf * 2, submodule=block)
        self.model = UNetSkipBlock(output_nc, ngf, input_nc=input_nc, submodule=block, outermost=True)
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
