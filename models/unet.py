from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _make_norm(channels: int) -> nn.Module:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(inputs)
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([upsampled, skip], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = EncoderBlock(in_channels, c1, dropout)
        self.enc2 = EncoderBlock(c1, c2, dropout)
        self.enc3 = EncoderBlock(c2, c3, dropout)
        self.bridge = ConvBlock(c3, c4, dropout)
        self.dec3 = DecoderBlock(c4, c3, c3, dropout)
        self.dec2 = DecoderBlock(c3, c2, c2, dropout)
        self.dec1 = DecoderBlock(c2, c1, c1, dropout)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1, pooled1 = self.enc1(inputs)
        skip2, pooled2 = self.enc2(pooled1)
        skip3, pooled3 = self.enc3(pooled2)
        bridge = self.bridge(pooled3)
        decoded3 = self.dec3(bridge, skip3)
        decoded2 = self.dec2(decoded3, skip2)
        decoded1 = self.dec1(decoded2, skip1)
        return self.head(decoded1)


def build_unet(config: dict) -> UNet:
    model_cfg = config["model"]
    return UNet(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
        dropout=model_cfg["dropout"],
    )
