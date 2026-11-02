#!/usr/bin/env python3
"""Generate deterministic ResNet-class and ViT-class fixture weights and reference logits."""

from __future__ import annotations

import json
import pathlib

import torch
from torch import nn
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "modules/jtorch-vision/src/main/resources/fixtures"


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


class ResNet9(nn.Module):
    def __init__(self, width: int = 8, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(BasicBlock(width, width, 1), BasicBlock(width, width, 1))
        self.layer2 = nn.Sequential(
            BasicBlock(width, 2 * width, 2),
            BasicBlock(2 * width, 2 * width, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(2 * width, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


class ViT(nn.Module):
    def __init__(self, image_size: int, patch_size: int, dim: int, heads: int, depth: int,
                 mlp_dim: int, classes: int) -> None:
        super().__init__()
        self.patchEmbed = nn.Conv2d(3, dim, patch_size, patch_size, bias=True)
        self.classToken = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        tokens = (image_size // patch_size) ** 2
        self.posEmbed = nn.Parameter(torch.randn(1, tokens + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.blocks = nn.Sequential(layer)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchEmbed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.classToken.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.posEmbed
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def synthetic_batch(count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros(count, 3, 32, 32)
    labels = torch.randint(0, 4, (count,), generator=generator)
    for i in range(count):
        label = int(labels[i])
        if label < 3:
            images[i, label, :, :] = 1.0
            for channel in range(3):
                if channel != label:
                    images[i, channel, :, :] = 0.1
        else:
            for y in range(32):
                for x in range(32):
                    value = 1.0 if (x // 4 + y // 4) % 2 == 0 else 0.0
                    images[i, :, y, x] = value
        images[i] = images[i].add(torch.randn(3, 32, 32, generator=generator) * 0.02)
    return images, labels


def train_fixture() -> tuple[ResNet9, float]:
    generator = torch.Generator().manual_seed(20261104)
    model = ResNet9(width=8, classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(30):
        images, labels = synthetic_batch(64, generator)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        images, labels = synthetic_batch(64, generator)
        accuracy = (model(images).argmax(dim=1) == labels).float().mean().item()
    return model, accuracy


def save_model(name: str, model: nn.Module, input_tensor: torch.Tensor) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
    weights = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(weights, str(OUT / f"{name}-weights.safetensors"))
    save_file({"input": input_tensor, "logits": logits},
              str(OUT / f"{name}-io.safetensors"))
    top1 = logits.argmax(dim=1).tolist()
    return {
        "name": name,
        "inputShape": list(input_tensor.shape),
        "logitsShape": list(logits.shape),
        "top1": top1,
        "weights": f"{name}-weights.safetensors",
        "io": f"{name}-io.safetensors",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float32)

    torch.manual_seed(20261102)
    resnet, resnet_accuracy = train_fixture()
    resnet_input = torch.linspace(-1.0, 1.0, steps=2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    resnet_info = save_model("resnet9", resnet, resnet_input)
    resnet_info["fixtureAccuracy"] = resnet_accuracy

    torch.manual_seed(20261103)
    vit = ViT(image_size=16, patch_size=4, dim=16, heads=2, depth=1, mlp_dim=32, classes=10)
    vit_input = torch.linspace(-1.0, 1.0, steps=2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    vit_info = save_model("vit", vit, vit_input)

    metadata = {
        "labels": ["red", "green", "blue", "checker"],
        "models": [resnet_info, vit_info],
    }
    (OUT / "vision-fixtures.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote vision fixtures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
