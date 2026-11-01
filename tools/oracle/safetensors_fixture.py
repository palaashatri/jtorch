#!/usr/bin/env python3
"""Generate a Safetensors golden file with the reference Python implementation."""

from __future__ import annotations

import json
import pathlib

import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "modules/jtorch-serialization/src/test/resources/golden"

tensors = {
    "weights": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    "bias": torch.tensor([0.5, -0.5], dtype=torch.float64),
    "labels": torch.tensor([1, 2, 3], dtype=torch.int64),
    "mask": torch.tensor([True, False, True], dtype=torch.bool),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT / "reference.safetensors"))
    expected = {
        name: {
            "dtype": {"torch.float32": "float32", "torch.float64": "float64",
                      "torch.int64": "int64", "torch.bool": "bool"}[str(t.dtype)],
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),
        }
        for name, t in tensors.items()
    }
    with open(OUT / "reference.json", "w", encoding="utf-8") as handle:
        json.dump({"tensors": expected}, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {OUT.relative_to(ROOT)}/reference.safetensors")


if __name__ == "__main__":
    main()
