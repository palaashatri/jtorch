#!/usr/bin/env python3
"""Generate optimizer parity fixtures (SGD, momentum, Nesterov, Adam, AdamW).

Run from the repository root: ``python3 tools/oracle/optimizers.py``.
"""

from __future__ import annotations

import json
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "modules/jtorch-optim/src/test/resources/optimizers.json"


def run(name, optimizer_factory, initial, gradients, **kwargs):
    parameter = torch.tensor(initial, dtype=torch.float64).reshape(2, 2)
    parameter.requires_grad_(True)
    configuration = {"optimizer": name}
    configuration.update(kwargs)
    optimizer = optimizer_factory([parameter])
    steps = []
    for gradient in gradients:
        parameter.grad = torch.tensor(gradient, dtype=torch.float64).reshape(2, 2).clone()
        optimizer.step()
        steps.append(
            {
                "grad": list(gradient),
                "expected": parameter.detach().reshape(-1).tolist(),
            }
        )
    return {
        "name": name,
        "optimizer": name,
        "configuration": configuration,
        "initial": list(initial),
        "shape": [2, 2],
        "steps": steps,
    }


def main():
    initial = [1.0, 2.0, 3.0, 4.0]
    gradients = [
        [0.1, -0.2, 0.3, -0.4],
        [0.05, 0.05, -0.1, 0.2],
        [-0.3, 0.15, 0.25, -0.05],
    ]
    cases = [
        run("sgd", lambda p: torch.optim.SGD(p, lr=0.1), initial, gradients, learningRate=0.1),
        run(
            "sgd-momentum",
            lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.01),
            initial,
            gradients,
            learningRate=0.1,
            momentum=0.9,
            weightDecay=0.01,
        ),
        run(
            "sgd-nesterov",
            lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.8, nesterov=True),
            initial,
            gradients,
            learningRate=0.05,
            momentum=0.8,
            nesterov=True,
        ),
        run(
            "adam",
            lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.001),
            initial,
            gradients,
            learningRate=0.01,
            weightDecay=0.001,
        ),
        run(
            "adamw",
            lambda p: torch.optim.AdamW(p, lr=0.02, weight_decay=0.01),
            initial,
            gradients,
            learningRate=0.02,
            weightDecay=0.01,
        ),
    ]
    document = {"torch": torch.__version__, "cases": cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(cases)} optimizer cases to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
