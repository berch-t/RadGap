"""Smoke test d'environnement (M0) : GPU, versions, et forward AMP de démo.

Usage: uv run python scripts/check_env.py
DoD M0 : affiche le GPU, les versions clés, et fait un forward en mixed-precision.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=" * 60)
    print("RadGap — check_env")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")

    import torch

    print(f"torch       : {torch.__version__}")
    print(f"CUDA dispo  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA build  : {torch.version.cuda}")
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Capability  : sm_{cap[0]}{cap[1]}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM totale : {total:.1f} Go")
    else:
        print("⚠️  Pas de GPU CUDA visible — le forward AMP tournera sur CPU.")

    # Forward AMP de démo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = torch.nn.Linear(768, 10).to(device)
    x = torch.randn(8, 768, device=device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
        y = layer(x)
    print(f"Forward AMP : OK — sortie {tuple(y.shape)} sur {device}")

    # Versions des dépendances clés
    for mod in ("transformers", "open_clip", "torchxrayvision", "peft", "hydra"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{mod:<12}: {ver}")
        except ImportError:
            print(f"{mod:<12}: (non installé)")

    print("=" * 60)
    print("Environnement OK ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
