from __future__ import annotations

import os
import sys
from pathlib import Path


def _ok(label: str, value: str) -> None:
    print(f"[OK] {label}: {value}")


def _warn(label: str, value: str) -> None:
    print(f"[WARN] {label}: {value}")


def _fail(label: str, value: str) -> None:
    print(f"[FAIL] {label}: {value}")


def main() -> int:
    print("Matplotlib smoke test started")

    try:
        import numpy as np

        _ok("numpy", np.__version__)
    except Exception as exc:
        _fail("numpy import", str(exc))
        return 1

    try:
        import matplotlib
        import matplotlib.pyplot as plt

        _ok("matplotlib", matplotlib.__version__)
        _ok("backend", matplotlib.get_backend())
    except Exception as exc:
        _fail("matplotlib import", str(exc))
        return 1

    try:
        import pandas as pd

        _ok("pandas", pd.__version__)
    except Exception as exc:
        _warn("pandas import", str(exc))

    try:
        import seaborn as sns

        _ok("seaborn", sns.__version__)
    except Exception as exc:
        _warn("seaborn import", str(exc))

    _ok("python", sys.version.split()[0])
    _ok("DISPLAY", os.environ.get("DISPLAY", "<not set>"))
    _ok("MPLBACKEND env", os.environ.get("MPLBACKEND", "<not set>"))

    out_path = Path("plots") / "matplotlib_smoke_test.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 2.0 * np.pi, 300)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y1, label="sin(x)", linewidth=2.0)
    ax.plot(x, y2, label="cos(x)", linewidth=2.0, linestyle="--")
    ax.set_title("Matplotlib Smoke Test")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()
    # plt.close(fig)

    if not out_path.exists():
        _fail("png output", f"file not created: {out_path}")
        return 1
    size = out_path.stat().st_size
    if size <= 0:
        _fail("png output", f"file is empty: {out_path}")
        return 1

    _ok("png output", f"{out_path} ({size} bytes)")
    print("Smoke test finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
