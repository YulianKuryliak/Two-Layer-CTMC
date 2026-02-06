import json
from pathlib import Path
from typing import Any, Dict, Tuple


def resolve_path(path_like: str | Path, base_dir: Path | None = None) -> Path:
    normalized = str(path_like).replace("\\", "/")
    path = Path(normalized).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        base_dir = Path.cwd()
    return base_dir / path


def load_config(path: str | Path = "config.json") -> Tuple[Dict[str, Any], Path]:
    cfg_path = resolve_path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg, cfg_path.parent


__all__ = ["load_config", "resolve_path"]
