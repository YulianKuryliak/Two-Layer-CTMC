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


def _strip_jsonc_comments(text: str) -> str:
    """
    Remove // line comments and /* block comments */ from JSONC text.
    Comment markers inside quoted strings are preserved.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escaped = False

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2 if i + 1 < n else 0
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def load_config(path: str | Path = "config.json") -> Tuple[Dict[str, Any], Path]:
    cfg_path = resolve_path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = f.read()
    cfg = json.loads(_strip_jsonc_comments(raw))
    return cfg, cfg_path.parent


__all__ = ["load_config", "resolve_path"]
