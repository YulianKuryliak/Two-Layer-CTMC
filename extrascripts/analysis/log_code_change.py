"""Append a code-change record with immutable source snapshots (no git commit)."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    paths = [Path(name) if Path(name).is_absolute() else ROOT / name for name in args.files]
    paths = [path.resolve() for path in paths]
    for path in paths:
        path.relative_to(ROOT)
        if not path.is_file():
            raise ValueError(f"Not a source file: {path}")
    folder = ROOT / "logs"
    folder.mkdir(exist_ok=True)
    with (folder / "code_changes.jsonl").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = {}
        for line in handle:
            for item in json.loads(line).get("files", []):
                previous[item["path"]] = item["sha256"]
        files = []
        for path in paths:
            contents = path.read_bytes()
            sha = hashlib.sha256(contents).hexdigest()
            relative = str(path.relative_to(ROOT))
            snapshot = folder / "code_snapshots" / sha / path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if snapshot.exists() and snapshot.read_bytes() != contents:
                raise RuntimeError(f"Corrupt source snapshot: {snapshot}")
            snapshot.write_bytes(contents)
            files.append({"path": relative, "previous_logged_sha256": previous.get(relative),
                          "sha256": sha, "snapshot": str(snapshot.relative_to(ROOT))})
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                text=True, capture_output=True)
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": "code_change",
               "message": args.message, "verification": args.verification,
               "git_head": commit.stdout.strip() if commit.returncode == 0 else None,
               "files": files,
               "note": "Hashes describe the working tree, including uncommitted edits. Null previous hash means no earlier logged snapshot; it does not mean a newly created file."}
        handle.seek(0, 2)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    print(f"Recorded {len(files)} files in logs/code_changes.jsonl")


if __name__ == "__main__":
    main()
