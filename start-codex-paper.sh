#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC"

cd "$PROJECT_DIR"

if [ ! -d ".codex" ]; then
  echo "ERROR: .codex is not a directory. Create it before starting Codex." >&2
  exit 1
fi

if ! (tmp=".codex/.write_test.$$"; printf ok > "$tmp" && rm "$tmp") 2>/dev/null; then
  echo "WARNING: .codex is not writable from this shell. Project custom-agent files may be read-only." >&2
fi

exec codex \
  -C "$PROJECT_DIR" \
  -s danger-full-access \
  -a on-request \
  --enable multi_agent_v2 \
  "$@"
