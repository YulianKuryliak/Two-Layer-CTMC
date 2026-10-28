#!/usr/bin/env python3
"""Set up a private Secure MCP Tunnel to the local Codex job bridge (WSL only)."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
STATE = Path.home() / ".local/share/codex-web-bridge"
ALIAS = "two-layer-ctmc-wsl"
TUNNELS_URL = "https://platform.openai.com/settings/organization/tunnels"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "codex-web-bridge-setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def tunnel_binary(install=False):
    local = Path.home() / ".local/bin/tunnel-client"
    receipt = STATE / "tunnel_install.json"
    if local.is_file():
        if not receipt.is_file():
            raise SystemExit("Existing tunnel-client has no verified install receipt; inspect it locally first")
        expected = json.loads(receipt.read_text())["binary_sha256"]
        if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            raise SystemExit("Managed tunnel-client SHA256 changed; refusing an unverified executable")
        return str(local)
    if not install:
        raise SystemExit("Run manage.py install-tunnel or manage.py connect first")
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
    if sys.platform != "linux" or not arch:
        raise SystemExit("This setup supports Linux/WSL amd64 and arm64")
    release = json.loads(fetch("https://api.github.com/repos/openai/tunnel-client/releases/latest"))
    name = f"tunnel-client-{release['tag_name']}-linux-{arch}.zip"
    asset = next(a for a in release["assets"] if a["name"] == name)
    url = asset["browser_download_url"]
    if not url.startswith("https://github.com/openai/tunnel-client/releases/download/"):
        raise SystemExit("Unexpected release download origin")
    blob = fetch(url)
    digest = hashlib.sha256(blob).hexdigest()
    if asset.get("digest") != "sha256:" + digest:
        raise SystemExit("Missing or mismatched official release SHA256; binary not installed")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        entries = [n for n in archive.namelist() if Path(n).name == "tunnel-client"]
        if len(entries) != 1:
            raise SystemExit("Unexpected release archive")
        binary = archive.read(entries[0])
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("xb") as handle:
        handle.write(binary)
    local.chmod(0o755)
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    (STATE / "tunnel_install.json").write_text(json.dumps({
        "version": release["tag_name"], "release": release["html_url"], "asset": name,
        "archive_sha256": digest, "binary_sha256": hashlib.sha256(binary).hexdigest(),
    }, indent=2) + "\n")
    print(f"Installed {local}; official release SHA256 verified")
    return str(local)


def connect(tunnel_id=None):
    binary = tunnel_binary(install=True)
    if not shutil.which("codex"):
        raise SystemExit("Codex CLI must be on PATH inside WSL")
    subprocess.run(["codex", "login", "status"], check=True)
    print("Create or select a tunnel at " + TUNNELS_URL)
    print("Associate the intended ChatGPT workspace; runtime key needs Tunnels Read + Use.")
    tunnel_id = tunnel_id or input("Tunnel ID (tunnel_...): ").strip()
    if not re.fullmatch(r"tunnel_[A-Za-z0-9_-]+", tunnel_id):
        raise SystemExit("Invalid tunnel ID")
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE, 0o700)
    key_file = STATE / "runtime-key"
    if not key_file.exists():
        print(f"The runtime key will be stored only in {key_file} (0600, outside the repository).")
        if not sys.stdin.isatty():
            raise SystemExit("Enter the key in an interactive WSL terminal; noninteractive input is refused")
        secret = getpass.getpass("Runtime API key (hidden): ").strip()
        if not secret or "\n" in secret:
            raise SystemExit("Empty or invalid runtime key")
        with key_file.open("x", encoding="utf-8") as handle:
            os.chmod(key_file, 0o600)
            handle.write(secret)
        del secret
    else:
        if key_file.is_symlink() or key_file.stat().st_uid != os.getuid():
            raise SystemExit("Refusing runtime key with unexpected owner or symlink")
        os.chmod(key_file, 0o600)
        print("Reusing the existing private runtime key file.")
    mcp_command = shlex.join([sys.executable, str(HERE / "bridge.py"), "serve"])
    command = [binary, "runtimes", "connect", "--alias", ALIAS,
               "--tunnel-id", tunnel_id, "--mcp-command", mcp_command,
               "--runtime-api-key", "file:" + str(key_file)]
    subprocess.run(command, check=True)
    checked = subprocess.run([binary, "runtimes", "status", ALIAS, "--json"], check=True,
                             capture_output=True, text=True)
    # Do not infer readiness from a zero exit status; show the actual runtime report.
    print(checked.stdout)
    print("Connect ChatGPT Plugins > Add custom MCP server > Connection: Tunnel.")
    print("Choose the same tunnel. Authentication: None at the stdio server layer;")
    print("access is controlled by your private tunnel's organization/workspace permissions.")
    print("Install the plugin, open a new Work chat and ask it to call codex_info.")
    print("End-to-end setup is complete only after that ChatGPT tool call succeeds.")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("install-tunnel")
    command = sub.add_parser("connect")
    command.add_argument("--tunnel-id")
    sub.add_parser("status")
    sub.add_parser("stop")
    sub.add_parser("check")
    args = parser.parse_args()
    if args.action == "install-tunnel":
        print(tunnel_binary(install=True))
    elif args.action == "connect":
        connect(args.tunnel_id)
    elif args.action in {"status", "stop"}:
        operation = "status" if args.action == "status" else "stop"
        subprocess.run([tunnel_binary(), "runtimes", operation, ALIAS], check=True)
    else:
        subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE),
                        "-p", "test_*.py", "-v"], check=True)


if __name__ == "__main__":
    main()
