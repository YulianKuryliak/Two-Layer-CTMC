#!/usr/bin/env python3
"""Private stdio MCP adapter for asynchronous, project-scoped Codex CLI jobs.

Python standard library only. Expose through an authenticated Secure MCP Tunnel,
never through a public unauthenticated shell/HTTP listener.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

PROJECT = Path(__file__).resolve().parents[2]
STATE = Path.home() / ".local/share/codex-web-bridge/jobs"
TERMINAL = {"completed", "failed", "cancelled", "interrupted", "timed_out"}
MAX_PROMPT = 32000
MAX_LOG_BYTES = 20 * 1024 * 1024
INSTRUCTIONS = (
    "Run tasks only when explicitly requested by the user. Use codex_submit with a "
    "new UUID request_id, then poll codex_status; never report completion from an "
    "accepted job alone. Reuse the request_id on retries. Default to read-only; set "
    "allow_project_edits only for user-authorized edits. Resume only a completed "
    "bridge job using previous_job_id. Honor project AGENTS.md, especially no "
    "simulation runs without explicit authorization. Do not send secrets."
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(data, handle, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def process_identity(pid):
    """Use Linux start ticks as well as PID, so PID reuse is not a live worker."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def tail(path, count=12000):
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - count))
            return handle.read(count).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


class Jobs:
    def __init__(self, project=PROJECT, state=STATE, codex=None, timeout=3600):
        self.project = Path(project).resolve(strict=True)
        self.state = Path(state).resolve()
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state, 0o700)
        self.codex = codex or shutil.which("codex")
        self.timeout = timeout

    @contextmanager
    def lock(self):
        with (self.state / "registry.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def folder(self, job_id):
        if not isinstance(job_id, str) or str(uuid.UUID(job_id)) != job_id:
            raise ValueError("job_id/request_id must be a canonical UUID")
        return self.state / job_id

    def load(self, job_id):
        data = json.loads((self.folder(job_id) / "job.json").read_text())
        if data["project"] != str(self.project):
            raise ValueError("Job belongs to a different project")
        return data

    def refresh(self, data):
        if data["status"] not in TERMINAL:
            pid = data.get("worker_pid")
            if pid and process_identity(pid) != data.get("worker_start"):
                data.update(status="interrupted", ended_at=utc(),
                            error="Worker exited or WSL restarted; no automatic rerun.")
                atomic_json(self.folder(data["job_id"]) / "job.json", data)
        return data

    def status(self, job_id):
        with self.lock():
            data = self.refresh(self.load(job_id))
        folder = self.folder(job_id)
        # Prompts and private local paths are omitted from returned metadata.
        result = {k: v for k, v in data.items() if k not in
                  {"prompt", "worker_pid", "worker_start", "command"}}
        result["answer"] = tail(folder / "answer.txt", 24000)
        result["progress_tail"] = tail(folder / "events.jsonl", 8000)
        result["stderr_tail"] = tail(folder / "stderr.log", 4000)
        result["output_is_bounded"] = True
        return result

    def recent(self):
        rows = []
        with self.lock():
            for path in sorted(self.state.glob("*/job.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True):
                data = json.loads(path.read_text())
                if data["project"] == str(self.project):
                    data = self.refresh(data)
                    rows.append({k: data.get(k) for k in
                                 ("job_id", "status", "created_at", "thread_id")})
                    if len(rows) == 20:
                        break
        return rows

    def submit(self, prompt, request_id, allow_project_edits=False, previous_job_id=None):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT:
            raise ValueError(f"prompt must contain 1..{MAX_PROMPT} characters")
        if type(allow_project_edits) is not bool:
            raise ValueError("allow_project_edits must be boolean")
        if not self.codex:
            raise ValueError("codex is not on PATH; sign in with codex login in WSL")
        folder = self.folder(request_id)
        with self.lock():
            if folder.exists():
                old = self.load(request_id)
                for key, value in {"prompt": prompt, "allow_project_edits": allow_project_edits,
                                   "previous_job_id": previous_job_id}.items():
                    if old.get(key) != value:
                        raise ValueError("request_id was already used for different arguments")
                return {"job_id": request_id, "status": self.refresh(old)["status"],
                        "deduplicated": True}
            thread = None
            if previous_job_id:
                previous = self.refresh(self.load(previous_job_id))
                if previous["status"] != "completed" or not previous.get("thread_id"):
                    raise ValueError("Only a successfully completed bridge job can be resumed")
                thread = previous["thread_id"]
            for path in self.state.glob("*/job.json"):
                other = json.loads(path.read_text())
                if other["project"] == str(self.project):
                    if self.refresh(other)["status"] not in TERMINAL:
                        raise ValueError(f"Project has active job {other['job_id']}; wait or cancel it")
            folder.mkdir(mode=0o700)
            job = {"job_id": request_id, "project": str(self.project), "prompt": prompt,
                   "allow_project_edits": allow_project_edits, "previous_job_id": previous_job_id,
                   "thread_id": thread, "status": "queued", "created_at": utc(),
                   "bridge_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            atomic_json(folder / "job.json", job)
            try:
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--project", str(self.project), "--state", str(self.state),
                           "--codex", self.codex, "--timeout", str(self.timeout),
                           "worker", request_id]
                with (folder / "worker.log").open("ab") as log:
                    worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=log, start_new_session=True,
                                              env=clean_environment())
                job.update(worker_pid=worker.pid, worker_start=process_identity(worker.pid))
                atomic_json(folder / "job.json", job)
                # Keep the child reaped while the MCP server lives. The worker itself
                # remains independent if the tunnel/stdin process exits.
                threading.Thread(target=worker.wait, daemon=True).start()
            except Exception as error:
                job.update(status="failed", ended_at=utc(), error=str(error))
                atomic_json(folder / "job.json", job)
                raise
        return {"job_id": request_id, "status": "queued", "deduplicated": False,
                "next": "Call codex_status; accepted does not mean completed."}

    def cancel(self, job_id):
        with self.lock():
            data = self.refresh(self.load(job_id))
            if data["status"] not in TERMINAL:
                # The worker terminates its own child, never an unverified arbitrary PID.
                (self.folder(job_id) / "cancel").touch(mode=0o600, exist_ok=True)
                return {"job_id": job_id, "status": "cancellation_requested"}
            return {"job_id": job_id, "status": data["status"]}

    def work(self, job_id):
        folder = self.folder(job_id)
        child = None
        final = "failed"
        error = None
        code = None
        try:
            with self.lock():
                data = self.load(job_id)
                if data["status"] != "queued":
                    raise ValueError("Worker can start only a queued job")
                data.update(status="running", started_at=utc())
                atomic_json(folder / "job.json", data)
            command = [self.codex, "--ask-for-approval", "never",
                       *permission_flags(data["allow_project_edits"]), "exec"]
            if data.get("thread_id"):
                command += ["resume", data["thread_id"]]
            # Ignore legacy user sandbox_mode (which overrides named profiles),
            # while retaining the existing account authentication and AGENTS.md.
            command += ["--ignore-user-config", "--json", "--output-last-message",
                        str(folder / "answer.txt"), "-"]
            prompt = (
                "This request was explicitly submitted through the user's private WSL bridge. "
                "Work in the current project and follow AGENTS.md. Never read bridge/tunnel "
                "credentials. No simulations, diagnostics or smoke simulations unless this "
                "request explicitly authorizes them. If interaction or unavailable permissions "
                "are needed, explain the blocker in your final answer.\n\n" + data["prompt"]
            )
            if (folder / "cancel").exists():
                final = "cancelled"
                return
            with (folder / "events.jsonl").open("wb") as events, \
                    (folder / "stderr.log").open("wb") as stderr, \
                    (folder / "input.txt").open("w+b") as stdin:
                stdin.write(prompt.encode()); stdin.seek(0)
                child = subprocess.Popen(command, cwd=self.project, stdin=stdin,
                                         stdout=events, stderr=stderr, start_new_session=True,
                                         env=clean_environment())
                started = time.monotonic()
                while child.poll() is None:
                    if (folder / "cancel").exists():
                        final = "cancelled"
                        break
                    if time.monotonic() - started >= self.timeout:
                        final = "timed_out"
                        break
                    if any(p.exists() and p.stat().st_size > MAX_LOG_BYTES for p in
                           (folder / "events.jsonl", folder / "stderr.log", folder / "answer.txt")):
                        error = "Per-job output limit reached"
                        break
                    time.sleep(0.2)
                if child.poll() is None:
                    terminate_group(child)
                else:
                    final = "completed" if child.returncode == 0 else "failed"
                code = child.returncode
        except BaseException as exc:
            error = str(exc)
        finally:
            if child is not None and child.poll() is None:
                terminate_group(child)
            with self.lock():
                data = self.load(job_id)
                for line in tail(folder / "events.jsonl", MAX_LOG_BYTES).splitlines():
                    try:
                        event = json.loads(line)
                        if event.get("type") == "thread.started":
                            data["thread_id"] = str(uuid.UUID(event["thread_id"]))
                    except (ValueError, KeyError, TypeError):
                        continue
                data.update(status=final, ended_at=utc(), exit_code=code)
                if error:
                    data["error"] = error
                atomic_json(folder / "job.json", data)


def permission_flags(allow_edits, extra_denied=()):
    home = Path.home()
    denied = [home / ".local/share/codex-web-bridge/runtime-key",
              home / ".local/state/tunnel-client", home / ".config/tunnel-client",
              Path(os.environ.get("CODEX_HOME", str(home / ".codex"))) / "auth.json",
              *extra_denied]
    filesystem = ",".join(json.dumps(str(path)) + '="deny"' for path in denied)
    parent = ":workspace" if allow_edits else ":read-only"
    profile = ('permissions.wsl_bridge={extends=' + json.dumps(parent)
               + ',filesystem={' + filesystem + '},network={enabled=false}}')
    return ["-c", 'default_permissions="wsl_bridge"', "-c", profile]


def clean_environment():
    # Keep account auth in Codex's host credential store. Do not hand ambient
    # API keys/tokens or arbitrary shell initialization settings to remote jobs.
    core = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TERM", "COLORTERM",
            "TZ", "TMPDIR", "TMP", "TEMP", "VIRTUAL_ENV", "CODEX_HOME",
            "SSL_CERT_FILE", "SSL_CERT_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
            "XDG_STATE_HOME", "XDG_DATA_HOME", "WSL_DISTRO_NAME"}
    return {k: v for k, v in os.environ.items() if k in core or k.startswith("LC_")}


def terminate_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def tool(name, description, properties=None, required=None, read_only=False):
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties or {},
                            "required": required or [], "additionalProperties": False},
            "annotations": {"readOnlyHint": read_only, "destructiveHint": not read_only,
                            "idempotentHint": True, "openWorldHint": not read_only}}


TOOLS = [
    tool("codex_info", "Report the fixed WSL project and bridge capabilities; runs no agent.",
         read_only=True),
    tool("codex_submit", "Submit a user-authorized task to Codex in WSL. Returns immediately. "
         "Use a stable UUID request_id for retries; poll codex_status for the result.",
         {"prompt": {"type": "string", "minLength": 1, "maxLength": MAX_PROMPT},
          "request_id": {"type": "string", "description": "New canonical UUID; reuse on retries."},
          "allow_project_edits": {"type": "boolean", "default": False},
          "previous_job_id": {"type": "string", "description": "Optional completed job to continue."}},
         ["prompt", "request_id"]),
    tool("codex_status", "Read job status, bounded output and final answer; never starts an agent.",
         {"job_id": {"type": "string"}}, ["job_id"], read_only=True),
    tool("codex_recent", "List up to 20 recent jobs in this WSL project.", read_only=True),
    tool("codex_cancel", "Request termination of a bridge job and its Codex process group.",
         {"job_id": {"type": "string"}}, ["job_id"]),
]


def call_tool(jobs, name, args):
    spec = next((t for t in TOOLS if t["name"] == name), None)
    if spec is None:
        raise ValueError("Unknown tool")
    if not isinstance(args, dict) or set(args) - set(spec["inputSchema"]["properties"]):
        raise ValueError("Unknown tool arguments")
    if set(spec["inputSchema"]["required"]) - set(args):
        raise ValueError("Missing required tool arguments")
    if name == "codex_info":
        return {"project": str(jobs.project), "host": os.uname().nodename,
                "codex_available": bool(jobs.codex), "timeout_seconds": jobs.timeout,
                "default_sandbox": "read-only", "parallel_project_jobs": 1,
                "instructions": INSTRUCTIONS}
    if name == "codex_submit":
        return jobs.submit(**args)
    if name == "codex_status":
        return jobs.status(**args)
    if name == "codex_recent":
        return {"jobs": jobs.recent()}
    return jobs.cancel(**args)


def serve(jobs):
    """Newline-delimited JSON-RPC stdio MCP; stdout contains protocol only."""
    for line in sys.stdin.buffer:
        request = None
        try:
            if len(line) > 256 * 1024:
                raise ValueError("Request too large")
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                raise ValueError("Expected a JSON-RPC 2.0 object")
            if "id" not in request:
                continue
            method = request.get("method")
            params = request.get("params") or {}
            if method == "initialize":
                offered = params.get("protocolVersion")
                version = offered if offered in {"2025-06-18", "2025-03-26", "2024-11-05"} else "2025-06-18"
                result = {"protocolVersion": version, "capabilities": {"tools": {}},
                          "serverInfo": {"name": "wsl-codex-bridge", "version": "1.0.0"},
                          "instructions": INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                try:
                    value = call_tool(jobs, params["name"], params.get("arguments", {}))
                    result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                              "isError": False}
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            else:
                reply = {"jsonrpc": "2.0", "id": request["id"],
                         "error": {"code": -32601, "message": "Method not supported"}}
                print(json.dumps(reply), flush=True)
                continue
            reply = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            reply = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                     "error": {"code": -32600, "message": str(exc)}}
        print(json.dumps(reply, ensure_ascii=False), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--codex", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("serve")
    sub.add_parser("info")
    worker = sub.add_parser("worker")
    worker.add_argument("job_id")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 86400:
        parser.error("timeout must be between 1 and 86400 seconds")
    jobs = Jobs(args.project, args.state, args.codex, args.timeout)
    if args.action == "worker":
        jobs.work(args.job_id)
    elif args.action == "serve":
        serve(jobs)
    else:
        print(json.dumps(call_tool(jobs, "codex_info", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
