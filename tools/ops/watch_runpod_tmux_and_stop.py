#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stop a RunPod Pod after a remote tmux session exits"
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-hours", type=float, default=36.0)
    parser.add_argument("--log", required=True, type=Path)
    return parser.parse_args()


def append_log(path: Path, event: str, **fields: object) -> None:
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def session_probe(args: argparse.Namespace) -> int:
    result = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(args.port), "-i", str(args.identity), args.host,
            "tmux", "has-session", "-t", args.session,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode


def stop_pod(args: argparse.Namespace) -> None:
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("empty RunPod API key")
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{args.pod_id}/stop",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            body = response.read(512).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        body = error.read(512).decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod stop failed: HTTP {error.code}: {body}") from error
    if status != 200:
        raise RuntimeError(f"RunPod stop returned HTTP {status}: {body}")
    append_log(args.log, "pod_stopped", pod_id=args.pod_id, http_status=status)


def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    append_log(args.log, "watcher_started", pod_id=args.pod_id,
               session=args.session, max_hours=args.max_hours)
    deadline = time.monotonic() + args.max_hours * 3600.0
    seen_session = False
    consecutive_missing = 0
    while time.monotonic() < deadline:
        code = session_probe(args)
        if code == 0:
            if not seen_session:
                append_log(args.log, "session_seen")
            seen_session = True
            consecutive_missing = 0
        elif code == 1 and seen_session:
            consecutive_missing += 1
            # Require two probes to avoid stopping on a transient tmux-server race.
            if consecutive_missing >= 2:
                append_log(args.log, "session_finished")
                stop_pod(args)
                args.api_key_file.unlink(missing_ok=True)
                return
        else:
            append_log(args.log, "probe_retry", returncode=code)
        time.sleep(args.poll_seconds)
    append_log(args.log, "watcher_timeout")
    stop_pod(args)
    args.api_key_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
