#!/usr/bin/env python3
"""Download NNUE checkpoints and stop a RunPod pod after a target epoch."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--remote-stop", action="store_true")
    parser.add_argument("--remote-terminate", action="store_true")
    parser.add_argument("--no-stop", action="store_true")
    parser.add_argument("--remote-log", required=True)
    parser.add_argument("--remote-output", required=True)
    parser.add_argument("--local-output", required=True, type=Path)
    parser.add_argument("--target-epoch", required=True, type=int)
    parser.add_argument("--poll-seconds", default=20, type=int)
    parser.add_argument("--max-hours", default=4.0, type=float)
    parser.add_argument("--log", required=True, type=Path)
    return parser.parse_args()


def append_log(path: Path, event: str, **fields: object) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def ssh_base(args: argparse.Namespace) -> list[str]:
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=accept-new", "-p", str(args.port),
        "-i", str(args.identity), args.host,
    ]


def completed_epochs(args: argparse.Namespace) -> list[int] | None:
    result = subprocess.run(
        [*ssh_base(args), "tail", "-n", "300", args.remote_log],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        append_log(args.log, "epoch_probe_retry", returncode=result.returncode)
        return None
    epochs: list[int] = []
    for line in result.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "epoch":
            epochs.append(int(payload["epoch"]))
    return epochs


def interrupt_training(args: argparse.Namespace) -> None:
    result = subprocess.run(
        [*ssh_base(args), "tmux", "send-keys", "-t", args.session, "C-c"],
        check=False,
    )
    append_log(args.log, "interrupt_sent", returncode=result.returncode)
    time.sleep(3)


def download(args: argparse.Namespace) -> None:
    args.local_output.mkdir(parents=True, exist_ok=True)
    ssh_transport = (
        f"ssh -o BatchMode=yes -o ConnectTimeout=15 "
        f"-i {args.identity} -p {args.port}"
    )
    subprocess.run(
        [
            "rsync", "-rtz", "--partial", "--no-owner", "--no-group",
            "--no-perms", "-e", ssh_transport,
            f"{args.host}:{args.remote_output.rstrip('/')}/",
            f"{args.local_output}/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync", "-rtz", "--partial", "--no-owner", "--no-group",
            "--no-perms", "-e", ssh_transport,
            f"{args.host}:{args.remote_log}",
            str(args.local_output / "train.runpod.log"),
        ],
        check=True,
    )
    required = (
        args.local_output / "phase_component_best.pt",
        args.local_output / "phase_component_current.pt",
        args.local_output / "train.runpod.log",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"download is missing required files: {missing}")
    append_log(
        args.log,
        "download_complete",
        output=str(args.local_output),
        best_bytes=required[0].stat().st_size,
        current_bytes=required[1].stat().st_size,
    )


def verify_download(args: argparse.Namespace) -> None:
    python = REPO_ROOT / ".venv/bin/python"
    if not python.is_file():
        raise RuntimeError(f"verification Python is missing: {python}")
    verifier = REPO_ROOT / "tools/train/verify_phase_training_checkpoint.py"
    common = [
        str(python), str(verifier), "--arch", "F2M",
        "--phase-layout", "independent", "--allow-missing-ranking",
    ]
    best = subprocess.run(
        [*common, "--checkpoint", str(args.local_output / "phase_component_best.pt")],
        text=True,
        capture_output=True,
        check=True,
    )
    current = subprocess.run(
        [
            *common,
            "--checkpoint", str(args.local_output / "phase_component_current.pt"),
            "--expected-epoch", str(args.target_epoch),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    append_log(
        args.log,
        "checkpoint_verification_passed",
        best=best.stdout.strip(),
        current=current.stdout.strip(),
    )


def stop_pod(args: argparse.Namespace) -> None:
    if args.no_stop:
        append_log(args.log, "pod_stop_pending", pod_id=args.pod_id)
        return
    if args.remote_stop or args.remote_terminate:
        operation = "remove" if args.remote_terminate else "stop"
        remote_command = (
            "set -a; source /etc/rp_environment; set +a; "
            f"runpodctl {operation} pod {shlex.quote(args.pod_id)}"
        )
        result = subprocess.run(
            [*ssh_base(args), "bash", "-lc", shlex.quote(remote_command)],
            text=True,
            capture_output=True,
            check=False,
        )
        append_log(
            args.log,
            (
                "pod_remote_terminate_requested"
                if args.remote_terminate
                else "pod_remote_stop_requested"
            ),
            pod_id=args.pod_id,
            returncode=result.returncode,
            response=result.stdout.strip()[:512],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"RunPod remote {operation} failed: "
                + result.stderr.strip()[:512]
            )
        return

    if args.api_key_file is None:
        raise RuntimeError("--api-key-file is required unless --remote-stop is set")
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
    args.api_key_file.unlink(missing_ok=True)


def finish(args: argparse.Namespace, reason: str) -> None:
    append_log(args.log, "finish_started", reason=reason)
    interrupt_training(args)
    download(args)
    verify_download(args)
    stop_pod(args)
    append_log(args.log, "finish_complete", reason=reason)


def main() -> None:
    args = parse_args()
    if args.target_epoch <= 0 or args.poll_seconds <= 0 or args.max_hours <= 0:
        raise ValueError("invalid epoch, poll interval, or timeout")
    stop_modes = sum(
        (
            args.remote_stop,
            args.remote_terminate,
            args.no_stop,
            args.api_key_file is not None,
        )
    )
    if stop_modes != 1:
        raise ValueError(
            "choose exactly one of --remote-stop, --remote-terminate, "
            "--no-stop, and --api-key-file"
        )
    append_log(
        args.log,
        "epoch_watcher_started",
        target_epoch=args.target_epoch,
        max_hours=args.max_hours,
        pod_id=args.pod_id,
    )
    deadline = time.monotonic() + args.max_hours * 3600.0
    while time.monotonic() < deadline:
        epochs = completed_epochs(args)
        if epochs and max(epochs) >= args.target_epoch:
            finish(args, f"epoch_{args.target_epoch}_complete")
            return
        time.sleep(args.poll_seconds)
    finish(args, "watcher_timeout")


if __name__ == "__main__":
    main()
