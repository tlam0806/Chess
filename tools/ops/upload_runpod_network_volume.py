#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.transferred = 0
        self.last_report = 0
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def __call__(self, amount: int) -> None:
        with self.lock:
            self.transferred += amount
            if self.transferred - self.last_report < 256 * 1024 * 1024 and self.transferred < self.total:
                return
            self.last_report = self.transferred
            elapsed = max(time.monotonic() - self.started, 1e-9)
            mib = self.transferred / (1024 * 1024)
            total_mib = self.total / (1024 * 1024)
            print(
                f"uploaded={mib:.1f}/{total_mib:.1f}MiB "
                f"progress={100.0 * self.transferred / self.total:.1f}% "
                f"speed={mib / elapsed:.1f}MiB/s",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload one bundle to a RunPod Network Volume")
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--datacenter", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--credentials-stdin",
        action="store_true",
        help="read one JSON line containing access_key and secret_key from stdin",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing object without uploading it again",
    )
    return parser.parse_args()


def remote_object_size(client: object, volume_id: str, key: str) -> int:
    try:
        remote = client.head_object(Bucket=volume_id, Key=key)
        return int(remote["ContentLength"])
    except ClientError as error:
        status = int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if status != 403:
            raise

    # Some RunPod S3 endpoints accept the upload but return 403 to HeadObject.
    # ListObjectsV2 is an independent, documented way to verify the exact path.
    response = client.list_objects_v2(Bucket=volume_id, Prefix=key)
    for item in response.get("Contents", []):
        if item.get("Key") == key:
            return int(item["Size"])
    raise RuntimeError(f"uploaded object is not visible: s3://{volume_id}/{key}")


def main() -> None:
    args = parse_args()
    if not args.file.is_file():
        raise SystemExit(f"file does not exist: {args.file}")
    if args.credentials_stdin:
        credentials = json.loads(sys.stdin.readline())
        access_key = credentials["access_key"]
        secret_key = credentials["secret_key"]
    else:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise SystemExit("S3 access key and secret key are required")

    size = args.file.stat().st_size
    client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=args.endpoint,
        region_name=args.datacenter,
        config=Config(
            signature_version="s3v4",
            retries={"mode": "standard", "max_attempts": 12},
            connect_timeout=30,
            read_timeout=7200,
        ),
    )
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=4,
        use_threads=True,
    )
    if not args.verify_only:
        client.upload_file(
            str(args.file),
            args.volume_id,
            args.key,
            Callback=Progress(size),
            Config=transfer,
        )
    remote_size = remote_object_size(client, args.volume_id, args.key)
    if remote_size != size:
        raise SystemExit(f"remote size mismatch: local={size} remote={remote_size}")
    print(f"upload_complete bytes={size} s3://{args.volume_id}/{args.key}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("upload interrupted", file=sys.stderr)
        raise SystemExit(130)
