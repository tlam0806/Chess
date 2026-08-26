#!/usr/bin/env bash
set -euo pipefail

session="${1:?usage: runpod_autoterminate_after_tmux.sh SESSION POD_ID}"
pod_id="${2:?usage: runpod_autoterminate_after_tmux.sh SESSION POD_ID}"
poll_seconds="${POLL_SECONDS:-30}"

echo "watching_session=${session} pod_id=${pod_id} started=$(date -Is)"
while tmux has-session -t "${session}" 2>/dev/null; do
    sleep "${poll_seconds}"
done

echo "session_finished=$(date -Is)"
while [[ ! -s "${HOME}/.runpod/config.toml" ]]; do
    echo "waiting_for_runpod_api_config=$(date -Is)"
    sleep "${poll_seconds}"
done

echo "stopping_pod=${pod_id} at=$(date -Is)"
# Stop compute billing while preserving the Pod definition and its independent
# network volume. Deleting the Pod is unnecessary and makes recovery harder.
runpodctl stop pod "${pod_id}"
