#!/usr/bin/env bash
set -euo pipefail

# Prepare the deterministic 500M corpus on the Pod's ephemeral container disk,
# publish only the final shards to the persistent network volume, and resume
# the epoch-1 checkpoint on CUDA.

workspace="${WORKSPACE:-/workspace}"
repo_root="${REPO_ROOT:-$workspace/Chess}"
scratch="${SCRATCH_DIR:-/root/nnue-corpus-build}"
status_file="$repo_root/logs/runpod-bootstrap.status"
source_name="test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst"
source_url="https://huggingface.co/datasets/linrock/test80-2024/resolve/main/$source_name"
source_sha256="dd233aea41b23bc5fbecade91e78080d785472742f02d04e83d66192d634d9c7"
payload_sha256="7fe518a64491bcab4c17ce66b4aff7a4a815bf850b9e0a4187752c69a15cd891"
base_revision="9ae61602f3e297b12a780595187881f44292cdf3"
data_name="robotmoon_test80_2024_500m_unique_v2"
data_dir="$repo_root/data/${data_name}_shards_1m"
data_manifest="$repo_root/data/$data_name.manifest.json"

mkdir -p "$repo_root/logs" "$repo_root/data"

write_status() {
  printf 'state=%s updated_at=%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
}

on_exit() {
  code=$?
  if [[ $code -ne 0 ]]; then
    printf 'state=failed exit_code=%s updated_at=%s\n' \
      "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
  fi
}
trap on_exit EXIT

write_status preflight

if ! command -v git >/dev/null 2>&1 \
  || ! command -v cmake >/dev/null 2>&1 \
  || ! command -v zstd >/dev/null 2>&1 \
  || ! command -v wget >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ git make wget zstd
fi

# The bootstrap archive contains the exact dirty working tree used by epoch 1
# but intentionally omits .git. Create a local provenance commit so the corpus
# and training manifests remain self-describing even when the GitHub repository
# is private and the Pod has no GitHub credentials.
if [[ ! -d "$repo_root/.git" ]]; then
  git -C "$repo_root" init
fi
if ! git -C "$repo_root" rev-parse --verify HEAD >/dev/null 2>&1; then
  git -C "$repo_root" config user.name "RunPod NNUE bootstrap"
  git -C "$repo_root" config user.email "runpod-nnue@localhost"
  git -C "$repo_root" add \
    .gitignore CMakeLists.txt README.md compile_flags.txt context.md \
    deploy docs include nn src tests tests_py tools ui
  git -C "$repo_root" commit --no-gpg-sign -m \
    "RunPod bootstrap snapshot based on $base_revision"
fi

if [[ ! -d "$data_dir" || ! -f "$data_manifest" ]]; then
  write_status preparing_corpus
  mkdir -p "$scratch/source"

  wget -c -O "$scratch/source/$source_name" "$source_url"
  printf '%s  %s\n' "$source_sha256" "$scratch/source/$source_name" \
    | sha256sum -c -

  rm -rf \
    "$scratch/build" \
    "$scratch/$data_name.cbin.zst" \
    "$scratch/$data_name.conversion.log" \
    "$scratch/$data_name.validation.json" \
    "$scratch/$data_name.validation_spool" \
    "$scratch/${data_name}_shards_1m" \
    "$scratch/${data_name}_shards_1m.tmp" \
    "$scratch/$data_name.manifest.json" \
    "$scratch/$data_name.manifest.tmp.json"
  mkdir -p "$scratch/build"

  cmake -S "$repo_root" -B "$scratch/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$scratch/build" \
    --target robotmoon_binpack_to_cbin validate_robotmoon_cbin \
    -j "$(nproc)"

  cd "$repo_root"
  python3 tools/build_robotmoon_unique_corpus.py \
    --source "$scratch/source/$source_name" \
    --output-prefix "$scratch/$data_name" \
    --records 500000000 \
    --records-per-shard 1000000 \
    --dedup-bits-per-record 16 \
    --converter "$scratch/build/robotmoon_binpack_to_cbin" \
    --validator "$scratch/build/validate_robotmoon_cbin"

  actual_payload_sha256="$(python3 - "$scratch/$data_name.manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["output"]["payload_sha256"])
PY
)"
  if [[ "$actual_payload_sha256" != "$payload_sha256" ]]; then
    echo "corpus payload SHA-256 mismatch" >&2
    exit 1
  fi

  mv "$scratch/${data_name}_shards_1m" "$data_dir"
  mv "$scratch/$data_name.manifest.json" "$data_manifest"
  cp "$scratch/$data_name.conversion.log" "$repo_root/logs/"
  cp "$scratch/$data_name.validation.json" "$repo_root/logs/"
  rm -rf "$scratch"
fi

write_status training
cd "$repo_root"
bash tools/run_old_score_huber200_500m_runpod_resume.sh
write_status complete
trap - EXIT
