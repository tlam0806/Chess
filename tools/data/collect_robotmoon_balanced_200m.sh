#!/bin/zsh
set -eu

repo_root="${0:A:h:h:h}"
cd "$repo_root"

output_dir="data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2_shards_1m"
manifest="data/robotmoon_balanced_cp_200m_rawmax5000_unique_v2.manifest.json"
local_jan="data/external/robotmoon/test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst"
base_url="https://huggingface.co/datasets/linrock/test80-2024/resolve/main"
remote_files=(
  test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst
  test80-2024-03-mar-2tb7p.min-v2.v6.binpack.zst
  test80-2024-04-apr-2tb7p.min-v2.v6.binpack.zst
  test80-2024-05-may-2tb7p.min-v2.v6.binpack.zst
  test80-2024-06-jun-2tb7p.min-v2.v6.binpack.zst
)

stream_sources() {
  /opt/homebrew/bin/zstd -dc "$local_jan"
  for name in "${remote_files[@]}"; do
    print -u2 "source_start=$name"
    curl -fL --retry 20 --retry-all-errors --connect-timeout 30 \
      "$base_url/$name" | /opt/homebrew/bin/zstd -dc
    print -u2 "source_done=$name"
  done
}

stream_sources \
  | nice -n 15 build/robotmoon_binpack_to_cbin \
      --input - \
      --output - \
      --deduplicate-positions \
      --dedup-expected-records 5000000000 \
      --dedup-bits-per-record 8 \
      --require-legal-move \
      --progress-interval 5000000 \
  | nice -n 15 .venv/bin/python tools/data/build_stratified_cp_corpus.py \
      --source - \
      --output-dir "$output_dir" \
      --manifest "$manifest" \
      --bins '0:100:50000000,100:300:50000000,300:600:40000000,600:1000:26000000,1000:1600:16000000,1600:2000:10000000,2000:5000:8000000' \
      --records-per-shard 1000000 \
      --zstd-level 6 \
      --seed 20260717
