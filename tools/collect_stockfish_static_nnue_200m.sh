#!/bin/zsh
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h}"
cd "$repo_root"

eligible_dir="data/robotmoon_balanced_cp_200m_sf_static_eligible_v1_shards_1m"
eligible_manifest="data/robotmoon_balanced_cp_200m_sf_static_eligible_v1.manifest.json"
labeled_dir="data/stockfish_static_nnue_psqtpos_200m_balanced_v1_shards_1m"
label_manifest="data/stockfish_static_nnue_psqtpos_200m_balanced_v1.manifest.json"
validation_report="data/stockfish_static_nnue_psqtpos_200m_balanced_v1.validation.json"
validation_spool="data/stockfish_static_nnue_psqtpos_200m_balanced_v1.validation_spool"
local_jan="data/external/robotmoon/test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst"
base_url="https://huggingface.co/datasets/linrock/test80-2024/resolve/main"
download_cache="data/external/robotmoon/download_cache"
stockfish_commit="ebcea3efe9c1b8748e080111c727c33c544d7e06"
stockfish_network="build/stockfish_static_nnue_upstream/src/nn-0ee0657fb25e.nnue"
label_workers="${STOCKFISH_LABEL_WORKERS:-5}"
remote_files=(
  test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst
  test80-2024-03-mar-2tb7p.min-v2.v6.binpack.zst
  test80-2024-04-apr-2tb7p.min-v2.v6.binpack.zst
  test80-2024-05-may-2tb7p.min-v2.v6.binpack.zst
  test80-2024-06-jun-2tb7p.min-v2.v6.binpack.zst
)

mkdir -p "$download_cache"

for output_path in "$eligible_dir" "$eligible_manifest" "$labeled_dir" "$label_manifest" "$validation_report"; do
  if [[ -e "$output_path" ]]; then
    print -u2 "refusing to overwrite $output_path"
    exit 1
  fi
done

/opt/homebrew/Cellar/cmake/4.2.0/bin/cmake --build build \
  --target robotmoon_binpack_to_cbin validate_robotmoon_cbin -j 4
STOCKFISH_BUILD_JOBS=4 tools/build_stockfish_static_nnue_labeler.sh

stream_sources() {
  /opt/homebrew/bin/zstd -dc "$local_jan"
  for name in "${remote_files[@]}"; do
    print -u2 "source_start=$name"
    tools/download_zstd_with_resume.sh \
      "$base_url/$name" \
      "$download_cache/$name" 1>&2
    /opt/homebrew/bin/zstd -dc "$download_cache/$name"
    /bin/rm -f "$download_cache/$name"
    print -u2 "source_done=$name"
  done
}

stream_sources \
  | /usr/bin/nice -n 15 build/robotmoon_binpack_to_cbin \
      --input - \
      --output - \
      --deduplicate-positions \
      --dedup-expected-records 5000000000 \
      --dedup-bits-per-record 8 \
      --require-legal-move \
      --exclude-in-check \
      --progress-interval 5000000 \
  | /usr/bin/nice -n 15 .venv/bin/python tools/build_stratified_cp_corpus.py \
      --source - \
      --output-dir "$eligible_dir" \
      --manifest "$eligible_manifest" \
      --bins '0:100:50000000,100:300:50000000,300:600:40000000,600:1000:26000000,1000:1600:16000000,1600:2000:10000000,2000:5000:8000000' \
      --records-per-shard 1000000 \
      --zstd-level 6 \
      --seed 20260719

.venv/bin/python tools/label_stockfish_static_nnue_shards.py \
  --source-dir "$eligible_dir" \
  --output-dir "$labeled_dir" \
  --manifest "$label_manifest" \
  --labeler build/stockfish_static_nnue_labeler \
  --workers "$label_workers" \
  --records-per-shard 1000000 \
  --zstd-level 6 \
  --stockfish-commit "$stockfish_commit" \
  --network "$stockfish_network"

.venv/bin/python tools/stream_cbin_shards.py --input "$labeled_dir" \
  | build/validate_robotmoon_cbin \
      --input - \
      --spool-directory "$validation_spool" \
      --expected-records 200000000 \
      --progress-interval 10000000 \
      > "$validation_report"

print "done labeled_dir=$labeled_dir"
print "label_manifest=$label_manifest"
print "validation_report=$validation_report"
