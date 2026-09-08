#!/bin/zsh
set -eu

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h:h}"
cd "$repo_root"

output_dir="${LICHESS_POPULATION_OUTPUT:-$repo_root/data/lichess_bot_2300_2600_population_v1}"
python_bin="${LICHESS_POPULATION_PYTHON:-$repo_root/.venv/bin/python}"
collector="$repo_root/tools/data/collect_lichess_bot_population.py"
analyzer="$repo_root/tools/analyze/analyze_lichess_bot_population.py"
labeler="$repo_root/build/stockfish_static_nnue_labeler"
validator="$repo_root/build-release/validate_robotmoon_cbin"

[[ -x "$python_bin" ]] || { print -u2 "missing Python: $python_bin"; exit 1; }
[[ -x "$labeler" ]] || { print -u2 "missing Stockfish teacher labeler: $labeler"; exit 1; }
[[ -x "$validator" ]] || { print -u2 "missing CBin validator: $validator"; exit 1; }
command -v curl >/dev/null || { print -u2 "curl is required"; exit 1; }
command -v zstd >/dev/null || { print -u2 "zstd is required"; exit 1; }

mkdir -p "$output_dir"
exec > >(tee -a "$output_dir/run.log") 2>&1

common_args=(
  --output-dir "$output_dir"
  --min-elo 2300
  --max-elo 2600
  --min-base 60
  --max-base 600
  --min-increment 0
  --max-increment 10
  --target-positions 500000
  --min-accounts 100
  --account-position-cap 2000
  --random-seed 20260826
)

months=(
  2026-07
  2026-06
  2026-05
  2026-04
  2026-03
  2026-02
  2026-01
  2025-12
  2025-11
  2025-10
)

checkpoint() {
  "$python_bin" "$collector" "${common_args[@]}" --resume --export-only >/dev/null
  local cbin="$output_dir/positions.cbin"
  local labeled="$output_dir/positions-labeled.cbin"
  local labeled_tmp="$output_dir/positions-labeled.cbin.tmp"
  /bin/rm -f "$labeled_tmp"
  "$labeler" \
    --input "$cbin" \
    --output "$labeled_tmp" \
    --progress-interval 50000
  mv "$labeled_tmp" "$labeled"
  set +e
  "$python_bin" "$analyzer" \
    --labeled-cbin "$labeled" \
    --metadata "$output_dir/positions.jsonl" \
    --output "$output_dir/distribution.json" \
    --bootstrap-replicates 1000 \
    --batch-size 25000 \
    --stability-threshold 0.0025 \
    --require-stable
  analysis_status=$?
  set -e
  return "$analysis_status"
}

for month in "${months[@]}"; do
  if [[ -f "$output_dir/source-$month.done" ]]; then
    print "source_skip month=$month reason=completed_marker"
    continue
  fi
  current_positions="$($python_bin - "$output_dir/collector.sqlite3" <<'PY'
import sqlite3, sys
path = sys.argv[1]
try:
    connection = sqlite3.connect(path)
    print(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
except sqlite3.Error:
    print(0)
PY
)"
  if (( current_positions >= 500000 )); then
    break
  fi

  resume_args=()
  [[ -f "$output_dir/collector.sqlite3" ]] && resume_args=(--resume)
  source_id="lichess-standard-$month"
  source_url="https://database.lichess.org/standard/lichess_db_standard_rated_$month.pgn.zst"
  print "source_start month=$month positions=$current_positions"
  set +e
  curl --fail --location --silent --show-error "$source_url" \
    | zstd --decompress --stdout \
    | "$python_bin" "$collector" \
        "${common_args[@]}" \
        "${resume_args[@]}" \
        --database-pgn - \
        --database-source-id "$source_id" \
        --database-source-position-limit 50000 \
        --database-progress-interval 1000000
  pipeline_status=("${pipestatus[@]}")
  set -e
  collector_status="${pipeline_status[3]}"
  if [[ "$collector_status" != 0 && "$collector_status" != 2 ]]; then
    print -u2 "collector failed for $month with status $collector_status"
    exit "$collector_status"
  fi

  current_positions="$($python_bin - "$output_dir/collector.sqlite3" <<'PY'
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1])
print(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
PY
)"
  print "source_done month=$month positions=$current_positions"
  touch "$output_dir/source-$month.done"
  if (( current_positions < 200000 )); then
    continue
  fi
  if checkpoint; then
    print "stability_gate=pass positions=$current_positions"
    break
  fi
  print "stability_gate=continue positions=$current_positions"
done

final_positions="$($python_bin - "$output_dir/collector.sqlite3" <<'PY'
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1])
print(connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0])
PY
)"
if (( final_positions < 200000 )); then
  print -u2 "population stopped before the 200000-position minimum"
  exit 1
fi

checkpoint || true
validation_spool="$(mktemp -d /private/tmp/lichess-population-validation.XXXXXX)"
trap '/bin/rm -rf "$validation_spool"' EXIT
"$validator" \
  --input "$output_dir/positions-labeled.cbin" \
  --spool-directory "$validation_spool" \
  --expected-records "$final_positions" \
  > "$output_dir/cbin-validation.json"
print "population_complete positions=$final_positions output=$output_dir"
