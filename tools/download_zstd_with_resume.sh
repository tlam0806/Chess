#!/bin/zsh
set -euo pipefail

if [[ $# -ne 2 ]]; then
  print -u2 "usage: $0 URL DESTINATION"
  exit 2
fi

url="$1"
destination="$2"
partial="${destination}.part"

/bin/mkdir -p "${destination:h}"

validate_zstd() {
  [[ -f "$1" ]] && /opt/homebrew/bin/zstd -q -t "$1"
}

while true; do
  if validate_zstd "$destination"; then
    break
  fi
  if [[ -e "$destination" ]]; then
    print -u2 "download_corrupt_complete=$destination"
    /bin/rm -f "$destination"
  fi

  if validate_zstd "$partial"; then
    /bin/mv "$partial" "$destination"
    break
  fi

  partial_bytes="$(/usr/bin/stat -f %z "$partial" 2>/dev/null || print 0)"
  print -u2 "download_start=$destination partial_bytes=$partial_bytes"
  if /usr/bin/curl \
      -fL \
      -C - \
      --output "$partial" \
      --retry 20 \
      --retry-all-errors \
      --retry-delay 15 \
      --connect-timeout 30 \
      --speed-limit 1024 \
      --speed-time 120 \
      "$url"; then
    if validate_zstd "$partial"; then
      /bin/mv "$partial" "$destination"
      break
    fi
    print -u2 "download_validation_failed=$destination restarting_from_zero"
    /bin/rm -f "$partial"
  else
    print -u2 "download_retry_cycle_failed=$destination sleeping=30"
    /bin/sleep 30
  fi
done

print -u2 "download_complete=$destination bytes=$(/usr/bin/stat -f %z "$destination")"
