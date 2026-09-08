#!/bin/zsh
set -eu

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

repo_root="${0:A:h:h:h}"
checkout="$repo_root/build/stockfish_static_nnue_upstream"
binary="$repo_root/build/stockfish_static_nnue_labeler"
commit="ebcea3efe9c1b8748e080111c727c33c544d7e06"
jobs="${STOCKFISH_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN)}"

if [[ ! -d "$checkout/.git" ]]; then
  git clone --filter=blob:none https://github.com/official-stockfish/Stockfish.git "$checkout"
fi
git -C "$checkout" fetch --depth 1 origin "$commit"
git -C "$checkout" checkout --detach "$commit"

cp "$repo_root/tools/data/stockfish_static_nnue_labeler.cpp" "$checkout/src/stockfish_static_nnue_labeler.cpp"

arch="${STOCKFISH_ARCH:-native}"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  arch="${STOCKFISH_ARCH:-apple-silicon}"
fi

sources='attacks.cpp benchmark.cpp bitboard.cpp evaluate.cpp misc.cpp movegen.cpp movepick.cpp position.cpp search.cpp thread.cpp timeman.cpp tt.cpp uci.cpp ucioption.cpp tune.cpp syzygy/tbprobe.cpp nnue/nnue_accumulator.cpp nnue/nnue_misc.cpp nnue/network.cpp nnue/features/half_ka_v2_hm.cpp nnue/features/full_threats.cpp engine.cpp score.cpp memory.cpp stockfish_static_nnue_labeler.cpp'

make -C "$checkout/src" -j "$jobs" build \
  ARCH="$arch" \
  EXE=stockfish_static_nnue_labeler \
  SRCS="$sources"

cp "$checkout/src/stockfish_static_nnue_labeler" "$binary"
chmod +x "$binary"

network="$checkout/src/nn-0ee0657fb25e.nnue"
print "binary=$binary"
print "stockfish_commit=$commit"
print "stockfish_arch=$arch"
shasum -a 256 "$binary"
shasum -a 256 "$network"
