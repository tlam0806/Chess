FROM ubuntu:24.04 AS engine-build

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY include ./include
COPY src ./src
COPY experiments ./experiments
COPY tools ./tools
COPY tests ./tests
RUN cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
       -DCHESS_ENABLE_LTO=ON \
    && cmake --build build-release --target uci_nnue_v43 -j 2

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LICHESS_BOT_DOCKER=true

WORKDIR /app
COPY lichess-bot ./lichess-bot
RUN python -m pip install --no-cache-dir -r /app/lichess-bot/requirements.txt

COPY --from=engine-build /src/build-release/uci_nnue_v43 \
    /app/chess-engine/build-release/uci_nnue_v43
COPY models /app/chess-engine/models
COPY tools/benchmark/benchmark_uci_platform.py \
    /app/chess-engine/tools/benchmark/benchmark_uci_platform.py
COPY tools/benchmark/nnue_v43_production_profile.py \
    /app/chess-engine/tools/benchmark/nnue_v43_production_profile.py
COPY tools/debug/test_uci_ponder_protocol.py \
    /app/chess-engine/tools/debug/test_uci_ponder_protocol.py
COPY benchmarks/uci_platform_v1.json \
    /app/chess-engine/benchmarks/uci_platform_v1.json
COPY release-manifest.json /app/chess-engine/release-manifest.json
COPY deploy/lichess/config-nnue-v43.yml /tmp/config-nnue-v43.yml
RUN sed 's#__ENGINE_ROOT__#/app/chess-engine#g' \
      /tmp/config-nnue-v43.yml \
      > /app/lichess-bot/config-nnue-v43.yml \
    && rm /tmp/config-nnue-v43.yml \
    && chmod +x \
      /app/chess-engine/build-release/uci_nnue_v43 \
      /app/chess-engine/tools/benchmark/benchmark_uci_platform.py \
      /app/chess-engine/tools/benchmark/nnue_v43_production_profile.py \
      /app/chess-engine/tools/debug/test_uci_ponder_protocol.py

WORKDIR /app/lichess-bot
CMD ["python", "lichess-bot.py", "--config", "config-nnue-v43.yml"]
