FROM ubuntu:24.04 AS engine-build

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY tools ./tools
RUN cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build-release --target uci_nnue_v38 -j 2

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LICHESS_BOT_DOCKER=true

WORKDIR /app
COPY lichess-bot ./lichess-bot
RUN python -m pip install --no-cache-dir -r /app/lichess-bot/requirements.txt

COPY --from=engine-build /src/build-release/uci_nnue_v38 \
    /app/chess-engine/build-release/uci_nnue_v38
COPY models /app/chess-engine/models
COPY deploy/lichess/config-nnue-v38.yml /tmp/config-nnue-v38.yml
RUN sed 's#__ENGINE_ROOT__#/app/chess-engine#g' \
      /tmp/config-nnue-v38.yml \
      > /app/lichess-bot/config-nnue-v38.yml \
    && rm /tmp/config-nnue-v38.yml \
    && chmod +x /app/chess-engine/build-release/uci_nnue_v38

WORKDIR /app/lichess-bot
CMD ["python", "lichess-bot.py", "--config", "config-nnue-v38.yml"]
