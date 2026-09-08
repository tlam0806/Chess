from __future__ import annotations

import argparse
import random
from pathlib import Path


def load_lines(path: Path, kind: str | None = None) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if kind is not None and f'"kind":"{kind}"' not in line:
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f"{path} produced no lines")
    return lines


def take_lines(lines: list[str], count: int, rng: random.Random) -> list[str]:
    if count <= 0:
        return []
    if count <= len(lines):
        return rng.sample(lines, count)
    result = list(lines)
    while len(result) < count:
        result.extend(rng.sample(lines, min(len(lines), count - len(result))))
    return result


def parse_source(spec: str) -> tuple[Path, int, str | None]:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"bad source spec: {spec}")
    path = Path(parts[0])
    count = int(parts[1])
    kind = parts[2] if len(parts) == 3 and parts[2] else None
    return path, count, kind


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix JSONL files deterministically")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="path:count or path:count:kind, where kind filters JSONL field kind",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    mixed: list[str] = []
    for spec in args.source:
        path, count, kind = parse_source(spec)
        lines = load_lines(path, kind)
        chosen = take_lines(lines, count, rng)
        mixed.extend(chosen)
        print(f"{path} kind={kind or '*'} requested={count} available={len(lines)}")

    rng.shuffle(mixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for line in mixed:
            output.write(line)
            output.write("\n")
    print(f"wrote {len(mixed)} lines to {args.output}")


if __name__ == "__main__":
    main()
