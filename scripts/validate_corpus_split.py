from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus_split import build_split_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate a deterministic corpus split.")
    parser.add_argument("split", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads(args.split.read_text(encoding="utf-8"))
    recomputed = build_split_from_files(args.config)
    if stored != recomputed:
        raise SystemExit("Stored split differs from independent recomputation")
    print(f"Validated {recomputed['mode']} split")


if __name__ == "__main__":
    main()
