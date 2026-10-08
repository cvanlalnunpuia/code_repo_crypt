from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus_split import build_split_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic matched-pair corpus split.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build_split_from_files(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Built {result['mode']} split for {result['matched_pair_count']} matched pairs")


if __name__ == "__main__":
    main()
