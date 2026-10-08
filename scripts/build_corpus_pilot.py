from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus_matching import build_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic matched bilingual corpus selection.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build_selection(args.input, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Selected {result['selected_pair_count']} matched pairs")


if __name__ == "__main__":
    main()
