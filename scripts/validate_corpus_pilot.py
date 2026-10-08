from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus_matching import build_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and validate a matched corpus selection.")
    parser.add_argument("input", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads(args.selection.read_text(encoding="utf-8"))
    recomputed = build_selection(args.input, args.config)
    if stored != recomputed:
        raise SystemExit("Stored corpus selection differs from independent recomputation")
    identifiers = []
    languages = recomputed["config"]["languages"]
    for pair in recomputed["pairs"]:
        identifiers.extend(pair[language] for language in languages)
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("A record occurs in more than one selected pair")
    print(f"Validated {recomputed['selected_pair_count']} matched pairs")


if __name__ == "__main__":
    main()
