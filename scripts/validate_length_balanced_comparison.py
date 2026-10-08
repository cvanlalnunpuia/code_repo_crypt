from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyse_length_balanced_comparison import build, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the length-balanced corpus comparison.")
    parser.add_argument("--result", type=Path, default=Path("results/corpus/length-balanced-comparison.json"))
    parser.add_argument("--document", type=Path, default=Path("docs/length-balanced-comparison.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    rebuilt = build(root)
    stored = json.loads((root / args.result).read_text(encoding="utf-8"))
    if rebuilt != stored:
        raise SystemExit("Stored length-balanced comparison differs from recomputation")
    if (root / args.document).read_text(encoding="utf-8") != markdown(rebuilt):
        raise SystemExit("Stored length-balanced document differs from recomputation")
    print(f"Validated {len(rebuilt['comparisons'])} length-balanced comparisons")


if __name__ == "__main__":
    main()
